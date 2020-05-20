#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : estimator.py                                                       #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 7:15:36 pm                          #
# Last Modified : Sunday, March 15th 2020, 7:15:46 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Gradient Descent base class, from which regression and classification inherit."""
from abc import ABC, abstractmethod, ABCMeta
import sys
import copy
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.testing import ignore_warnings
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from mlstudio.supervised.callbacks.base import CallbackList, Callback
from mlstudio.supervised.callbacks.debugging import GradientCheck
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.learning_rate import Constant
from mlstudio.supervised.callbacks.monitor import BlackBox, Progress, summary
from mlstudio.supervised.core.activation import Sigmoid, Softmax
from mlstudio.supervised.core.optimizer import Standard
from mlstudio.supervised.core.cost import MSE, CrossEntropy, CategoricalCrossEntropy
from mlstudio.supervised.core.scorers import R2, Accuracy
from mlstudio.utils.data_manager import batch_iterator, data_split, shuffle_data

# --------------------------------------------------------------------------- #
#                          GRADIENT DESCENT                                   #
# --------------------------------------------------------------------------- #
class GradientDescent(ABC, BaseEstimator):
    """Base class gradient descent estimator."""

    def __init__(self):
        raise NotImplementedError("GradientDescent base class can not be\
             instantiated.")

    @property
    def variant(self):
        if self.batch_size is None:
            variant = 'Batch Gradient Descent'
        elif self.batch_size == 1:
            variant = 'Stochastic Gradient Descent'
        else:
            variant = 'Minibatch Gradient Descent'
        return variant

    @abstractmethod
    def description(self):
        pass

    @property
    def eta(self):
        return self._eta

    @eta.setter  
    def eta(self, x):
        self._eta = x

    @property
    def converged(self):
        return self._converged

    @converged.setter
    def converged(self, x):
        self._converged = x

    def _handle_no_validation_data(self):
        """Issues warning if validation set could not be created due to training set size."""
        # If no validation data (training samples < 2), then warn user
        # and change the cross-validation metric to train-cost.
        if self.X_val_ is None:            
            if 'val' in self.early_stop.metric:
                self.early_stop.metric = "train_cost"         
                warnings.warn("Validation set could not be created. Training set \
                    only has {s} observations. Changing cross-validation metric \
                        to 'train_cost'.\
                        ".format(s=str(self.X_train_.shape[0])), UserWarning)        
            else:
                warnings.warn("Validation set could not be created. Training set \
                    only has {s} observations.".format(s=str(self.X_train_.shape[0])), UserWarning)                        

    @abstractmethod
    def _prepare_training_data(self, X, y):
        """Prepares X and y data for training."""
        self.X_train_ = self.X_val_ = self.y_train_ = self.y_val_ = None
        # Validate inputs
        self.X_train_, self.y_train_ = check_X_y(X, y, accept_sparse=['csr'], 
                                                 accept_large_sparse=True,
                                                 estimator=self)
        # Add a column of ones to create the X design matrix                                         
        if sparse.issparse(self.X_train_):         
            # If COO matrix, convert to CSR
            if sparse.isspmatrix_coo(self.X_train_):
                self.X_train_ = self.X_train_.tocsr()                        
            ones = np.ones((self.X_train_.shape[0],1))
            bias_term = sparse.csr_matrix(ones, dtype=float)
            self.X_train_ = sparse.hstack((bias_term, self.X_train_))
        else:
            self.X_train_ = np.insert(self.X_train_, 0, 1.0, axis=1)

        # If y is COO convert to CSR
        if sparse.isspmatrix_coo(self.y_train_):
            self.y_train_ = self.y_train_.tocsr()

        self.n_features_ = self.X_train_.shape[1]      

    @abstractmethod
    def _prepare_test_data(self, X, y=None):
        """Prepares test data for prediction and scoring."""
        X = check_array(X, accept_sparse=['csr'], 
                           accept_large_sparse=True,
                           estimator=self)
        # Add a column of ones to create the X design matrix
        if sparse.issparse(X):         
        # If COO matrix, convert to CSR
            if sparse.isspmatrix_coo(X):
                X = X.tocsr()            
            # Add bias term 
            ones = np.ones((X.shape[0],1))
            bias_term = sparse.csr_matrix(ones, dtype=float)
            X = sparse.hstack((bias_term, X))
        else:
            # Add bias term
            X = np.insert(X, 0, 1.0, axis=1)      

        # If y is COO convert to CSR
        if y is not None: 
            if sparse.isspmatrix_coo(y):
                y = y.tocsr()                
        
        return X, y 

    @abstractmethod
    def _init_weights(self):
        """Initializes weights"""       
        pass

    @abstractmethod
    def _compute_output(self, X, theta=None):
        """Computes output of the model."""
        if theta is None:
            theta = self.theta_
        return X.dot(theta)

    def _evaluate_epoch(self, log=None):
        """Computes training costs, and optionally scores, for each epoch."""
        log = log or {}
        # Compute training costs and scores
        y_out = self._compute_output(self.X_train_)
        log['train_cost'] = self._cost(self.theta_, self.y_train_, y_out)
        y_pred = self._predict(self.X_train_)
        log['train_score'] = self._scorer(self.y_train_orig_, y_pred)

        # If early stop object is provided, get validation cost and score
        if self.early_stop and self.y_val_ is not None:
            if self.early_stop.val_size:
                y_out_val = self._compute_output(self.X_val_)
                log['val_cost'] = self._cost(self.theta_, self.y_val_, y_out_val)        
                y_pred_val = self._predict(self.X_val_)
                log['val_score'] = self._scorer(self.y_val_orig_, y_pred_val)

        return log

    def _copy_mutable_parameters(self):
        """Takes copies of mutable parameters in compliance with sklearn."""
        self._scorer = copy.copy(self.scorer)
        self._cost = copy.copy(self.cost)
        self._optimizer = copy.copy(self.optimizer)          

    
    def _compile(self):
        """Initializes all callbacks and copies mutable parameters to private members."""
        # Initialize callback list
        self._cbks = CallbackList()        
        # BlackBox callback
        self.blackbox_ = BlackBox()
        self._cbks.append(self.blackbox_)
        # Learning rate
        self._cbks.append(copy.copy(self.learning_rate))
        # Progress callback
        if self.verbose:
            self._cbks.append(Progress())
        # Add early stop if object injected.
        if self.early_stop:
            self._cbks.append(copy.copy(self.early_stop))
        # Add gradient checking if object injected.
        if self.gradient_check:
            self._cbks.append(GradientCheck())        
        # Initialize all callbacks.
        self._cbks.set_params(self.get_params())
        self._cbks.set_model(self)

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self._epoch = 0
        self._batch = 0        
        self._converged = False
        self.is_fitted_ = False                
        self._prepare_training_data(log.get("X"),log.get("y"))
        self._init_weights()            
        self._compile()      
        self._copy_mutable_parameters()  
        self._cbks.on_train_begin(log)
        
    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self._cbks.on_train_end()
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        self.n_iter_ = self._epoch
        self.is_fitted_ = True

    def _begin_epoch(self, log=None):
        """Increment the epoch count and shuffle the data."""
        self._epoch += 1
        rs = None
        if self.random_state:
            rs = self.random_state + self._epoch
        rs = self.random_state
        self.X_train_, self.y_train_ = shuffle_data(self.X_train_, self.y_train_, random_state=rs) 
        self._cbks.on_epoch_begin(self._epoch, log)

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}
        # Update log with current learning rate and parameters theta
        log['epoch'] = self._epoch
        log['learning_rate'] = self._eta
        log['theta'] = self.theta_.copy()     
        log['gradient'] = self._gradient.copy()      
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch(log)
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)

    def _begin_batch(self, log=None):
        self._batch += 1
        self._cbks.on_batch_begin(self._batch, log)

    def _end_batch(self, log=None):
        self._cbks.on_batch_end(self._batch, log)

    def fit(self, X, y):
        """Trains model until stop condition is met.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        y : numpy array, shape (n_samples,)
            Target values 

        Returns
        -------
        self : returns instance of self._
        """
        train_log = {'X': X, 'y': y}
        self._begin_training(train_log)

        while (self._epoch < self.epochs and not self._converged):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X_train_, self.y_train_, batch_size=self.batch_size):

                self._begin_batch()
                
                # Compute model output
                y_out = self._compute_output(X_batch)

                # Compute costs
                J = self._cost(self.theta_, y_batch, y_out)                

                # Format batch log with weights, gradient and cost
                batch_log = {'batch': self._batch, 'batch_size': X_batch.shape[0],
                             'theta': self.theta_.copy(), 'train_cost': J}
                
                # Compute the parameter updates.
                self.theta_, self._gradient = self._optimizer.update(gradient=self._cost.gradient, \
                    theta=self.theta_, learning_rate=self._eta, X=X_batch, y=y_batch,\
                        y_out=y_out)

                # Update batch log
                self._end_batch(batch_log)

            # Wrap up epoch
            self._end_epoch()

        self._end_training()
        return self         

    @abstractmethod
    def _predict(self, X, theta=None):
        """Predict function used in training and testing."""
        return self._compute_output(X, theta)            

    def predict(self, X):
        """Computes prediction for test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : prediction
        """
        check_is_fitted(self)     
        X, _ = self._prepare_test_data(X)           
        return  self._predict(X)

    @abstractmethod
    def _score(self, X, y):
        """Calculates score in training and testing."""

        y_pred = self._predict(X)
        return self._scorer(y, y_pred)
    
    def score(self, X, y):
        """Computes scores using test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        y : array_like of shape (n_samples,) or (n_samples, n_classes)
            The target variable.

        Returns
        -------
        score based upon the scorer object.
        
        """
        X, y = self._prepare_test_data(X, y)
        return self._score(X, y)

    def summary(self, features=None):
        summary(self.blackbox_, features)            

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT REGRESSOR                              #
# --------------------------------------------------------------------------- #
class GradientDescentRegressor(GradientDescent, RegressorMixin):
    """Gradient descent estimator for regression."""

    def __init__(self, optimizer=Standard(), learning_rate=Constant(eta0=0.01), 
                 batch_size=None, theta_init=None,  epochs=1000, cost=MSE(),
                 scorer=R2(), early_stop=False, verbose=False, checkpoint=100,  
                 random_state=None, gradient_check=False):

        self.optimizer = optimizer
        self.learning_rate = learning_rate        
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.cost = cost
        self.scorer = scorer
        self.early_stop = early_stop
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.gradient_check = gradient_check   

    @property
    def description(self):
        """Returns the estimator description."""                   
        regularization = self._cost.regularization.__class__.__name__       
        if regularization == "L0":
            return "Linear Regression with "  + self.variant    
        else:
            return "Linear Regression (" + self._cost.regularization.name + ") with " + self.variant                     



    def _prepare_training_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        super(GradientDescentRegressor, self)._prepare_training_data(X,y)
        # If early stopping, set aside a proportion of the data for the validation set    
        if self.early_stop and self.X_train_.shape[0] > 1:            
            if self.early_stop.val_size:                
                self.X_train_, self.X_val_, self.y_train_, self.y_val_ = \
                    data_split(self.X_train_, self.y_train_, stratify=False,
                    test_size=self.early_stop.val_size, random_state=self.random_state)                   
                if self.X_val_ is None:
                    self._handle_no_validation_data()                                    
        
        self.y_train_orig_ = self.y_train_
        self.y_val_orig_ = self.y_val_        

    def _prepare_test_data(self, X, y=None):
        """Prepares test data for prediction and scoring."""
        return super(GradientDescentRegressor,self)._prepare_test_data(X, y)         

    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.X_train_.shape[1],), \
                    "Initial parameters theta must have shape (n_features+1,)."
            self.theta_ = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)                
            self.theta_ = rng.randn(self.X_train_.shape[1])      

    def _compute_output(self, X, theta=None):
        """Computes output as linear combination of inputs and weights."""      
        return super(GradientDescentRegressor,self)._compute_output(X, theta) 

    def _predict(self, X, theta=None):
        """Computes predictions for use during training and validation."""
        return super(GradientDescentRegressor,self)._predict(X, theta) 

    def _score(self, X, y):
        """Computes scores for cross-validation and testing."""
        return super(GradientDescentRegressor,self)._score(X, y) 

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT CLASSIFIFER                            #
# --------------------------------------------------------------------------- #
class GradientDescentClassifier(GradientDescent, ClassifierMixin):
    """Gradient descent estimator for classification."""

    def __init__(self, optimizer=Standard(), learning_rate=Constant(eta0=0.01), 
                 batch_size=None, theta_init=None,  epochs=1000, cost=CrossEntropy(),
                 scorer=Accuracy(), early_stop=False, verbose=False, checkpoint=100,  
                 random_state=None, gradient_check=False):

        self.optimizer = optimizer
        self.learning_rate = learning_rate        
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.cost = cost        
        self.scorer = scorer
        self.early_stop = early_stop
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.gradient_check = gradient_check   

    @property
    def description(self):
        """Returns the estimator description."""                   
        cost_function = self._cost.__class__.__name__
        regularization = self._cost.regularization.__class__.__name__
        task = {'CrossEntropy': "Logistic Regression",
                'CategoricalCrossEntropy': "Multinomial Logistic Regression"}
        if regularization == "Nill":
            return task[cost_function] + " with " + self.variant    
        else:
            return task[cost_function] +  "(" + self._cost.regularization.name + ") with " + self.variant               

    def _compile(self):
        super(GradientDescentClassifier, self)._compile()
        self._sigmoid = Sigmoid()
        self._softmax = Softmax()

    def _prepare_training_data(self, X, y):
        """Prepares data for binary or multiclass classification."""
        super(GradientDescentClassifier, self)._prepare_training_data(X,y)            
        self.classes_ = np.unique(self.y_train_)
        self.n_classes_ = len(self.classes_)

        self._binarizer = LabelBinarizer()
        self._encoder = LabelEncoder()
    
        if self.early_stop:            
            if self.early_stop.val_size:                
                self.X_train_, self.X_val_, self.y_train_, self.y_val_ = \
                    data_split(X, y, stratify=True,
                    test_size=self.early_stop.val_size, random_state=self.random_state)

                if self.X_val_ is None:
                    self._handle_no_validation_data()
                    self.y_train_ = self._encoder.fit_transform(self.y_train_)                    
                    self.y_train_orig_ = self.y_train_
                    if self.n_classes_ > 2:
                        self.y_train_ = self._binarizer.fit_transform(self.y_train_)

                else:
                    self.y_train_ = self._encoder.fit_transform(self.y_train_)                    
                    self.y_val_ = self._encoder.transform(self.y_val_)
                    self.y_train_orig_ = self.y_train_
                    self.y_val_orig_ = self.y_val_
                    if self.n_classes_ > 2:
                        self.y_train_ = self._binarizer.fit_transform(self.y_train_)
                        self.y_val_ = self._binarizer.transform(self.y_val_)                    
        else:
            self.y_train_ = self._encoder.fit_transform(self.y_train_)
            self.y_train_orig_ = self.y_train_            
            if self.n_classes_ > 2:
                self.y_train_ = self._binarizer.fit_transform(self.y_train_)    

    def _prepare_test_data(self, X, y=None):
        """Prepares test data for prediction and scoring."""
        X, y = super(GradientDescentClassifier, self)._prepare_test_data(X, y)
        y = self._encoder.transform(y)
        if self.n_classes_ > 2:
            y = self._binarizer.transform(y)
        return X, y
                
        
    def _init_weights_binary_classification(self):
        """Initializes weights for binary classification."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.n_features_,), \
                "Initial parameters theta must have shape (n_features+1)."
            self.theta_ = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)                
            self.theta_ = rng.randn(self.n_features_)   

    def _init_weights_multiclass(self):
        """Initializes weights for multiclass classification."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.n_features_, self.n_classes_), \
                "Initial parameters theta must have shape (n_features+1, n_classes)."
            self.theta_ = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)                
            self.theta_ = rng.randn(self.n_features_, self.n_classes_)        

    def _init_weights(self):
        """Initializes model parameters."""        
        if self.y_train_.ndim == 1:
            self._init_weights_binary_classification()
        else:
            self._init_weights_multiclass()

    def _compute_output(self, X, theta=None):
        """Computes output as sigmoid or softmax result."""
        if not theta:
            theta = self.theta_
        z = X.dot(theta)

        if self.n_classes_ == 2:
            o = self._sigmoid(z)
        else:
            o = self._softmax(z)

        return o

    def _predict(self, X, theta=None):
        """Computes prediction of class labels."""
        o = self._compute_output(X, theta)
        if self.n_classes_ == 2:
            y_pred = np.round(o).astype(int)
        else:
            y_pred = o.argmax(axis=1)
        return y_pred

    def _score(self, X, y):
        """Computes scores for cross-validation and testing."""

        y_pred = self._predict(X)
        return self._scorer(y, y_pred)

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT OPTIMIZER                              #
# --------------------------------------------------------------------------- #        
class GradientDescentOptimizer(GradientDescent):
    """Performs pure optimization of a 2d objective function."""
    def __init__(self, optimizer=Standard(), learning_rate=Constant(eta0=0.01), 
                 batch_size=None, theta_init=None,  epochs=1000, early_stop=False, 
                 verbose=False, checkpoint=100,  random_state=None, 
                 gradient_check=False):

        self.optimizer = optimizer
        self.learning_rate = learning_rate        
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.early_stop = early_stop
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.gradient_check = gradient_check     

    @property
    def description(self):
        """Returns the estimator description."""    
        if self.batch_size is None:
            variant = 'Batch Gradient Descent'
        elif self.batch_size == 1:
            variant = 'Stochastic Gradient Descent'
        else:
            variant = 'Minibatch Gradient Descent'                       
        optimizer = self.optimizer.__class__.__name__       
        return variant + ' with ' + optimizer + ' Optimization'  

    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init:
            if self.theta_init.shape[0] != 2:
                raise ValueError("Parameters theta must have shape (2,)")
            else:
                self.theta_ = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)         
            self.theta_ = rng.randn(2)

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self._epoch = 0
        self._batch = 0        
        self._converged = False
        self.is_fitted_ = False                        
        self._init_weights()            
        self._compile()              
        self._cbks.on_train_begin(log)   

    def _begin_epoch(self, log=None):
        """Runs 'begin_epoch' methods on all callbacks."""
        self._epoch += 1        
        self._cbks.on_epoch_begin(self._epoch, log) 

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}
        # Update log with current learning rate and parameters theta
        log['epoch'] = self._epoch
        log['learning_rate'] = self._eta
        log['theta'] = self.theta_.copy()     
        log['train_cost'] = self._cost.copy()
        log['gradient'] = self._gradient.copy()
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)        

    def fit(self, objective):
        """Fits the objective function.
        
        Parameters
        ----------
        objective : object derived from Objective class
            The objective function to be minimized

        Returns
        -------
        self
        """
        self._objective = objective
        self._begin_training()

        while (self._epoch < self.epochs and not self._converged):

            self._begin_epoch()

            self._cost = self._objective(self.theta_)
            self._gradient = self._objective.gradient(self.theta_)
            self._theta_ = self._optimizer(self.theta_, self._gradient, self._eta)

            self._end_epoch()

        self._end_training()
        return self  

        
