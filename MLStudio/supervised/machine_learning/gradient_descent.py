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
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.testing import ignore_warnings
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from mlstudio.supervised.core.tasks import LinearRegression, LogisticRegression
from mlstudio.supervised.core.tasks import MultinomialLogisticRegression
from mlstudio.supervised.core.objectives import MSE, CrossEntropy
from mlstudio.supervised.core.objectives import CategoricalCrossEntropy
from mlstudio.supervised.core.objectives import Adjiman
from mlstudio.supervised.core.optimizers import Standard
from mlstudio.supervised.core.scorers import R2, Accuracy
from mlstudio.supervised.callbacks.early_stop import EarlyStop
from mlstudio.supervised.callbacks.debugging import GradientCheck
from mlstudio.supervised.callbacks.base import CallbackList
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.learning_rate import Constant
from mlstudio.supervised.callbacks.monitor import BlackBox, Progress, summary
from mlstudio.utils.data_manager import batch_iterator, data_split, shuffle_data

# =========================================================================== #
#                          GRADIENT DESCENT                                   #
# =========================================================================== #        
class GradientDescent(BaseEstimator):
    """Performs pure optimization of a 2d objective function."""
    def __init__(self, optimizer=Standard(), objective=Adjiman(),
                 learning_rate=Constant(eta0=0.01), 
                 theta_init=None,  epochs=1000, early_stop=None, 
                 verbose=False, checkpoint=100,  random_state=None, 
                 gradient_check=None):

        self.optimizer = optimizer
        self.objective  = objective
        self.learning_rate = learning_rate
        self.theta_init = theta_init
        self.epochs = epochs
        self.early_stop = early_stop
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.gradient_check = gradient_check

# --------------------------------------------------------------------------- #
#                               PROPERTIES                                    #
# --------------------------------------------------------------------------- #    
    @property    
    def description(self):
        """Returns the estimator description."""                         
        optimizer = self._optimizer.__class__.__name__       
        return 'Gradient Descent with ' + optimizer + ' Optimization'  

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

# --------------------------------------------------------------------------- #
#                               COMPILE                                       #
# --------------------------------------------------------------------------- #    
    def _copy_mutable_parameters(self):
        """Copies mutable parameters to new members for sklearn compatibility."""
        self._optimizer = copy.deepcopy(self.optimizer)
        self._objective = copy.deepcopy(self.objective)
        self._learning_rate = copy.deepcopy(self.learning_rate)
        self._early_stop = copy.deepcopy(self.early_stop) if self.early_stop \
            else self.early_stop
        self._gradient_check = copy.deepcopy(self.gradient_check) if \
            self.gradient_check else self.gradient_check

    def _compile(self):
        """Initializes all callbacks."""
        # Copy mutable classes to private members in order to comply with
        # sklearn's API standards.
        self._copy_mutable_parameters()  
        # Initialize implicit dependencies. Yeah, yeah I know... But adding these
        # to the constructor seemed a bit much.         
        self._cbks = CallbackList()       
        self._progress = Progress()
        self.blackbox_ = BlackBox()        

        # Add callbacks to callback list         
        self._cbks.append(self.blackbox_)        
        self._cbks.append(self._learning_rate)        
        if self.verbose:
            self._cbks.append(self._progress)        
        if isinstance(self._early_stop, EarlyStop):
            self._cbks.append(self._early_stop)        
        if isinstance(self._gradient_check, GradientCheck):
            self._cbks.append(self._gradient_check)        

        # Initialize all callbacks.
        self._cbks.set_params(self.get_params())
        self._cbks.set_model(self)        

# --------------------------------------------------------------------------- #
#                             INITIALIZATION                                  #
# --------------------------------------------------------------------------- #              
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


# --------------------------------------------------------------------------- #
#                  INTERNAL AND EXTERNAL CALLBACKS                            #
# --------------------------------------------------------------------------- #

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self._epoch = 0
        self._batch = 0        
        self._converged = False
        self.is_fitted_ = False          
        self._compile()              
        self._init_weights()            
        self._cbks.on_train_begin(log) 

    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self._cbks.on_train_end()
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        self.n_iter_ = self._epoch
        self.is_fitted_ = True          

    def _begin_epoch(self, log=None):
        """Runs 'begin_epoch' methods on all callbacks."""
        self._epoch += 1        
        self._cbks.on_epoch_begin(self._epoch, log) 

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)        

# --------------------------------------------------------------------------- #
#                                 FIT                                         #
# --------------------------------------------------------------------------- #
    def fit(self, X=None, y=None):
        """Fits the objective function.
        
        Parameters
        ----------
        objective : object derived from Objective class
            The objective function to be minimized

        Returns
        -------
        self
        """
        
        self._begin_training()

        while (self._epoch < self.epochs and not self._converged):

            self._begin_epoch()

            cost = self._objective(self.theta_)
            self.theta_, gradient = self._optimizer.update(gradient=self._objective.gradient, \
                    theta=self.theta_, learning_rate=self._eta)
            epoch_log = {'epoch': self._epoch, 'learning_rate': self._eta, 'train_cost': cost,
                   'theta': self.theta_, 'gradient': gradient}

            self._end_epoch(copy.deepcopy(epoch_log))

        self._end_training()
        return self         


# =========================================================================== #
#                          GRADIENT DESCENT ESTIMATOR                         #
# =========================================================================== #
class GradientDescentEstimator(ABC, GradientDescent):
    """Base class gradient descent estimator."""

    def __init__(self, task=LinearRegression(), optimizer=Standard(), 
                 objective=MSE(), learning_rate=Constant(eta0=0.01), 
                 batch_size=None, theta_init=None,  epochs=1000, 
                 scorer=R2(), early_stop=None, verbose=False, 
                 checkpoint=100, random_state=None, gradient_check=None):
                 
        super(GradientDescentEstimator, self).__init__(
            optimizer = optimizer,
            objective = objective,
            learning_rate = learning_rate,
            theta_init = theta_init,
            epochs = epochs,
            early_stop = early_stop,
            verbose = verbose,
            checkpoint = checkpoint,
            random_state = random_state,
            gradient_check = gradient_check
        )

        self.task = task
        self.batch_size = batch_size
        self.scorer = scorer
# --------------------------------------------------------------------------- #
#                               PROPERTIES                                    #
# --------------------------------------------------------------------------- #    
    @property
    def variant(self):
        if self.batch_size is None:
            variant = 'Batch Gradient Descent'
        elif self.batch_size == 1:
            variant = 'Stochastic Gradient Descent'
        else:
            variant = 'Minibatch Gradient Descent'
        return variant

    @property
    def description(self):
        """Returns the estimator description."""                   
        task = self._task.name
        regularizer = self._objective.regularizer.__class__.__name__     
        optimizer = self._optimizer.__class__.__name__
        regularizer_title = ""
        optimizer_title = ""

        if regularizer != "Nill":
            regularizer_title = " (" + optimizer_title + " Regularization) "

        if optimizer != "Standard":
            optimizer_title = " (" + optimizer_title + " Optimization) "
        
        return task + regularizer_title + " with " + \
            self.variant + optimizer_title      
# --------------------------------------------------------------------------- #
#                               COMPILE                                       #
# --------------------------------------------------------------------------- # 
    def _copy_mutable_parameters(self):
        """Copies mutable parameters for sklearn compliance."""
        super(GradientDescentEstimator, self)._copy_mutable_parameters()
        self._task = copy.deepcopy(self.task)
        self._scorer = copy.deepcopy(self.scorer)
# --------------------------------------------------------------------------- #
#                            INITIALIZATION                                   #
# --------------------------------------------------------------------------- #                                        
    @abstractmethod
    def _init_weights(self):
        pass

# --------------------------------------------------------------------------- #
#                             DATA PREPARATION                                #
# --------------------------------------------------------------------------- #
    def _handle_no_validation_data(self):
        """Issues warning if validation set could not be created due to training set size."""        
        if self.X_val_ is None:            
            if 'val' in self._early_stop.metric:
                self._early_stop.metric = "train_cost"         
                warnings.warn("Validation set could not be created. Training set \
                    only has {s} observations. Changing cross-validation metric \
                        to 'train_cost'.\
                        ".format(s=str(self.X_train_.shape[0])), UserWarning)        
            else:
                warnings.warn("Validation set could not be created. Training set \
                    only has {s} observations.".format(s=str(self.X_train_.shape[0])), UserWarning)     


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


# --------------------------------------------------------------------------- #
#                  INTERNAL AND EXTERNAL CALLBACKS                            #
# --------------------------------------------------------------------------- #

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self._epoch = 0
        self._batch = 0        
        self._converged = False
        self.is_fitted_ = False                                
        self._compile()      
        self._prepare_training_data(log.get("X"),log.get("y"))
        self._init_weights()            
        self._cbks.on_train_begin(log)
        

    def _begin_epoch(self, log=None):
        """Increment the epoch count and shuffle the data."""
        super(GradientDescentEstimator, self)._begin_epoch(log)        
        rs = None
        if self.random_state:
            rs = self.random_state + self._epoch
        self.X_train_, self.y_train_ = shuffle_data(self.X_train_, self.y_train_, random_state=rs) 
        

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}    
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch(log)
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)

    def _begin_batch(self, log=None):
        self._batch += 1
        self._cbks.on_batch_begin(self._batch, log)

    def _end_batch(self, log=None):
        self._cbks.on_batch_end(self._batch, log)

# --------------------------------------------------------------------------- #
#                               EVALUATION                                    #
# --------------------------------------------------------------------------- #
    def _evaluate_epoch(self, log=None):
        """Computes training costs, and optionally scores, for each epoch."""
        log = log or {}
        # Compute training costs and scores
        y_out = self._task.compute_output(self.theta_, self.X_train_)
        log['train_cost'] = self._objective(self.theta_, self.y_train_, y_out)
        y_pred = self._task.predict(self.theta_, self.X_train_)
        log['train_score'] = self._scorer(self.y_train_orig_, y_pred)

        # If early stop object is provided, get validation cost and score
        if self._early_stop:
            if self._early_stop.val_size and self.X_val_.shape[0] > 0:
                y_out_val = self._task.compute_output(self.theta_, self.X_val_)
                log['val_cost'] = self._objective(self.theta_, self.y_val_, y_out_val)        
                y_pred_val = self._task.predict(self.theta_, self.X_val_)
                log['val_score'] = self._scorer(self.y_val_orig_, y_pred_val)

        return log        

# --------------------------------------------------------------------------- #
#                                  FIT                                        #
# --------------------------------------------------------------------------- #
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
        X, y = check_X_y(X, y)
        train_log = {'X': X, 'y': y}
        self._begin_training(train_log)

        while (self._epoch < self.epochs and not self._converged):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X_train_, self.y_train_, batch_size=self.batch_size):

                self._begin_batch()
                
                # Compute model output
                y_out = self._task.compute_output(self.theta_, X_batch)

                # Compute costs
                J = self._objective(self.theta_, y_batch, y_out)                

                # Format batch log with weights, gradient and cost
                batch_log = {'batch': self._batch, 'batch_size': X_batch.shape[0],
                             'theta': self.theta_, 'train_cost': J}
                
                # Compute the parameter updates.
                self.theta_, gradient = self._optimizer.update(gradient=self._objective.gradient, \
                    theta=self.theta_, learning_rate=self._eta, X=X_batch, y=y_batch,\
                        y_out=y_out)

                # Update batch log
                self._end_batch(copy.deepcopy(batch_log))

            epoch_log = {'epoch': self._epoch, 'learning_rate': self._eta, 'theta': self.theta_,
                         'gradient': gradient}

            # Wrap up epoch
            self._end_epoch(copy.deepcopy(epoch_log))

        self._end_training()
        return self     

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
        return self._task.predict(self.theta_, X)
    
    def _score(self, X, y):
        """Calculates score in training and testing."""

        y_pred = self.predict(X)
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
        return self._score(X, y)

    def summary(self, features=None):
        summary(self.blackbox_, features)            

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT REGRESSOR                              #
# --------------------------------------------------------------------------- #
class GradientDescentRegressor(GradientDescentEstimator, RegressorMixin):
    """Gradient descent estimator for regression."""

    def __init__(self, task=LinearRegression(), optimizer=Standard(), 
                 objective=MSE(),learning_rate=Constant(eta0=0.01), 
                 batch_size=None, theta_init=None,  epochs=1000, 
                 scorer=R2(), early_stop=None, verbose=False, 
                 checkpoint=100, random_state=None, gradient_check=None):
        
        super(GradientDescentRegressor, self).__init__(\
            task=task,
            optimizer = optimizer,
            objective = objective,
            learning_rate = learning_rate,        
            batch_size = batch_size,
            theta_init = theta_init,
            epochs = epochs,        
            scorer = scorer,
            early_stop = early_stop,
            verbose = verbose,
            checkpoint = checkpoint,
            random_state = random_state,
            gradient_check = gradient_check   
        )

    def _prepare_training_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        super(GradientDescentRegressor, self)._prepare_training_data(X,y)
        # If early stopping, set aside a proportion of the data for the validation set    
        if self._early_stop:            
            if self._early_stop.val_size:                
                self.X_train_, self.X_val_, self.y_train_, self.y_val_ = \
                    data_split(self.X_train_, self.y_train_, stratify=False,
                        test_size=self._early_stop.val_size, 
                        random_state=self.random_state)                   
                if self.X_val_ is None:
                    self._handle_no_validation_data()                                    
        
        self.y_train_orig_ = self.y_train_
        self.y_val_orig_ = self.y_val_        

    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.X_train_.shape[1],), \
                    "Initial parameters theta must have shape (n_features+1,)."
            self.theta_ = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)                
            self.theta_ = rng.randn(self.X_train_.shape[1])      

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT CLASSIFIFER                            #
# --------------------------------------------------------------------------- #
class GradientDescentClassifier(GradientDescentEstimator, ClassifierMixin):
    """Gradient descent estimator for classification."""

    def __init__(self, task=LogisticRegression(), optimizer=Standard(), 
                 objective=CrossEntropy(),learning_rate=Constant(eta0=0.01), 
                 batch_size=None, theta_init=None,  epochs=1000, 
                 scorer=Accuracy(), early_stop=None, verbose=False, 
                 checkpoint=100,  random_state=None, gradient_check=None):
        
        super(GradientDescentClassifier, self).__init__(
            task=task,
            optimizer = optimizer,
            objective = objective,
            learning_rate = learning_rate,        
            batch_size = batch_size,
            theta_init = theta_init,
            epochs = epochs,        
            scorer = scorer,
            early_stop = early_stop,
            verbose = verbose,
            checkpoint = checkpoint,
            random_state = random_state,
            gradient_check = gradient_check   
        )

    def _prepare_training_data(self, X, y):
        """Prepares data for binary or multiclass classification."""
        super(GradientDescentClassifier, self)._prepare_training_data(X,y)            
        self.classes_ = np.unique(self.y_train_)
        self.n_classes_ = len(self.classes_)

        self._binarizer = LabelBinarizer()
        self._encoder = LabelEncoder()
    
        if self._early_stop:            
            if self._early_stop.val_size:                
                self.X_train_, self.X_val_, self.y_train_, self.y_val_ = \
                    data_split(X, y, stratify=True,
                    test_size=self._early_stop.val_size, random_state=self.random_state)

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

