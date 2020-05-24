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

from mlstudio.supervised.core.tasks import LinearRegression, LogisticRegression
from mlstudio.supervised.core.tasks import MultinomialLogisticRegression
from mlstudio.supervised.core.objectives import MSE, CrossEntropy
from mlstudio.supervised.core.objectives import CategoricalCrossEntropy
from mlstudio.supervised.core.objectives import Adjiman
from mlstudio.supervised.core.optimizers import Classic
from mlstudio.supervised.core.scorers import R2, Accuracy
from mlstudio.supervised.callbacks.early_stop import EarlyStop
from mlstudio.supervised.callbacks.debugging import GradientCheck
from mlstudio.supervised.callbacks.base import CallbackList
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.monitor import BlackBox, Progress
from mlstudio.supervised.callbacks.learning_rate import LearningRateSchedule
from mlstudio.utils.data_manager import batch_iterator, data_split, shuffle_data
from mlstudio.utils.data_manager import add_bias_term, encode_labels, one_hot_encode
from mlstudio.utils.data_manager import RegressionDataProcessor, ClassificationDataProcessor
from mlstudio.utils.validation import check_X, check_X_y, check_is_fitted
from mlstudio.utils.validation import validate_zero_to_one, validate_metric
from mlstudio.utils.validation import validate_objective, validate_optimizer
from mlstudio.utils.validation import validate_scorer, validate_early_stop
from mlstudio.utils.validation import validate_learning_rate_schedule
from mlstudio.utils.validation import validate_int, validate_string
from mlstudio.utils.validation import validate_early_stop, validate_metric
from mlstudio.utils.validation import validate_scorer
# =========================================================================== #
#                          GRADIENT DESCENT                                   #
# =========================================================================== #        
class GradientDescent(BaseEstimator):
    """Performs pure optimization of a 2d objective function."""
    def __init__(self, learning_rate=0.01, epochs=1000, theta_init=None,
                 optimizer=Classic(), objective=MSE(), schedule=None,                  
                 early_stop=None, verbose=False, checkpoint=100, 
                 random_state=None, gradient_check=None):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_init = theta_init
        self.optimizer = optimizer
        self.objective  = objective
        self.schedule = schedule
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

    @property
    def stabilized(self):
        return self._stabilized

    @stabilized.setter
    def stabilized(self, x):
        self._stabilized = x 

    @property
    def critical_points(self):
        return self._critical_points   

    @property
    def feature_names(self):
        return self._feature_names

    @feature_names.setter
    def feature_names(self, x):
        self._feature_names = x     

# --------------------------------------------------------------------------- #
#                                 VALIDATION                                  #
# --------------------------------------------------------------------------- #   
    def _validate_params(self):
        """Performs validation on the hyperparameters."""
        validate_zero_to_one(param=self.learning_rate, param_name="learning_rate", 
                            left="open", right="closed")
        validate_int(param=self.epochs, param_name='epochs')
        validate_optimizer(self.optimizer)
        validate_objective(self.objective)
        if self.schedule:
            validate_learning_rate_schedule(self.schedule)
        if self.early_stop:
            validate_early_stop(self.early_stop)
        validate_bool(param=self.verbose, param_name='verbose')
        validate_int(param=self.checkpoint, param_name='checkpoint')
        if self.random_state:
            validate_int(param=self.random_state, param_name='random_state')
        if self.gradient_check:
            validate_gradient_check(self.gradient_check)


# --------------------------------------------------------------------------- #
#                                 CHECK STATE                                 #
# --------------------------------------------------------------------------- #   
    def _check_state(self, log):
        d = {}
        d['Epoch'] = log.get('epoch')
        d['Learning Rate'] = log.get('learning_rate')
        d['Theta'] = np.linalg.norm(log.get('theta'))
        d['Gradient'] = np.linalg.norm(log.get('gradient'))
        kv = d.items()
        sp = {str(key): str(value) for key, value in kv}
        if self._stabilized != self._last_state:
            self._last_state = self._stabilized     
            if self._stabilized:       
                self._critical_points.append(sp)  


# --------------------------------------------------------------------------- #
#                               COMPILE                                       #
# --------------------------------------------------------------------------- #    
    def _copy_mutable_parameters(self):
        """Copies mutable parameters to new members for sklearn compatibility."""
        # Custom objects
        self._optimizer = copy.deepcopy(self.optimizer)
        self._objective = copy.deepcopy(self.objective)        
        self._early_stop = copy.deepcopy(self.early_stop) if self.early_stop \
            else self.early_stop
        self._gradient_check = copy.deepcopy(self.gradient_check) if \
            self.gradient_check else self.gradient_check
        self._schedule = copy.deepcopy(self.schedule) if \
            self.schedule else self.schedule            
        
    def _compile(self, log=None):
        """Initializes all callbacks."""
        # Copy mutable classes and parameters that will be modified during
        # training. 
        self._copy_mutable_parameters()  
        # Initialize implicit dependencies. Yeah, yeah I know... But adding these
        # to the constructor seemed a bit much.         
        self._cbks = CallbackList()       
        self._progress = Progress()
        self.blackbox_ = BlackBox()        

        # Add callbacks to callback list         
        self._cbks.append(self.blackbox_)        
        if self.verbose:
            self._cbks.append(self._progress)        
        if isinstance(self._early_stop, EarlyStop):
            self._cbks.append(self._early_stop)        
        if isinstance(self._gradient_check, GradientCheck):
            self._cbks.append(self._gradient_check)        
        if isinstance(self._schedule, LearningRateSchedule):
            self._cbks.append(self._schedule)                

        # Initialize all callbacks.
        self._cbks.set_params(self.get_params())
        self._cbks.set_model(self)        

# --------------------------------------------------------------------------- #
#                             INITIALIZATION                                  #
# --------------------------------------------------------------------------- #              
    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init is not None:
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
        log = log or {}   
        self._validate_params(self) 
        # Private variables
        self._epoch = 0
        self._batch = 0        
        self._last_state = False
        self._converged = False
        self._stabilized = False
        self._eta = copy.copy(self.learning_rate)
        # Attributes
        self._critical_points = []   
        self.theta_ = 0
        self.gradient_ = 0
        # Dependencies, data, and weights.
        self._compile(log)              
        self._init_weights()            
        self._cbks.on_train_begin(log) 

    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        log = log or {}    
        self._cbks.on_train_end()
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        self.n_iter_ = self._epoch

    def _begin_epoch(self, log=None):
        """Runs 'begin_epoch' methods on all callbacks."""
        log = log or {}    
        self._epoch += 1        
        self._cbks.on_epoch_begin(self._epoch, log) 
        self._check_state(log)     

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

            epoch_log = {'epoch': self._epoch, 'learning_rate': self._eta, 'train_cost': cost,
                         'theta': self.theta_, 'gradient': self.gradient_}

            self._begin_epoch(copy.deepcopy(epoch_log))

            cost = self._objective(self.theta_)
            self.theta_, self.gradient_ = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self.theta_))

            self._end_epoch()

        self._end_training()
        return self         


# =========================================================================== #
#                          GRADIENT DESCENT ESTIMATOR                         #
# =========================================================================== #
class GradientDescentEstimator(ABC, GradientDescent):
    """Base class gradient descent estimator."""

    def __init__(self, learning_rate=0.01, epochs=1000, theta_init=None,
                 optimizer=Classic(), objective=MSE(), batch_size=None, 
                 val_size=0.3, schedule=None, scorer=R2(), early_stop=None, 
                 verbose=False, checkpoint=100, random_state=None, 
                 gradient_check=None):
                 
        super(GradientDescentEstimator, self).__init__(
            learning_rate = learning_rate,      
            epochs = epochs,
            theta_init = theta_init,
            optimizer = optimizer,
            objective = objective,                                          
            schedule=schedule,
            early_stop = early_stop,
            verbose = verbose,
            checkpoint = checkpoint,
            random_state = random_state,
            gradient_check = gradient_check
        )
        self.val_size = val_size        
        self.batch_size = batch_size
        self.scorer = scorer
# --------------------------------------------------------------------------- #
#                                 VALIDATION                                  #
# --------------------------------------------------------------------------- #   
    def _validate_params(self):        
        super(GradientDescentEstimator, self)._validate_params()
        if self.val_size:
            validate_zero_to_one(param=self.val_size, param_name='val_size')
        if self.batch_size:
            validate_int(param=self.batch_size, param_name='batch_size')
        if self.scorer:
            validate_scorer(self.scorer)
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
    def task(self):
        return self._task

    @property
    def description(self):
        """Returns the estimator description."""                   
        task = self._task.name
        regularizer = self._objective.regularizer.__class__.__name__     
        optimizer = self._optimizer.__class__.__name__
        regularizer_title = ""
        optimizer_title = ""

        if regularizer != "Nill":
            regularizer_title = " (" + regularizer + " Regularizer) "

        if optimizer != "Classic":
            optimizer_title = " (" + optimizer_title + " Optimization) "
        
        return task + regularizer_title + " with " + \
            self.variant + optimizer_title      
# --------------------------------------------------------------------------- #
#                               COMPILE                                       #
# --------------------------------------------------------------------------- # 
    def _copy_mutable_parameters(self):
        """Copies mutable parameters for sklearn compliance."""
        super(GradientDescentEstimator, self)._copy_mutable_parameters()
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
    @abstractmethod
    def _prepare_training_data(self, X, y):
        """Prepares X and y data for training."""
        pass


# --------------------------------------------------------------------------- #
#                  INTERNAL AND EXTERNAL CALLBACKS                            #
# --------------------------------------------------------------------------- #

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        log = log or {}    
        # Private variables
        self._epoch = 0
        self._batch = 0        
        self._last_state = False
        self._converged = False
        self._stabilized = False
        self._eta = copy.copy(self.learning_rate)
        # Attributes
        self._critical_points = []   
        self.theta_ = 0
        self.gradient_ = 0
        # Dependencies, data, and weights.
        self._compile(log)              
        self._prepare_training_data(log.get("X"),log.get("y"))        
        self._init_weights()            
        self._cbks.on_train_begin(log)        

    def _begin_epoch(self, log=None):
        """Increment the epoch, evaluate using current parameters and shuffle the data."""
        log = log or {}    
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch(log)                
        # Shuffle data      
        rs = None
        if self.random_state:
            rs = self.random_state + self._epoch
        self.X_train_, self.y_train_ = shuffle_data(self.X_train_, self.y_train_, random_state=rs) 
        # Call 'on_epoch_begin' methods on callbacks.
        self._cbks.on_epoch_begin(self._epoch, log)               
        # Once the performance callback has executed, check whether the optimization has stabilized.
        self._check_state(log) 
        self._epoch += 1         

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}    
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)        

    def _begin_batch(self, log=None):
        log = log or {}            
        self._cbks.on_batch_begin(self._batch, log)
        self._batch += 1

    def _end_batch(self, log=None):
        log = log or {}    
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

        # Compute validation error and score if a validation set has been 
        # designated.
        if hasattr(self, 'X_val_'):
            if self.X_val_.shape[0] > 0:
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

            epoch_log = {'epoch': self._epoch, 'learning_rate': self._eta, 'theta': self.theta_,
                         'gradient': self.gradient_}            

            self._begin_epoch(copy.deepcopy(epoch_log))

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
                self.theta_, self.gradient_ = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self.theta_),  X=X_batch, y=y_batch,\
                        y_out=y_out)

                # Update batch log
                self._end_batch(copy.deepcopy(batch_log))

            # Wrap up epoch
            self._end_epoch()

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
        X = check_X(X)
        X = add_bias_term(X)
        return self._task.predict(self.theta_, X)
    
    def _score(self, X, y):
        """Calculates scores during training."""
        # Called during training. Assumes data has valid format. 
        y_pred = self._task.predict(self.theta_, X)
        return self._scorer(y, y_pred)
    
    def score(self, X, y):
        """Computes scores using test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        y : array_like of shape (n_samples,) 
            The target variable.

        Returns
        -------
        score based upon the scorer object.
        
        """        
        y_pred = self.predict(X)        
        return self._scorer(y, y_pred)

    def summary(self, features=None):
        observer = Performance()
        observer.report(self, features)

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT REGRESSOR                              #
# --------------------------------------------------------------------------- #
class GradientDescentRegressor(GradientDescentEstimator, RegressorMixin):
    """Gradient descent estimator for regression."""

    def __init__(self, learning_rate=0.01, epochs=1000, theta_init=None,
                 optimizer=Classic(), objective=MSE(), batch_size=None, 
                 val_size=0.3, schedule=None, scorer=R2(), early_stop=None, 
                 verbose=False, checkpoint=100, random_state=None, 
                 gradient_check=None):
        
        super(GradientDescentRegressor, self).__init__(\
            learning_rate = learning_rate,
            epochs = epochs,        
            theta_init = theta_init,
            optimizer = optimizer,
            objective = objective,        
            batch_size = batch_size,
            val_size = val_size,
            schedule=schedule,
            scorer = scorer,
            early_stop = early_stop,
            verbose = verbose,
            checkpoint = checkpoint,
            random_state = random_state,
            gradient_check = gradient_check   
        )

    def _compile(self, log=None):
        """Compiles required objects."""
        super(GradientDescentRegressor, self)._compile(log)        
        self._task = LinearRegression()        

    def _prepare_training_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        processor = RegressionDataProcessor(val_size=self.val_size, 
                                            random_state=self.random_state)
        data = processor.fit_transform(X, y)
        # Set attributes from data.
        for k, v in data.items():
            setattr(self, k, v)

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

    def __init__(self, learning_rate=0.01, epochs=1000, theta_init=None,
                 optimizer=Classic(), objective=CrossEntropy(), batch_size=None, 
                 val_size=0.3, schedule=None, scorer=Accuracy(), early_stop=None, 
                 verbose=False, checkpoint=100, random_state=None, 
                 gradient_check=None):
        
        super(GradientDescentClassifier, self).__init__(\
            learning_rate = learning_rate,
            epochs = epochs,        
            theta_init = theta_init,
            optimizer = optimizer,
            objective = objective,        
            batch_size = batch_size,
            val_size = val_size,
            schedule=schedule,
            scorer = scorer,
            early_stop = early_stop,
            verbose = verbose,
            checkpoint = checkpoint,
            random_state = random_state,
            gradient_check = gradient_check   
        )


    def _compile(self, log=None):
        """Compiles required objects."""
        super(GradientDescentClassifier, self)._compile(log)        
        y = log.get('y')
        if np.ndim(y) == 2 or len(np.unique(y)==2):
            self._task = LogisticRegression()
        else:
            self._task = MultinomialLogisticRegression()

    def _prepare_training_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        processor = ClassificationDataProcessor(val_size=self.val_size, 
                                            random_state=self.random_state)
        data = processor.fit_transform(X, y)
        # Set attributes from data.
        for k, v in data.items():
            setattr(self, k, v)                
        
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

