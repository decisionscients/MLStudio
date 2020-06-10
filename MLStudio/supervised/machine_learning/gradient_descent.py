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
from mlstudio.supervised.callbacks.monitor import BlackBox, Progress, Monitor
from mlstudio.supervised.callbacks.learning_rate import LearningRateSchedule, Constant
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
from mlstudio.utils.validation import validate_scorer, validate_bool
# =========================================================================== #
#                          GRADIENT DESCENT                                   #
# =========================================================================== #        
class GradientDescent(BaseEstimator):
    """Performs pure optimization of a 2d objective function."""
    def __init__(self, learning_rate=0.01, epochs=1000, theta_init=None,
                 optimizer=Classic(), objective=MSE(), schedule=Constant(),                  
                 monitor=Monitor(), early_stop=None, get_best_weights=True,
                 verbose=False, checkpoint=100, random_state=None, 
                 gradient_check=False):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta_init = theta_init
        self.optimizer = optimizer
        self.objective  = objective
        self.schedule = schedule
        self.monitor = monitor
        self.early_stop = early_stop
        self.get_best_weights = get_best_weights
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
    def final_result(self):
        return self._final_result

    @property
    def best_result(self):
        return self._best_result

    @property
    def critical_points(self):
        return self._monitor.critical_points

    @property
    def converged(self):
        return self._converged

    @converged.setter
    def converged(self, x):
        self._converged = x       

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

# --------------------------------------------------------------------------- #
#                               COMPILE                                       #
# --------------------------------------------------------------------------- #    
    def _copy_mutable_parameters(self):
        """Copies mutable parameters to new members for sklearn compatibility."""
        # Custom objects
        self._optimizer = copy.deepcopy(self.optimizer)
        self._objective = copy.deepcopy(self.objective)    

        self._early_stop = copy.deepcopy(self.early_stop) \
            if isinstance(self.early_stop, EarlyStop)\
                 else self.early_stop
        
        self._schedule = copy.deepcopy(self.schedule) \
            if isinstance(self.schedule, LearningRateSchedule)\
                 else self.schedule

        self._monitor = copy.deepcopy(self.monitor) \
            if isinstance(self.monitor, Monitor)\
                 else self.monitor
        
    def _compile(self, log=None):
        """Initializes all callbacks."""
        # Copy mutable classes and parameters that will be modified during
        # training. 
        self._copy_mutable_parameters()  

        # Initialize implicit dependencies.    
        self._cbks = CallbackList()               
        self.blackbox_ = BlackBox()                

        # Add callbacks to callback list         
        self._cbks.append(self.blackbox_)    
        if self.gradient_check:
            self._cbks.append(GradientCheck())    
        if self.verbose:
            self._cbks.append(Progress())        
        if isinstance(self._monitor, Monitor):            
            self._cbks.append(self._monitor)                
        if isinstance(self._early_stop, EarlyStop):            
            self._cbks.append(self._early_stop)            
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
                self._theta = self.theta_init
        else:            
            rng = np.random.RandomState(self.random_state)         
            self._theta = rng.randn(2)
# --------------------------------------------------------------------------- #
#                           EVALUATE EPOCH                                    #
# --------------------------------------------------------------------------- #
    def _evaluate_epoch(self, log=None):
        log = log or {}
        log['epoch'] = self._epoch
        log['learning_rate'] = self._eta
        log['theta'] = self._theta
        # Grab training cost if available
        if self._cost:
            log['train_cost'] = self._cost
        # Otherwise this is being called at train_begin and must be computed.
        else:
            log['train_cost'] = self._objective(self._theta)
        log['gradient'] = self._objective.gradient(self._theta)
        log['gradient_norm'] = np.linalg.norm(log['gradient'])
        self._final_result = log
        return log

# --------------------------------------------------------------------------- #
#                     INTERNAL AND EXTERNAL CALLBACKS                         #
# --------------------------------------------------------------------------- #

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        log = log or {}
        self._validate_params() 
        # Private variables
        self._epoch = 1
        self._batch = 1                
        self._converged = False        
        self._eta = copy.copy(self.learning_rate)
        self._gradient = None
        self._theta_new = None
        self._cost = None
        self._final_result = None     
        self._best_result = None      
        # Dependencies, data, and weights.
        self._compile()              
        self._init_weights()          
        log = self._evaluate_epoch(log)  
        self._cbks.on_train_begin(log) 

    def _get_results(self):
        # Obtain best results
        if self.early_stop:
            self._best_result = self._early_stop.best_results
        elif self._monitor:
            self._best_result = self._monitor.best_results
        else:
            self._best_result = self._final_result

        # Format thetas from final or best results as requested
        if self.get_best_weights:
            self.theta_ = self._best_result['theta']
        else:
            self.theta_ = self._final_result['theta']
        
        # Format final intercept and coefficient attributes
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]        

    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""                    
        log = log or {}
        self._get_results()
        self.n_iter_ = self._epoch
        self._cbks.on_train_end()

    def _begin_epoch(self, log=None):
        """Runs 'begin_epoch' methods on all callbacks."""            
        log = log or {}                    
        self._theta = self._theta_new if self._theta_new is not None else self._theta

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""        
        log = log or {}        
        # Capture current data representing current state of the optimization  
        log = self._evaluate_epoch(log)
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)          
        self._epoch += 1
        

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

        while (self._epoch <= self.epochs and not self._converged):

            self._begin_epoch()

            self._cost = self._objective(self._theta)

            self._theta_new, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self._theta))                    

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
                 val_size=0.3, schedule=Constant(), monitor=Monitor(),
                 scorer=R2(), early_stop=None, get_best_weights=True,
                 verbose=False, checkpoint=100, random_state=None, 
                 gradient_check=False):
                 
        super(GradientDescentEstimator, self).__init__(
            learning_rate = learning_rate,      
            epochs = epochs,
            theta_init = theta_init,
            optimizer = optimizer,
            objective = objective,                                          
            schedule=schedule,
            monitor=monitor,
            early_stop = early_stop,
            get_best_weights = get_best_weights,
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
    def _prepare_training_data(self, X, y):
        """Prepares X and y data for training."""
        data = self._data_processor.fit_transform(X, y)
        # Set attributes from data.
        for k, v in data.items():            
            setattr(self, k, v)
            # Attempt to extract feature names from the 'X' array  
            if np.ndim(v) > 1:
                if v.shape[1] > 1:
                    try:
                        self.features_ =  v.dtype.names                     
                    except:
                        self.features_ = None        

# --------------------------------------------------------------------------- #
#                               EVALUATION                                    #
# --------------------------------------------------------------------------- #
    def _evaluate_epoch(self, log=None):
        """Computes training costs, and optionally scores, for each epoch."""   
        log = log or {}
        log['epoch'] = self._epoch      
        log['learning_rate'] = self._eta    
        log['theta'] = self._theta            
        # Compute training costs and scores
        y_out = self._task.compute_output(self._theta, self.X_train_)
        log['train_cost'] = self._objective(self._theta, self.y_train_, y_out)
        # If there is a scoring object, compute scores
        if self.scorer:
            y_pred = self._task.predict(self._theta, self.X_train_)
            log['train_score'] = self._scorer(self.y_train_orig_, y_pred)

        # If we have a validation set, compute validation error and score
        if hasattr(self, 'X_val_'):
            if self.X_val_.shape[0] > 0:
                y_out_val = self._task.compute_output(self._theta, self.X_val_)
                log['val_cost'] = self._objective(self._theta, self.y_val_, y_out_val)        
                if self.scorer:
                    y_pred_val = self._task.predict(self._theta, self.X_val_)
                    log['val_score'] = self._scorer(self.y_val_orig_, y_pred_val)

        # If we have the gradient, i.e. not the first iteration, grab it
        if self._gradient is not None:
            log['gradient'] = self._gradient
        # Otherwise, compute it
        else:
            log['gradient'] = \
                self._objective.gradient(theta=self._theta, X=self.X_train_,
                                         y=self.y_train_, y_out=y_out)            
        # Compute the gradient norm        
        log['gradient_norm'] = \
                np.linalg.norm(log['gradient']) 
        self._final_result = log
        return log      

# --------------------------------------------------------------------------- #
#                  INTERNAL AND EXTERNAL CALLBACKS                            #
# --------------------------------------------------------------------------- #

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""            
        log = log or {}
        self._validate_params() 
        # Private variables
        self._epoch = 1
        self._batch = 1                
        self._converged = False        
        self._eta = copy.copy(self.learning_rate)
        self._gradient = None
        self._theta_new = None
        self._cost = None
        self._final_result = None
        self._best_result = None   
        self._features = None
        # Dependencies, data, and weights.
        self._compile(log)              
        self._prepare_training_data(log.get("X"),log.get("y"))        
        self._init_weights()            
        # Perform an epoch 0 evaluation on initial weights
        log = self._evaluate_epoch(log)
        self._cbks.on_train_begin(log)        

    def _begin_epoch(self, log=None):
        """Increment the epoch, evaluate using current parameters and shuffle the data."""            
        log = log or {}        
        # Shuffle data      
        rs = None
        if self.random_state:
            rs = self.random_state + self._epoch
        self.X_train_, self.y_train_ = shuffle_data(self.X_train_, self.y_train_, random_state=rs) 
        # Call 'on_epoch_begin' methods on callbacks.
        self._cbks.on_epoch_begin(self._epoch)               

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""            
        log = log or {}        
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch(log)                
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)      
        self._epoch += 1        

    def _begin_batch(self, log=None):
        log = log or {}
        self._theta = self._theta_new if self._theta_new is not None else self._theta

    def _end_batch(self, log=None):        
        log = log or {}
        log['batch'] = self._batch
        log['learning_rate'] = self._eta
        log['theta'] = self._theta
        log['train_cost'] = self._cost
        log['gradient'] = copy.copy(self._gradient)
        log['gradient_norm'] = np.linalg.norm(log['gradient'])            
        self._cbks.on_batch_end(self._batch, log)
        self._batch += 1

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
        train_log = {'X': X, 'y': y}
        self._begin_training(train_log)        

        while (self._epoch <= self.epochs and not self._converged):            

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X_train_, self.y_train_, batch_size=self.batch_size):

                self._begin_batch()
                
                # Compute model output
                y_out = self._task.compute_output(self._theta, X_batch)     

                # Compute costs
                self._cost = self._objective(self._theta, y_batch, y_out)

                # Compute gradient and update parameters 
                self._theta_new, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.copy(self._theta),  X=X_batch, y=y_batch,\
                        y_out=y_out)                       

                # Update batch log
                self._end_batch()

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
        return self._task.predict(self._theta, X)
    
    def _score(self, X, y):
        """Calculates scores during training."""
        # Called during training. Assumes data has valid format. 
        y_pred = self._task.predict(self._theta, X)
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
        self.blackbox_.report(features)

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT REGRESSOR                              #
# --------------------------------------------------------------------------- #
class GradientDescentRegressor(GradientDescentEstimator, RegressorMixin):
    """Gradient descent estimator for regression."""

    def __init__(self, learning_rate=0.01, epochs=1000, theta_init=None,
                 optimizer=Classic(), objective=MSE(), batch_size=None, 
                 val_size=0.3, schedule=None, monitor=Monitor(), 
                 scorer=R2(), early_stop=None, get_best_weights=True,
                 verbose=False, checkpoint=100, random_state=None, 
                 gradient_check=False):
        
        super(GradientDescentRegressor, self).__init__(\
            learning_rate = learning_rate,
            epochs = epochs,        
            theta_init = theta_init,
            optimizer = optimizer,
            objective = objective,        
            batch_size = batch_size,
            val_size = val_size,
            schedule=schedule,
            monitor=monitor,
            scorer = scorer,
            early_stop = early_stop,
            get_best_weights=get_best_weights,
            verbose = verbose,
            checkpoint = checkpoint,
            random_state = random_state,
            gradient_check = gradient_check   
        )

    def _compile(self, log=None):
        """Compiles required objects."""
        super(GradientDescentRegressor, self)._compile(log)        
        self._task = LinearRegression()        
        self._data_processor = RegressionDataProcessor(val_size=self.val_size, 
                                            random_state=self.random_state)

    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.X_train_.shape[1],), \
                    "Initial parameters theta must have shape (n_features+1,)."
            self._theta = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)                
            self._theta = rng.randn(self.X_train_.shape[1])      

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT CLASSIFIFER                            #
# --------------------------------------------------------------------------- #
class GradientDescentClassifier(GradientDescentEstimator, ClassifierMixin):
    """Gradient descent estimator for classification."""

    def __init__(self, learning_rate=0.01, epochs=1000, theta_init=None,
                 optimizer=Classic(), objective=CrossEntropy(), batch_size=None, 
                 val_size=0.3, schedule=None, monitor=Monitor(), 
                 scorer=Accuracy(), early_stop=None, get_best_weights=True,
                 verbose=False, checkpoint=100, random_state=None, 
                 gradient_check=False):
        
        super(GradientDescentClassifier, self).__init__(\
            learning_rate = learning_rate,
            epochs = epochs,        
            theta_init = theta_init,
            optimizer = optimizer,
            objective = objective,        
            batch_size = batch_size,
            val_size = val_size,
            schedule=schedule,
            monitor = monitor,
            scorer = scorer,
            early_stop = early_stop,
            get_best_weights=get_best_weights,
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
        self._data_processor = ClassificationDataProcessor(val_size=self.val_size, 
                                            random_state=self.random_state)

    def _init_weights_binary_classification(self):
        """Initializes weights for binary classification."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.n_features_,), \
                "Initial parameters theta must have shape (n_features+1)."
            self._theta = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)
            self._theta = rng.randn(self.n_features_)   

    def _init_weights_multiclass(self):
        """Initializes weights for multiclass classification."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.n_features_, self.n_classes_), \
                "Initial parameters theta must have shape (n_features+1, n_classes)."
            self._theta = self.theta_init
        else:
            rng = np.random.RandomState(self.random_state)                
            self._theta = rng.randn(self.n_features_, self.n_classes_)        

    def _init_weights(self):
        """Initializes model parameters."""        
        if self.y_train_.ndim == 1:
            self._init_weights_binary_classification()
        else:
            self._init_weights_multiclass()

