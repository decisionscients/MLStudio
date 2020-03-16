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
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin
import warnings

from mlstudio.utils.data_manager import batch_iterator, data_split, shuffle_data
from mlstudio.supervised.estimator.callbacks import CallbackList, Callback
from mlstudio.supervised.estimator.monitor import History, Progress, summary
from mlstudio.supervised.estimator.early_stop import EarlyStop
# --------------------------------------------------------------------------- #

class GradientDescent(ABC, BaseEstimator, RegressorMixin, metaclass=ABCMeta):
    """Base class gradient descent estimator."""

    DEFAULT_METRIC = 'mse'

    def __init__(self, name=None, learning_rate=0.01, batch_size=None, 
                 theta_init=None,  epochs=1000, early_stop=False, patience=5, 
                 precision=0.001, cost='quadratic', metric='mse',  val_size=0.0, 
                 verbose=False, checkpoint=100, seed=None):
        # Public parameters
        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.early_stop = early_stop
        self.patience = patience
        self.precision = precision  
        self.metric = metric        
        self.verbose = verbose
        # Private parameters
        self._cost = cost                
        self._val_size = val_size      
        self._checkpoint = checkpoint        
        self._seed = seed  

        # Public instance variables        
        self.converged = False
        self.history = None
        # Private instance variables        
        self._fitted = False
        self._theta = None
        self._epoch = 0
        self._batch = 0        
        self._cbks = None
        self._X = self._y = self._X_val = self._y_val = None
        self._regularizer = lambda x: 0
        self._regularizer.gradient = lambda x: 0        
        # Private Functions        
        self.scorer = None
        self._cost_function = None
        self._convergence_monitor = None
        # Read only attributes / properties        
        self._algorithm = None
        self._coef = None
        self._intercept = None
        self._epochs_trained = 0

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def coef(self):
        return self._coef

    @property
    def intercept(self):
        return self._intercept

    @property
    def epochs_trained(self):
        return self._epochs_trained

    def _set_algorithm_name(self):
        """Sets the name of the algorithm for plotting purposes."""
        if self.batch_size is None:
            self._algorithm = 'Batch Gradient Descent'
        elif self.batch_size == 1:
            self._algorithm = 'Stochastic Gradient Descent'
        else:
            self._algorithm = 'Minibatch Gradient Descent'

    @abstractmethod
    def _set_name(self):
        pass

    def set_params(self, **kwargs):
        """Sets parameters to **kwargs and validates."""
        super().set_params(**kwargs)
        self._validate_params()
        return self

    def _validate_params(self):
        """Validate parameters."""
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError("learning_rate must provide an int or a float.")
        if self.batch_size is not None:
            if not isinstance(self.batch_size, int):
                raise TypeError("batch_size must provide an integer.")            
        if self.theta_init is not None:
            if not isinstance(self.theta_init, (list, pd.core.series.Series, np.ndarray)):
                raise TypeError("theta must be an array like object.")            
        if not isinstance(self.epochs, int):
            raise TypeError("epochs must be an integer.")        
        if self.early_stop:
            if not isinstance(self.early_stop, (bool,EarlyStop)):
                raise TypeError("early stop is not a valid EarlyStop callable.")
        if self.early_stop:
            if not isinstance(self._val_size, float) or self._val_size < 0 or self._val_size >= 1:
                raise ValueError("val_size must be a float between 0 and 1.")
        if not isinstance(self.patience, int):
            raise ValueError("patience must be an integer")
        if not isinstance(self.precision, float) or self.precision < 0 or self.precision >= 1:
            raise ValueError("precision must be a float between 0 and 1.")            
        if self.metric is not None:
            if not isinstance(self.metric, str):
                raise TypeError("metric must be string containing name of metric for scoring")                
        if not isinstance(self.verbose, bool):
            raise TypeError("verbose must be either True or False")
        if self._checkpoint is not None:
            if not isinstance(self._checkpoint, int):
                raise TypeError(
                    "checkpoint must be a positive integer or None.")
            elif self._checkpoint < 0:
                raise ValueError(
                    "checkpoint must be a positive integer or None.")
        if self._seed is not None:
            if not isinstance(self._seed, int):
                raise TypeError("seed must be a positive integer.")

    def _validate_data(self, X, y=None):
        """Validates and reports n_features."""
        if not isinstance(X, (np.ndarray)):
            raise TypeError("X must be of type np.ndarray")
        if y is not None:
            if not isinstance(y, (np.ndarray)):
                raise TypeError("y must be of type np.ndarray")            
            if len(y.shape) > 1:
                raise ValueError("y should be of shape (m,), not %s" % str(y.shape))
            if X.shape[0] != y.shape[0]:
                raise ValueError("X and y have incompatible lengths")        

    def _prepare_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        self._X = self._X_val = self._y = self._y_val = None
        # Add a column of ones to train the intercept term
        self._X = X
        self._X_design = np.insert(X, 0, 1, axis=1)  
        self._y = y
        # Set aside val_size training observations for validation set 
        if self._val_size:
            self._X_design, self._X_val, self._y, self._y_val = \
                data_split(self._X_design, self._y, 
                test_size=self._val_size, seed=self._seed)

    def _evaluate_epoch(self, log=None):
        """Computes training (and validation) costs and scores."""
        log = log or {}
        # Compute costs 
        y_pred = self._predict(self._X_design)
        log['train_cost'] = self._cost_function(y=self._y, y_pred=y_pred)
        if self._val_size:
            y_pred_val = self._predict(self._X_val)
            log['val_cost'] = self._cost_function(y=self._y_val, y_pred=y_pred_val)        
        # Compute scores 
        if self.metric is not None:            
            log['train_score'] = self.score(self._X_design, self._y)
            if self._val_size:
                log['val_score'] = self.score(self._X_val, self._y_val)        

        return log


    @abstractmethod
    def _get_cost_function(self):
        """Obtains the cost function for the cost parameter."""
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod        
    def _get_scorer(self):
        """Obtains the scoring function for the metric parameter."""
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    def _get_convergence_monitor(self):
        
        if self.metric and self._val_size > 0:
            convergence_monitor = EarlyStop(early_stop=self.early_stop,
                                                        monitor='val_score',
                                                        precision=self.precision,
                                                        patience=self.patience)
        else:
            convergence_monitor = EarlyStop(early_stop=self.early_stop,
                                                        monitor='train_cost',
                                                        precision=self.precision,
                                                        patience=self.patience)
         
        return convergence_monitor

    def _compile(self):
        """Obtains external objects and add key functions to the log."""
        self._cost_function = self._get_cost_function()
        self.scorer = self._get_scorer()        
        self._convergence_monitor = self._get_convergence_monitor()

    def _init_callbacks(self):
        # Initialize callback list
        self._cbks = CallbackList()        
        # Instantiate monitor callbacks and add to list
        self.history = History()
        self._cbks.append(self.history)
        self._progress = Progress()        
        self._cbks.append(self._progress)
        # Add additional callbacks if available
        if isinstance(self._convergence_monitor, Callback):
            self._cbks.append(self._convergence_monitor)
        # Initialize all callbacks.
        self._cbks.set_params(self.get_params())
        self._cbks.set_model(self)
    
    def _init_weights(self):
        """Initializes weights"""        
        if self.theta_init is not None:
            if self.theta_init.shape[0] != self._X_design.shape[1]:
                raise ValueError("theta_init shape mispatch. Expected shape %s,"
                                 " but theta_init.shape = %s." % ((self._X_design.shape[1],1),
                                 self.theta_init.shape))
            else:
                self._theta = np.atleast_2d(self.theta_init).reshape(-1,1)
        else:
            n_features = self._X_design.shape[1]
            np.random.seed(seed=self._seed)
            self._theta = np.random.normal(size=n_features).reshape(-1,1)

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self._epoch = 0
        self._batch = 0
        self.converged = False
        self._fitted = False
        self._validate_params()
        self._validate_data(log.get('X'), log.get('y'))        
        self._prepare_data(log.get('X'), log.get('y'))
        self._set_name()
        self._init_weights()   
        self._compile()
        self._init_callbacks()
        self._cbks.on_train_begin(log)
        
    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self._cbks.on_train_end()
        self._intercept = self._theta[0]
        self._coef = self._theta[1:]
        self._epochs_trained = self._epoch
        self._fitted = True

    def _begin_epoch(self):
        """Increment the epoch count and shuffle the data."""
        self._epoch += 1
        self._X_design, self._y = shuffle_data(self._X_design, self._y, seed=self._seed)
        if self._seed:
            self._seed += 1
        self._cbks.on_epoch_begin(self._epoch)

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}
        # Update log with current learning rate and parameters theta
        log['epoch'] = self._epoch
        log['learning_rate'] = self.learning_rate
        log['theta'] = self._theta.copy()     
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch(log)
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)

    def _begin_batch(self, log=None):
        self._batch += 1
        self._cbks.on_batch_begin(self._batch)

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
        
        while (self._epoch < self.epochs and not self.converged):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self._X_design, self._y, batch_size=self.batch_size):

                self._begin_batch()
                # Compute prediction
                y_pred = self._predict(X_batch)
                # Compute costs
                J = self._cost_function(
                    y=y_batch, y_pred=y_pred) + self._regularizer(self._theta)
                # Update batch log with weights and cost
                batch_log = {'batch': self._batch, 'batch_size': X_batch.shape[0],
                             'theta': self._theta.copy(), 'train_cost': J}
                # Compute gradient 
                gradient = self._cost_function.gradient(
                    X_batch, y_batch, y_pred) + self._regularizer.gradient(self._theta)
                # Update parameters              
                self._theta -= self.learning_rate * gradient
                # Update batch log
                self._end_batch(batch_log)

            # Wrap up epoch
            self._end_epoch()

        self._end_training()
        return self
    
    def _linear_prediction(self, X):
        """Computes prediction as linear combination of inputs and thetas."""
        if X.shape[1] == self._theta.shape[0]:
            y_pred = X.dot(self._theta)
        else:
            if not self._fitted:
                raise Exception("This %(name)s instance is not fitted "
                                 "yet" % {'name': type(self).__name__})              
            y_pred = self._intercept + X.dot(self._coef)
        return y_pred            

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    @abstractmethod
    def score(self, X, y):
        pass

    def summary(self):
        summary(self.history)
