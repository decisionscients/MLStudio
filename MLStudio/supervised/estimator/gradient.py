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
import copy
import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_random_state, check_array
from sklearn.utils.validation import check_X_y, check_is_fitted
import sys
import time
import uuid
import warnings

from mlstudio.supervised.estimator.callbacks import CallbackList, Callback
from mlstudio.supervised.estimator.early_stop import EarlyStop
from mlstudio.supervised.estimator.debugging import GradientCheck
from mlstudio.supervised.estimator.monitor import History, Progress, summary
from mlstudio.supervised.estimator.scorers import R2, Accuracy
from mlstudio.supervised.regression import LinearRegression
from mlstudio.supervised.logistic_regression import LogisticRegression
from mlstudio.utils.data_manager import batch_iterator, data_split, shuffle_data
from mlstudio.utils.data_analyzer import check_y
# --------------------------------------------------------------------------- #
#                          GRADIENT DESCENT                                   #
# --------------------------------------------------------------------------- #
class GradientDescent(ABC, BaseEstimator):
    """Base class gradient descent estimator."""

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
        return self.algorithm.name + ' with ' + self.variant       

    def _prepare_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        self.X = self.X_val = self.y = self.y_val = None
        # Convert input to numpy arrays just in case.
        self.X = np.array(X)
        self.y = np.array(y)
        # Add a column of ones to create the X design matrix
        self.X_design = np.insert(self.X, 0, 1.0, axis=1)          
        # If early stopping, set aside a proportion of the data for the validation set
        if self.early_stop:
            if self.early_stop.val_size:
                self.X_design, self.X_val, self.y, self.y_val = \
                    data_split(self.X_design, self.y, 
                    test_size=self.early_stop.val_size, random_state=self.random_state)
        # Designate the number of features and classes (outputs)
        self.n_classes_ = check_y(self.y)
        self.n_features_ = self.X_design.shape[1]        

    def _evaluate_epoch(self, log=None):
        """Computes training (and validation) costs and scores."""
        log = log or {}
        # Create copy of scorer to avoid sklearn check_estimator assertion error
        # Sklearn check_estimator asserts that parameters are immutable during 
        # fit. Therefore calls to scorer are dissallowed during fit. So, we'll
        # create a copy of the scorer for internal evaluation. 
        scorer = copy.copy(self.scorer)
        # Compute costs 
        y_pred = self.algorithm.hypothesis(self.X_design, self._theta)
        log['train_cost'] = self.algorithm.compute_cost(self.y, y_pred, self._theta)
        log['train_score'] = scorer(self.y, y_pred)
        if self.early_stop:
            if self.early_stop.val_size:
                y_pred_val = self.algorithm.hypothesis(self.X_val, self._theta)
                log['val_cost'] = self.algorithm.compute_cost(self.y_val, y_pred_val, self._theta)        
                log['val_score'] = scorer(self.y_val, y_pred_val)

        return log

    def _init_callbacks(self):
        # Initialize callback list
        self._cbks = CallbackList()        
        # Instantiate monitor callbacks and add to list
        self.history_ = History()
        self._cbks.append(self.history_)
        self._progress = Progress()        
        self._cbks.append(self._progress)
        # Add early stop if object injected.
        if self.early_stop:
            self._cbks.append(self.early_stop)
        # Add gradient checking if object injected.
        if self.gradient_check:
            self._cbks.append(self.gradient_check)        
        # Initialize all callbacks.
        self._cbks.set_params(self.get_params())
        self._cbks.set_model(self)
    
    def _init_weights(self, X, y):
        """Initializes weights"""       
        if self.theta_init is not None:
            if y.ndim == 1:
                assert self.theta_init.shape == (self.n_features_,), \
                    "Initial parameters theta must have shape (n_features,)."
            else:
                assert self.theta_init.shape == (self.n_features_, self.n_classes_), \
                    "Initial parameters theta must have shape (n_classes, n_features)."
            self._theta = self.theta_init
        else:
            self.random_state_ = check_random_state(self.random_state)  
            if y.ndim == 1:          
                self._theta = self.random_state_.randn(self.n_features_)
            else:
                self._theta = self.random_state_.randn(self.n_features_, self.n_classes_)

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self._epoch = 0
        self._batch = 0
        X = log.get('X')
        y = log.get('y')
        self.converged_ = False
        self.is_fitted_ = False        
        self._prepare_data(X,y)
        self._init_weights(self.X_design, self.y)   
        self._init_callbacks()
        self._cbks.on_train_begin(log)
        
    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self._cbks.on_train_end()
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        self.n_iter_ = self._epoch
        self.is_fitted_ = True

    def _begin_epoch(self):
        """Increment the epoch count and shuffle the data."""
        self._epoch += 1
        self.X_design, self.y = shuffle_data(self.X_design, self.y)
        self._cbks.on_epoch_begin(self._epoch)

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}
        # Update log with current learning rate and parameters theta
        log['X'] = self.X_design
        log['y'] = self.y
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

        while (self._epoch < self.epochs and not self.converged_):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X_design, self.y, batch_size=self.batch_size):

                self._begin_batch()
                
                # Compute prediction
                y_pred = self.algorithm.hypothesis(X_batch, self._theta)

                # Compute costs
                J = self.algorithm.compute_cost(y_batch, y_pred, self._theta)                
                
                # Format batch log with weights and cost
                batch_log = {'batch': self._batch, 'batch_size': X_batch.shape[0],
                             'theta': self._theta.copy(), 'train_cost': J}

                # Compute gradient
                gradient = self.algorithm.compute_gradient(X_batch, y_batch, y_pred, self._theta)

                # Update parameters.
                self._theta = self._theta - self.learning_rate * gradient

                # Update batch log
                self._end_batch(batch_log)

            # Wrap up epoch
            self._end_epoch()

        self._end_training()
        return self         

    def predict(self, X):
        """Computes prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : prediction
        """
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X)

        return self.algorithm.predict(X, self._theta)
    
    def score(self, X, y):
        """Computes scores using the scorer parameter.

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
        y_pred = self.predict(X)
        return self.scorer(y, y_pred)

    def summary(self, features=None):
        summary(self.history_, features)

# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT REGRESSOR                              #
# --------------------------------------------------------------------------- #
class GradientDescentRegressor(GradientDescent, RegressorMixin):
    """Gradient descent estimator for regression."""

    def __init__(self, name=None, learning_rate=0.01, batch_size=None, 
                 theta_init=None,  epochs=1000, algorithm=LinearRegression(),
                 scorer=R2(), early_stop=False, verbose=False, checkpoint=100, 
                 random_state=None, gradient_check=False):

        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.algorithm = algorithm
        self.scorer = scorer
        self.early_stop = early_stop
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.gradient_check = gradient_check        

    def _prepare_data(self, X, y):
        """Creates the X design matrix and one-hot encodes y if appropriate."""
        check_X_y(X,y)
        super(GradientDescentRegressor, self)._prepare_data(X,y)        


# --------------------------------------------------------------------------- #
#                     GRADIENT DESCENT CLASSIFIFER                            #
# --------------------------------------------------------------------------- #
class GradientDescentClassifier(GradientDescent, ClassifierMixin):
    """Gradient descent estimator for classification."""

    def __init__(self, name=None, learning_rate=0.01, batch_size=None, 
                 theta_init=None,  epochs=1000, algorithm=LogisticRegression(),
                 scorer=Accuracy(), early_stop=False, verbose=False, checkpoint=100, 
                 random_state=None, gradient_check=False):

        self.name = name
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.theta_init = theta_init
        self.epochs = epochs
        self.algorithm = algorithm
        self.scorer = scorer
        self.early_stop = early_stop
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.gradient_check = gradient_check    
 
    def _prepare_data(self, X, y):
        """Creates the X design matrix and one-hot encodes y if appropriate."""
        if len(np.unique(y)>2):
            y = (np.arange(np.max(y) + 1) == y[:, None]).astype(float)
        super(GradientDescentClassifier, self)._prepare_data(X,y)