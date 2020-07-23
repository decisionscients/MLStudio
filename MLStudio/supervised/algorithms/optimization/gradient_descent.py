#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : gradient_descent.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                      #
# Last Modified : Saturday, June 13th 2020, 9:52:07 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
"""Gradient Descent Module"""
from abc import ABC, abstractmethod, abstractproperty
import collections
import copy
import warnings
from pathlib import Path
import site
import tracemalloc
PROJECT_DIR = Path(__file__).resolve().parents[4]
site.addsitedir(PROJECT_DIR)

import dependency_injector.containers as containers
import dependency_injector.providers as providers
import numpy as np
from sklearn.base import BaseEstimator

from mlstudio.supervised.algorithms.optimization.observers import base
from mlstudio.supervised.algorithms.optimization.observers import monitor
from mlstudio.supervised.algorithms.optimization.observers import early_stop
from mlstudio.supervised.algorithms.optimization.services import activations
from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.supervised.algorithms.optimization.services import tasks
from mlstudio.utils.data_analyzer import get_features
from mlstudio.utils.data_manager import unpack_parameters
from mlstudio.utils.data_manager import batch_iterator, AddBiasTerm
from mlstudio.utils import validation
# =========================================================================== #
#                              GRADIENT DESCENT                               #
# =========================================================================== #        
class GradientDescent(BaseEstimator):
    """Gradient descent base class for all estimators.
    
    Performs gradient descent optimization to estimate the parameters theta
    that best fit the data.

    Parameters
    ----------
    eta0 : float
        The initial learning rate on open interval (0,1) 

    epochs : int
        The number of epochs to execute        

    batch_size : None or int (default=None) 
        The number of observations to include in each batch. This also 
        specifies the gradient descent variant according to the following:

            Batch_Size      Variant
            ----------      -----------------------
            None            Batch Gradient Descent
            1               Stochastic Gradient Descent
            Other int       Minibatch Gradient Descent        


    val_size : float in interval [0,1) (default=0.3)
        The proportion of the training set to allocate a validation set

    theta_init : array_like
        Contains the initial values for the parameters theta. Should include
        the bias parameter in addition to the feature parameters.

    optimizer : An Optimizer object or None
        The optimization algorithm to use. If None, the generic 
        GradientDescentOptimizer will be used.

    metric : a Metric object (default=None)
        Supported Metric object for estimating performance.        

    early_stop : an EarlyStop object or None (default=None)
        Class responsible for stopping the optimization process once
        training has stabilized. 

    learning_rate : LearningRateSchedule object or None (default=None)
        This optional parameter can be a supported LearningRateSchedule
        object.

    observer_list : an ObserverListobject
        Manages observers and subscriptions

    progress : Progress observer object
        Reports optimization statistics each 'verbose' epochs.

    blackblox : Blackbox Object
        Tracks training and validation metrics during optimization. 

    verbose : Bool or Int
        If False, the parameter is ignored. If an integer is provided, this 
        will be the number of epochs between progress reports.

    summary : Summary object
        Reports summary data at end of training.

    random_state : int or None
        If an int, this will be the random state used anywhere pseudo-randomization
        occurs.
    
    """

    def __init__(self, task, eta0=0.01, epochs=1000,  batch_size=None,  val_size=0.3, 
                 theta_init=None, optimizer=None, early_stop=None, 
                 learning_rate=None,  observer_list=None, progress=None, 
                 blackbox=None, summary=None, verbose=False, random_state=None,
                 check_gradient=False, gradient_check=None):

        self.task = task
        self.eta0 = eta0
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_size = val_size
        self.theta_init = theta_init
        self.optimizer = optimizer                    
        self.early_stop=early_stop            
        self.learning_rate = learning_rate
        self.observer_list = observer_list
        self.progress = progress
        self.blackbox = blackbox
        self.summary = summary
        self.verbose = verbose
        self.random_state = random_state    
        self.check_gradient = check_gradient
        self.gradient_check = gradient_check

    # ----------------------------------------------------------------------- #                
    @property
    def variant(self):
        """Returns the gradient descent variant based upon the batch size."""
        if self.batch_size is None:
            variant = "Batch Gradient Descent"
        elif self.batch_size == 1:
            variant = "Stochastic Gradient Descent"   
        else:
            variant = "Minibatch Gradient Descent"   
        return variant

    # ----------------------------------------------------------------------- #                
    @property
    def description(self):
        """Creates and returns the estimator description."""                   

        try:
            schedule = " with " + self.learning_rate.name + " Learning Rate Schedule"
        except:
            schedule = ""           

        try:
            objective = " Optimizing " + self.task.loss.name
        except:
            objective = ""            
        
        try:
            optimizer = " with " + self.optimizer.name
        except:
            optimizer = ""

        try:
            early_stop = " implementing " + self.early_stop.name
        except:
            early_stop = ""
               
        try: 
            regularizer = " with  " + self.task.loss.regularizer.name 
        except:
            regularizer = ""        
        
        return self._task.name + " with " + self.variant + optimizer +\
            objective + regularizer + optimizer + early_stop + schedule

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

    # ----------------------------------------------------------------------- #
    def _compile(self, log=None):
        """Makes copies of mutable parameters and makes them private members."""

        self._eta = copy.copy(self.eta0)
        self._task = copy.deepcopy(self.task) 
        self._observer_list = copy.deepcopy(self.observer_list)           
        self._optimizer = copy.deepcopy(self.optimizer)
        self._progress = copy.deepcopy(self.progress)
        self._summary = copy.deepcopy(self.summary) 
        self._gradient_check = copy.deepcopy(self.gradient_check)

        # Attributes
        self.blackbox_ = copy.deepcopy(self.blackbox)        

        # Optional dependencies
        self._learning_rate = copy.deepcopy(self.learning_rate) if \
            self.learning_rate else self.learning_rate

        self._early_stop = copy.deepcopy(self.early_stop) if self.early_stop\
            else self.early_stop        

    # ----------------------------------------------------------------------- #
    def _initialize_observers(self, log=None):
        """Initialize remaining observers. Create and initialize observer list."""        

        self._observer_list.append(self.blackbox_)
        self._observer_list.append(self._summary)

        if self.verbose:
            self._observer_list.append(self._progress)

        if self._learning_rate:
            self._observer_list.append(self._learning_rate)

        if self._early_stop:
            self._observer_list.append(self._early_stop)

        if self.check_gradient:
            self._observer_list.append(self._gradient_check)
        
        # Publish model parameters and estimator instance on observer objects.
        self._observer_list.set_params(self.get_params())
        self._observer_list.set_model(self)            
        self._observer_list.on_train_begin()

    # ----------------------------------------------------------------------- #
    def _initialize_state(self, log=None):
        """Initializes variables that represent teh state of the estimator."""
        self._epoch = 0      
        self._batch = 0 
        self.theta_ = None
        self._gradient = None
        self._converged = False

    # ----------------------------------------------------------------------- #    
    def get_data(self):

    # ----------------------------------------------------------------------- #    
    def _prepare_data(self, X, y):
        """Prepares data for training and creates data and metadata attributes."""        
        data = self._task.prepare_data(X, y, self.val_size)                
        for k, v in data.items():     
            k = k + "_"
            setattr(self, k, v)

    # ----------------------------------------------------------------------- #
    def _init_weights(self):
        """Initialize weights with user values or random values."""
        self.theta_ = self._task.init_weights(self.theta_init) 

    # ----------------------------------------------------------------------- #
    def _on_train_begin(self, log=None):
        """Initializes all data, objects, and dependencies.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. 
        """
        log = log or {}        
        validation.validate_estimator(self)
        self._compile(log)    
        self._initialize_state(log)
        self._prepare_data(log.get('X'), log.get('y'))
        self._initialize_observers(log)
        self._init_weights()

    # ----------------------------------------------------------------------- #
    def _on_train_end(self, log=None):
        """Finalizes training, formats attributes, and ensures object state is fitted.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. Currently not used, but 
            kept for future tasks. 
        
        """
        log = log or {}
        self._memory_monitor.stop()
        self.n_iter_ = self._epoch         
        self._observer_list.on_train_end()
        self._format_results()
    # ----------------------------------------------------------------------- #
    def _on_epoch_begin(self, log=None):
        """Initializes all data, objects, and dependencies.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. Currently not used, but 
            kept for future tasks. 
        """
        log = log or {}
        self._set_current_state()
        self._observer_list.on_epoch_begin(epoch=self._epoch, log=self._current_state)
    # ----------------------------------------------------------------------- #
    def _on_epoch_end(self, log=None):
        """Finalizes epoching, formats attributes, and ensures object state is fitted.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. Currently not used, but 
            kept for future tasks. 
        
        """
        log = log or {}
        self._observer_list.on_epoch_end(epoch=self._epoch, log=self._current_state)
        self._epoch += 1

    # ----------------------------------------------------------------------- #            
    def _on_batch_begin(self, log=None):
        """Initializes the batch and notifies observers.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. Currently not used, but 
            kept for future tasks. 
        
        """
        log = log or {}
        self._observer_list.on_batch_begin(batch=self._batch, log=log)        


    # ----------------------------------------------------------------------- #            
    def _on_batch_end(self, log=None):
        """Wraps up the batch and notifies observers.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. Currently not used, but 
            kept for future tasks. 
        
        """
        log = log or {}
        self._observer_list.on_batch_end(batch=self._batch, log=log)            
        self._batch += 1 

    # ----------------------------------------------------------------------- #            
    def compute_output(self, theta, X):
        return self._task.compute_output(theta, X)

    # ----------------------------------------------------------------------- #            
    def compute_loss(self, theta, y, y_out):
        return self._task.compute_loss(theta, y, y_out)

    # ----------------------------------------------------------------------- #
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""
        s= {}
        s['epoch'] = self._epoch      
        s['eta'] = self._eta    
        s['theta'] = self.theta_ 

        s['current_memory'], s['peak_memory'] = self._memory_monitor.get_traced_memory()
        
        y_out = self._task.compute_output(self.theta_, self.X_train_)
        s['train_cost'] = self._task.compute_loss(self.theta_, self.y_train_, y_out)        
        s['train_score'] = self._score(self.X_train_, self.y_train_)

        # Check not only val_size but also for empty validation sets 
        if self.val_size:
            if self.X_val_.shape[0] > 0:                
                y_out_val = self._task.compute_output(self.theta_, self.X_val_)
                s['val_cost'] = self._task.compute_loss(self.theta_, self.y_val_, y_out_val)                                
                s['val_score'] = self._score(self.X_val_, self.y_val_)

        s['gradient'] = self._gradient
        s['gradient_norm'] = None
        if self._gradient is not None:
            s['gradient_norm'] = np.linalg.norm(self._gradient) 

        self._current_state = s
    
    # ----------------------------------------------------------------------- #
    def _format_results(self):
        """Format the attributes that hold the optimization solution."""         
        self.intercept_, self.coef_ = unpack_parameters(self.theta_)

    # ----------------------------------------------------------------------- #    
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
        self : returns instance of self
        """        
        train_log = {'X': X, 'y': y}
        self._on_train_begin(train_log)        

        while (self._epoch < self.epochs and not self._converged):            
            self._on_epoch_begin()

            for X_batch, y_batch in batch_iterator(self.X_train_, self.y_train_, batch_size=self.batch_size):
                self._on_batch_begin()

                y_out = self.compute_output(self.theta_, X_batch)     
                cost = self.compute_loss(self.theta_, y_batch, y_out)
                # Grab theta for the batch log before it is updated
                log = {'batch': self._batch,'theta': self.theta_, 
                       'train_cost': cost}
                # Update the model parameters and return gradient for monitoring purposes.
                self.theta_, self._gradient = self._optimizer(gradient=self._task.loss.gradient, \
                    learning_rate=self._eta, theta=copy.copy(self.theta_),  X=X_batch, y=y_batch,\
                        y_out=y_out)                       
                log['gradient'] = self._gradient
                log['gradient_norm'] = np.linalg.norm(self._gradient) 
                self._on_batch_end(log=log)

            self._on_epoch_end()
        self._on_train_end()
        return self 

    # ----------------------------------------------------------------------- #    
    def predict(self, X):
        """Computes prediction on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        Returns
        -------
        y_pred : prediction
        """
        validation.check_is_fitted(self)
        X = AddBiasTerm().fit_transform(X)
        return self._task.predict(self.theta_, X)    

    # ----------------------------------------------------------------------- #    
    def score_internal(self, X, y):
        """Computes the training (and validation) scores during training.

        This private method computes scores during training to monitor 
        optimization performance. Unlike the public score method, the 
        input data has already been processed and therefore, it contains
        the intercept term. 

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features+1)
            The input data

        y : array-like of shape (n_samples,) 
        
        Returns
        -------
        score 
        """        
        y_pred = self._task.predict(self.theta_, X)
        return self._task.score(y, y_pred, self.n_features_)
        

    # ----------------------------------------------------------------------- #    
    def score(self, X, y):
        """Computes scores for test data after training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data
        
        y : array_like of shape (n_samples,) 
            The target variable.
        
        Returns
        -------
        score based upon the metric object.
        
        """
        y_pred = self.predict(X)
        return self._task.score(y, y_pred, self.n_features_)        
        

    # ----------------------------------------------------------------------- #    
    def summarize(self):  
        """Prints and optimization report. """
        self._summary.report()      




# =========================================================================== #
#                    GRADIENT DESCENT PURE OPTIMIZER                          #
# =========================================================================== #
class GD(BaseEstimator):
    """Performs pure optimization of an objective function."""

    def __init__(self, eta0=0.01, epochs=1000, theta_init=None,
                 objective=None,  optimizer=None,  learning_rate=None,
                 blackbox=None, verbose=False, random_state=None):

        self.eta0 = eta0
        self.learning_rate=learning_rate
        self.epochs = epochs
        self.objective = objective
        self.theta_init = theta_init
        self.optimizer = optimizer
        self.verbose = verbose
        self.random_state = random_state               

    # ----------------------------------------------------------------------- #
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

    # ----------------------------------------------------------------------- #
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""        
        s = {}
        s['epoch'] = self._epoch
        s['eta'] = self._eta
        s['theta'] = self.theta_
        s['train_cost'] = self._objective(self.theta_)
        s['gradient'] = self._gradient
        s['gradient_norm'] = None
        if self._gradient is not None:
            s['gradient_norm'] = np.linalg.norm(self._gradient)
        self._current_state = s

    # ----------------------------------------------------------------------- #            
    def fit(self, X=None, y=None):
        """Performs the optimization of the objective function..
        
        Parameters
        ----------
        objective : object derived from Objective class
            The objective function to be optimized

        Returns
        -------
        self
        """
        
        self._on_train_begin()

        while (self._epoch < self.epochs and not self._converged):

            self._on_epoch_begin()

            cost = self._objective(self.theta_)

            self.theta_, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self.theta_))                    

            self._on_epoch_end()

        self._on_train_end()
        return self   
# %%
