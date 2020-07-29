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
                 check_gradient=False, gradient_checker=None):

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
        self.gradient_checker = gradient_checker

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

    @property
    def theta(self):
        return self._theta

    @property
    def train_data_package(self):
        return self._train_data_package

    @property
    def X_train(self):
        return self._X_train 

    @property
    def y_train(self):
        return self._y_train 

    @property
    def X_val(self):
        try:
            return self._X_val
        except:
            warnings.warn("This estimator has non X_val attribute.")

    @property
    def y_val(self):
        try:
            return self._y_val
        except:
            warnings.warn("This estimator has non y_val attribute.")

    def get_blackbox(self):
        return self._blackbox

    # TODO: add evaluator and profiler observers
    def get_evaluator(self):
        pass

    def get_profiler(self):
        pass

    # ----------------------------------------------------------------------- #
    def _compile(self, log=None):
        """Makes copies of mutable parameters and makes them private members."""

        self._eta = copy.copy(self.eta0)
        self._task = copy.deepcopy(self.task) 
        self._observer_list = copy.deepcopy(self.observer_list)           
        self._optimizer = copy.deepcopy(self.optimizer)
        self._progress = copy.deepcopy(self.progress)
        self._summary = copy.deepcopy(self.summary) 
        self._gradient_checker = copy.deepcopy(self.gradient_checker)
        self._blackbox = copy.deepcopy(self.blackbox)

        # Optional dependencies
        self._learning_rate = copy.deepcopy(self.learning_rate) if \
            self.learning_rate else self.learning_rate

        self._early_stop = copy.deepcopy(self.early_stop) if self.early_stop\
            else self.early_stop        

    # ----------------------------------------------------------------------- #
    def _initialize_state(self, log=None):
        """Initializes variables that represent teh state of the estimator."""
        self._epoch = 0      
        self._batch = 0 
        self._train_data_package = None
        self._theta = None
        self._gradient = None
        self._converged = False

    # ----------------------------------------------------------------------- #    
    def _prepare_data(self, X, y):
        """Prepares data for training and creates data and metadata attributes."""        
        if self.val_size:
            self._train_data_package = self._task.prepare_train_val_data(X, y, self.val_size, \
                    self.random_state)      
            self._X_train = self._train_data_package['X_train']['data']          
            self._X_val = self._train_data_package['X_val']['data']
            self._y_train = self._train_data_package['y_train']['data']
            self._y_val = self._train_data_package['y_val']['data']
        else:
            self._train_data_package = self._task.prepare_train_data(X, y, self.random_state)
            self._X_train = self._train_data_package['X_train']['data']          
            self._y_train = self._train_data_package['y_train']['data']

    # ----------------------------------------------------------------------- #
    def _initialize_observers(self, log=None):
        """Initialize remaining observers. Create and initialize observer list."""        

        self._observer_list.append(self._blackbox)
        self._observer_list.append(self._summary)

        if self.verbose:
            self._observer_list.append(self._progress)

        if self._learning_rate:
            self._observer_list.append(self._learning_rate)

        if self._early_stop:
            self._observer_list.append(self._early_stop)

        if self.check_gradient:
            self._observer_list.append(self._gradient_checker)
        
        # Publish model parameters and estimator instance on observer objects.
        self._observer_list.set_params(self.get_params())
        self._observer_list.set_model(self)            
        self._observer_list.on_train_begin(log)
    # ----------------------------------------------------------------------- #
    def _init_weights(self):
        """Initialize weights with user values or random values."""
        self._theta = self._task.init_weights(self.theta_init) 
    # ----------------------------------------------------------------------- #
    def _on_train_begin(self, log=None):
        """Compiles the estimator, initializes weights, observers, and state"""
        log = log or {}        
        validation.validate_estimator(self)
        self._compile(log)    
        self._initialize_state(log)
        self._prepare_data(log.get('X'), log.get('y'))
        self._initialize_observers(log)
        self._init_weights()
    # ----------------------------------------------------------------------- #
    def _on_train_end(self, log=None):
        """Finalizes training and posts model parameter attributes."""
        log = log or {}
        self.n_iter_ = self._epoch         
        self.intercept_, self.coef_ = unpack_parameters(self._theta)
        self._observer_list.on_train_end()        
    # ----------------------------------------------------------------------- #
    def _on_epoch_begin(self, log=None):
        """Initializes the epoch and notifies observers."""
        log = log or {}        
        self._observer_list.on_epoch_begin(epoch=self._epoch, log=log)
    # ----------------------------------------------------------------------- #
    def _on_epoch_end(self, log=None):
        """Finalizes epoching and notifies observers."""
        log = log or {}
        self._observer_list.on_epoch_end(epoch=self._epoch, log=log)
        self._epoch += 1
    # ----------------------------------------------------------------------- #            
    def _on_batch_begin(self, log=None):
        """Initializes the batch and notifies observers."""
        log = log or {}
        self._observer_list.on_batch_begin(batch=self._batch, log=log)        
    # ----------------------------------------------------------------------- #            
    def _on_batch_end(self, log=None):
        """Wraps up the batch and notifies observers."""
        log = log or {}
        self._observer_list.on_batch_end(batch=self._batch, log=log)            
        self._batch += 1 
    # ----------------------------------------------------------------------- #            
    def compute_output(self, theta, X):
        """Computes output of the current iteration.

        For linear regression, this is the linear combination of the inputs
        and the weights. For binary classification the output is the sigmoid
        probability of the positive class. For the multiclass case,
        the output is the softmax probabilities. 

        Parameters
        ----------
        theta : array-like (n_features,) or (n_features, n_classes)
            The model parameters at the current iteration

        X : array-like (n_samples, n_features)
            The features including a constant bias term.
        
        Returns
        -------
        y_out : float
        """

        return self._task.compute_output(theta, X)

    # ----------------------------------------------------------------------- #            
    def compute_loss(self, theta, y, y_out):
        """Computes the average loss of the model.

        Parameters
        ----------
        theta : array-like (n_features,) or (n_features, n_classes)
            The model parameters at the current iteration

        y : array-like of shape (n_samples,)
            True target values

        y_out : array-like of shape (n_samples,)
            The real-valued output of the model.

        Returns
        -------
        J : float
        """
        return self._task.compute_loss(theta, y, y_out)

    # ----------------------------------------------------------------------- #            
    def train_epoch(self):
        """Trains a single epoch."""
        self._on_epoch_begin()

        for X_batch, y_batch in batch_iterator(self._X_train, self._y_train, batch_size=self.batch_size):
            self._on_batch_begin()

            y_out = self.compute_output(self._theta, X_batch)     
            cost = self.compute_loss(self._theta, y_batch, y_out)
            # Grab theta for the batch log before it is updated
            log = {'batch': self._batch,'theta': self._theta, 
                    'train_cost': cost}
            # Update the model parameters and return gradient for monitoring purposes.
            self._theta, self._gradient = self._optimizer(gradient=self._task.loss.gradient, \
                learning_rate=self._eta, theta=copy.copy(self._theta),  X=X_batch, y=y_batch,\
                    y_out=y_out)                       
            
            log['gradient_norm'] = np.linalg.norm(self._gradient) 
            self._on_batch_end(log=log)
            
        log = {'epoch': self._epoch, 'theta': self._theta}        
        self._on_epoch_end(log)


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
        log = {'X': X, 'y': y}
        self._on_train_begin(log)        

        while (self._epoch < self.epochs and not self._converged):            
            self.train_epoch()

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
        return self._task.predict(X, self._theta)    


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
        return self._task.score(X, y, self._theta)               

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
                self._theta = self.theta_init
        else:            
            rng = np.random.RandomState(self.random_state)         
            self._theta = rng.randn(2)    

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

            cost = self._objective(self._theta)

            self._theta, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self._theta))                    

            self._on_epoch_end()

        self._on_train_end()
        return self   
# %%
