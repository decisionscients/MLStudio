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
"""Gradient Descent Module"""
from abc import ABC, abstractmethod, ABCMeta
import sys
import copy
import warnings

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator

from mlstudio.supervised.core.tasks import LinearRegression, LogisticRegression
from mlstudio.supervised.core.tasks import MultinomialLogisticRegression
from mlstudio.supervised.core.objectives import MSE, CrossEntropy
from mlstudio.supervised.core.objectives import CategoricalCrossEntropy
from mlstudio.supervised.core.objectives import Adjiman
from mlstudio.supervised.core.optimizers import GradientDescentOptimizer
from mlstudio.supervised.core.scorers import R2, Accuracy
from mlstudio.utils.data_analyzer import compute_gradient_norm
from mlstudio.utils.data_manager import batch_iterator, data_split, shuffle_data
from mlstudio.utils.data_manager import add_bias_term, encode_labels, one_hot_encode
from mlstudio.utils.data_manager import RegressionDataProcessor, ClassificationDataProcessor
from mlstudio.utils.validation import check_X, check_X_y, check_is_fitted
from mlstudio.utils.validation import validate_zero_to_one, validate_metric
from mlstudio.utils.validation import validate_objective, validate_optimizer
from mlstudio.utils.validation import validate_scorer
from mlstudio.utils.validation import validate_learning_rate_schedule
from mlstudio.utils.validation import validate_int, validate_string
from mlstudio.utils.validation import validate_metric
from mlstudio.utils.validation import validate_scorer, validate_bool
from mlstudio.utils.validation import validate_range, validate_monitor
from mlstudio.utils.validation import validate_array_like, validate_gradient_check
# =========================================================================== #
#                       GRADIENT DESCENT ABSTRACT                             #
# =========================================================================== #        
class GradientDescentAbstract(ABC,BaseEstimator):
    """Gradient Descent abstract base class."""
    def __init__(self, learning_rate=0.01, epochs=1000, objective=None,
                 theta_init=None, optimizer=None,  observers=None,
                 verbose=False, random_state=None):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.objective  = objective
        self.theta_init = theta_init
        self.optimizer = optimizer
        self.observers = observers
        self.verbose = verbose
        self.random_state = random_state

    # ----------------------------------------------------------------------- #
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

    # ----------------------------------------------------------------------- #
    def _copy_mutable_parameters(self):
        """Makes deepcopies of mutable parameters and makes them private members."""

        # Observers
        self._observers = copy.deepcopy(self.observers) if self.observers\
            else self.observers

        # The Optimizer algorithm
        if self.optimizer:
            self._optimizer = copy.deepcopy(self.optimizer)
        else:
            self._optimizer = GradientDescentOptimizer()

        # The objective function to be minimized.
        if self.objective:
            self._objective = copy.deepcopy(self.objective)
        else:
            self._objective = MSE()
    # ----------------------------------------------------------------------- #
    def _create_observer_attributes(self):
        """Adds each observer to model as an attribute."""
        # First, add any additional observers that should be attributes
        self._observers['blackbox_'] = BlackBox()        
        for name, observer in self._observers.items():
                setattr(self, name, observer)
    # ----------------------------------------------------------------------- #
    def _create_observer_list(self):
        """Adds all observers to the observer list that gets notified."""
        # Add any additional default observers to observer dictionary
        if self.verbose:
            self._observers['progress'] = Progress()

        self._observer_list = ObserverList()                
        for observer in self._observers.values():
            self._observer_list.append(observer)

        # Publish model parameters and instance on observer objects.
        self._observer_list.set_params(self.get_params())
        self._observer_list.set_model(self)            

    # ----------------------------------------------------------------------- #
    def _compile(self):        
        """Obtains, initializes object dependencies and registers observers."""

        self._copy_mutable_parameters()
        self._create_observer_attributes()
        self._create_observer_list()

    # ----------------------------------------------------------------------- #
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""
        d = {}
        d['epoch'] = self._epoch
        d['learning_rate'] = self._eta
        d['theta'] = self._theta
        d['train_cost'] = self._objective(self._theta)
        if self._gradient:
            d['gradient'] = self._gradient
            d['gradient_norm'] = compute_gradient_norm(self._gradient)
        self._current_state = d
    # ----------------------------------------------------------------------- #
    def _on_train_begin(self):
        """Initializes all data, objects, and dependencies.
        
        Parameters
        ----------
         : dict
            Data relevant this part of the process. Not used.
        """
        self._compile()    
        self._epoch = 0       
        self._theta = None
        self._theta_new = None
        self._gradient = None
        self._current_state = {}
        self._converged = False    
        self._init_weights()
        try:
            self._eta = self.schedule.initial_learning_rate
        except:
            self._eta = self.learning_rate
        
        self._observer_list.on_train_begin()

    # ----------------------------------------------------------------------- #
    def _on_train_end(self):
        """Finalizes training, formats attributes, and ensures object state is fitted."""
        self.n_iter_ = self._epoch 
        self.theta_ = self._theta
        self._observer_list.on_train_end()
    # ----------------------------------------------------------------------- #
    def _on_epoch_begin(self):
        """Initializes all data, objects, and dependencies.
        
        Parameters
        ----------
        epoch : int
            The current epoch number. Zero indexed.

        log : dict
            Data relevant this part of the process. Not used.
        """
        self._set_current_state()
        self._observer_list.on_epoch_begin(epoch=self._epoch, log=self._current_state)
    # ----------------------------------------------------------------------- #
    def _on_epoch_end(self):
        """Finalizes epoching, formats attributes, and ensures object state is fitted.
        
        Parameters
        ----------
        epoch : int
            The current epoch number. Zero indexed.
        
        log : dict
            Data relevant this part of the process. Not used.
        
        """
        self._observer_list.on_epoch_end(epoch=self._epoch, log=self._current_state)
        self._theta = self._theta_new        
        self._epoch += 1
    # ----------------------------------------------------------------------- #
    @abstractmethod
    def _init_weights(self):
        pass
    # ----------------------------------------------------------------------- #
    @abstractmethod
    def fit(self, X=None, y=None):
        pass

# =========================================================================== #
#                    GRADIENT DESCENT PURE OPTIMIZER                          #
# =========================================================================== #
class GradientDescentPureOptimizer(GradientDescentAbstract):
    """Performs pure optimization of an objective function."""

    def __init__(self, learning_rate=0.01, epochs=1000, objective=None,
                 theta_init=None, optimizer=None,  observers=None,
                 verbose=False, random_state=None):
        super(GradientDescentPureOptimizer, self).__init__(
            learning_rate = learning_rate,
            epochs = epochs,
            objective  = objective,
            theta_init = theta_init,
            optimizer = optimizer,
            observers = observers,
            verbose = verbose,
            random_state = random_state
        )

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

            self._theta_new, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self._theta))                    

            self._on_epoch_end()
            

        self._on_train_end()
        return self   

# =========================================================================== #
#                        GRADIENT DESCENT ESTIMATOR                           #
# =========================================================================== # 
"""Base class for gradient descent estimators. """
class GradientDescentEstimator(GradientDescentAbstract):
    """Gradient Descent abstract base class."""
    def __init__(self, learning_rate=0.01, learning_rate_schedule=None, epochs=1000, 
                 theta_init=None, optimizer=None, objective=None, observer=None, 
                 get_best_weights=True, verbose=False, checkpoint=100,  
                 random_state=None, gradient_check=False):

        self.learning_rate = learning_rate
        self.schedule = schedule
        self.epochs = epochs
        self.theta_init = theta_init
        self.optimizer = optimizer
        self.objective  = objective        
        self.observer = observer
        self.get_best_weights = get_best_weights
        self.verbose = verbose
        self.checkpoint = checkpoint
        self.random_state = random_state
        self.gradient_check = gradient_check
