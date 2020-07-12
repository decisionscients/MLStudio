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

from mlstudio.supervised.core.objectives import MSE, CrossEntropy, Adjiman
from mlstudio.supervised.core.objectives import CategoricalCrossEntropy
from mlstudio.supervised.core.optimizers import GradientDescentOptimizer
from mlstudio.supervised.core.regularizers import L2
from mlstudio.supervised.core.scorers import R2, Accuracy
from mlstudio.supervised.core.applications import LinearRegression
from mlstudio.supervised.core.applications import LogisticRegression
from mlstudio.supervised.core.applications import MultinomialLogisticRegression
from mlstudio.supervised.observers.base import Observer, ObserverList
from mlstudio.supervised.observers.early_stop import EarlyStop
from mlstudio.supervised.observers.history import BlackBox, Progress
from mlstudio.utils.data_analyzer import n_classes, n_features
from mlstudio.utils.data_manager import AddBiasTerm, unpack_parameters
from mlstudio.utils.data_manager import RegressionDataProcessor
from mlstudio.utils.data_manager import LogisticRegressionDataProcessor
from mlstudio.utils.data_manager import MulticlassDataProcessor, batch_iterator
from mlstudio.utils.data_manager import check_y
from mlstudio.utils.validation import check_X, check_X_y, check_is_fitted
from mlstudio.visual.text import OptimizationReport
# =========================================================================== #
#                       GRADIENT DESCENT ABSTRACT                             #
# =========================================================================== #        
class GDAbstract(ABC,BaseEstimator):
    """Gradient Descent abstract base class."""

    def __init__(self, eta0=0.01, learning_rate=None, epochs=1000, 
                 objective=None, theta_init=None, optimizer=None,  
                 early_stop=False, verbose=False, 
                 random_state=None):

        self.eta0 = eta0
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.objective = objective
        self.theta_init = theta_init        
        self.optimizer = optimizer 
        self.early_stop = early_stop
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
    def _copy_mutable_parameters(self, log=None):
        """Makes copies of mutable parameters and makes them private members."""

        self._eta = copy.copy(self.eta0)
        
        self._learning_rate = copy.deepcopy(self.learning_rate) if \
            self.learning_rate else self.learning_rate

        self._early_stop = copy.deepcopy(self.early_stop) if self.early_stop\
            else self.early_stop

        self._optimizer = copy.deepcopy(self.optimizer) if self.optimizer\
            else GradientDescentOptimizer()

        self._objective = copy.deepcopy(self.objective) if self.objective\
            else self.objective            

    # ----------------------------------------------------------------------- #
    def _obtain_implicit_dependencies(self, log=None):
        """Initialize delegated objects."""            
        pass

    # ----------------------------------------------------------------------- #
    def _initialize_observers(self, log=None):
        """Initialize remaining observers. Create and initialize observer list."""

        self._observer_list = ObserverList()                   

        self.blackbox_ = BlackBox()
        self._observer_list.append(self.blackbox_)

        if self.verbose:
            self._observer_list.append(Progress())

        if self._learning_rate:
            self._observer_list.append(self._learning_rate)

        if self._early_stop:
            self._observer_list.append(self._early_stop)
        
        # Publish model parameters and estimator instance on observer objects.
        self._observer_list.set_params(self.get_params())
        self._observer_list.set_model(self)            

    # ----------------------------------------------------------------------- #
    def _compile(self, log=None):        
        """Obtains, initializes object dependencies and registers observers."""
        self._copy_mutable_parameters(log)
        self._obtain_implicit_dependencies(log)
        self._initialize_observers(log)

    # ----------------------------------------------------------------------- #
    def _on_train_begin(self, log=None):
        """Initializes all data, objects, and dependencies.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. 
        """
        log = log or {}
        self._compile(log)    
        self._epoch = 0      
        self._batch = 0 
        self._theta = None
        self._gradient = None
        self._current_state = {}
        self._converged = False    

        # Initialize training on observers
        self._observer_list.on_train_begin()
        # Prepares data and adds data to estimator as attributes.
        if log:            
            self._prepare_data(log.get('X'), log.get('y'))
        # Weights are initialized based upon the number of features in the dataset 
        self._init_weights()

    # ----------------------------------------------------------------------- #
    def _on_train_end(self, log=None):
        """Finalizes training, formats attributes, and ensures object state is fitted.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. Currently not used, but 
            kept for future applications. 
        
        """
        log = log or {}
        self.n_iter_ = self._epoch         
        self._observer_list.on_train_end()
        self._get_results()
    # ----------------------------------------------------------------------- #
    def _on_epoch_begin(self, log=None):
        """Initializes all data, objects, and dependencies.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. Currently not used, but 
            kept for future applications. 
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
            kept for future applications. 
        
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
            kept for future applications. 
        
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
            kept for future applications. 
        
        """
        log = log or {}
        self._observer_list.on_batch_end(batch=self._batch, log=log)            
        self._batch += 1 

    # ----------------------------------------------------------------------- #
    @abstractmethod
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""
        pass           
    # ----------------------------------------------------------------------- #
    def _prepare_data(self):
        pass
    # ----------------------------------------------------------------------- #
    @abstractmethod
    def _init_weights(self):
        pass    
    # ----------------------------------------------------------------------- #
    @abstractmethod
    def fit(self, X=None, y=None):
        pass    

    # ----------------------------------------------------------------------- #
    def _get_results(self):
        # Set parameter attributes
        self.theta_ = self._theta
        self.intercept_, self.coef_ = unpack_parameters(self.theta_)


# =========================================================================== #
#                    GRADIENT DESCENT PURE OPTIMIZER                          #
# =========================================================================== #
class GDPureOptimizer(GDAbstract):
    """Performs pure optimization of an objective function."""

    def __init__(self, eta0=0.01, learning_rate=None, epochs=1000, 
                 objective=None, theta_init=None, optimizer=None,  
                 early_stop=False, verbose=False, 
                 random_state=None):
        super(GDPureOptimizer, self).__init__(
            eta0 = eta0,
            learning_rate=learning_rate,
            epochs = epochs,
            objective = objective,
            theta_init = theta_init,
            optimizer = optimizer,
            early_stop=early_stop,
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
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""        
        s = {}
        s['epoch'] = self._epoch
        s['eta'] = self._eta
        s['theta'] = self._theta
        s['train_cost'] = self._objective(self._theta)
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

            cost = self._objective(self._theta)

            self._theta, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self._theta))                    

            self._on_epoch_end()

        self._on_train_end()
        return self   
        

# =========================================================================== #
#                        GRADIENT DESCENT ESTIMATOR                           #
# =========================================================================== # 
class GDEstimator(GDAbstract):
    """Gradient descent base class for all estimators.
    
    Performs gradient descent optimization to estimate the parameters theta
    that best fit the data.

    Parameters
    ----------
    eta0 : float
        The initial learning rate on open interval (0,1) 

    learning_rate : LearningRateSchedule object or None (default=None)
        This optional parameter can be a supported LearningRateSchedule
        object.

    epochs : int
        The number of epochs to execute

    objective : An Objective object
        The objective function to be minimized

    batch_size : None or int (default=None) 
        The number of observations to include in each batch. This also 
        specifies the gradient descent variant according to the following:

            Batch_Size      Variant
            ----------      -----------------------
            None            Batch Gradient Descent
            1               Stochastic Gradient Descent
            Other int       Minibatch Gradient Descent

    theta_init : array_like
        Contains the initial values for the parameters theta. Should include
        the bias parameter in addition to the feature parameters.

    optimizer : An Optimizer object or None
        The optimization algorithm to use. If None, the generic 
        GradientDescentOptimizer will be used.

    early_stop : an EarlyStop object or None (default=None)
        Class responsible for stopping the optimization process once
        training has stabilized. 

    scorer : a Scorer object (default=None)
        Supported Scorer object for estimating performance.

    val_size : float in interval [0,1) (default=0.3)
        The proportion of the training set to allocate a validation set

    verbose : Bool or Int
        If False, the parameter is ignored. If an integer is provided, this 
        will be the number of epochs between progress reports.

    random_state : int or None
        If an int, this will be the random state used anywhere pseudo-randomization
        occurs.
    
    """
    def __init__(self, eta0=0.01, learning_rate=None, 
                 epochs=1000, objective=None, batch_size=None,  theta_init=None, 
                 optimizer=None, early_stop=None, scorer=None, 
                 val_size=0.3, verbose=False, random_state=None):
        super(GDEstimator, self).__init__(
            eta0 = eta0,
            learning_rate=learning_rate,
            epochs = epochs,
            objective = objective,
            theta_init = theta_init,
            optimizer = optimizer,            
            early_stop=early_stop,            
            verbose = verbose,
            random_state = random_state    
        )
        self.scorer = scorer
        self.val_size = val_size
        self.batch_size = batch_size               

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
            objective = " Optimizing " + self.objective.name
        except:
            objective = ""            
        
        try:
            optimizer = " using " + self.optimizer.name
        except:
            optimizer = ""

        try:
            early_stop = " and " + self.early_stop.name
        except:
            early_stop = ""
               
        try: 
            regularizer = " with  " + self.objective.regularizer_name
        except:
            regularizer = ""
        
        return self._application.name + " for " + self.variant + objective +\
            regularizer + optimizer + early_stop
    # ----------------------------------------------------------------------- #
    def _copy_mutable_parameters(self, log=None):
        """Makes copies of mutable parameters and makes them private members."""
        super(GDEstimator, self)._copy_mutable_parameters(log=log)
        self.scorer_ = copy.deepcopy(self.scorer) if self.scorer\
            else self.scorer

    # ----------------------------------------------------------------------- #    
    def _prepare_data(self, X, y):
        """Prepares X and y data for training.
        
        X and y data is prepared and if a Performance observer with a 
        validation set size parameter is set, the data is split. The 
        data is then added to the estimator as attributes.

        """
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
        # Set n_features_ as the number of features plus the intercept term
        self.n_features_ = n_features(self.X_train_)

    # ----------------------------------------------------------------------- #
    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.n_features_,), \
                "Initial parameters theta must have shape (n_features+1,)."
            self._theta = self.theta_init
            self._theta
        else:
            # Random normal initialization for weights.
            rng = np.random.RandomState(self.random_state)                
            self._theta = rng.randn(self.n_features_) 
            # Set the bias initialization to zero
            self._theta[0] = 0
            self._theta = self._theta

    # ----------------------------------------------------------------------- #
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""
        s= {}
        s['epoch'] = self._epoch      
        s['eta'] = self._eta    
        s['theta'] = self._theta 
        
        # Compute training costs 
        y_out = self._application.compute_output(self._theta, self.X_train_)
        s['train_cost'] = self._objective(self._theta, self.y_train_, y_out)
        # Compute training score
        s['train_score'] = self._score(self.X_train_, self.y_train_)

        # If we have a validation set, compute validation error and score
        if self.val_size:
            if self.X_val_.shape[0] > 0:
                # Compute validation error 
                y_out_val = self._application.compute_output(self._theta, self.X_val_)
                s['val_cost'] = self._objective(self._theta, self.y_val_, y_out_val)                
                # Compute validation score
                s['val_score'] = self._score(self.X_val_, self.y_val_)

        # Grab Gradient. Note: 1st iteration, the gradient will be None
        s['gradient'] = self._gradient
        # Compute the gradient norm if not first iteration
        s['gradient_norm'] = None
        if self._gradient is not None:
            s['gradient_norm'] = np.linalg.norm(self._gradient) 
        # This reflects current state for the epoch sent to all observers.
        self._current_state = s
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
                
                # Compute model output
                y_out = self._application.compute_output(self._theta, X_batch)     

                # Compute costs
                cost = self._objective(self._theta, y_batch, y_out)

                # Format batch log
                log = {'batch': self._batch,'theta': self._theta, 
                       'train_cost': cost}

                # Compute gradient and update parameters 
                self._theta, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.copy(self._theta),  X=X_batch, y=y_batch,\
                        y_out=y_out)                       

                # Update batch log
                log['gradient'] = self._gradient
                log['gradient_norm'] = np.linalg.norm(self._gradient) 
                self._on_batch_end(log=log)

            # Wrap up epoch
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
        check_is_fitted(self)
        X = check_X(X)
        X = AddBiasTerm().fit_transform(X)
        return self._application.predict(self.theta_, X)    

    # ----------------------------------------------------------------------- #    
    def _score(self, X, y):
        """Calculates scores during as the beginning of each training epoch."""        
        y = check_y(y)
        y_pred = self._application.predict(self._theta, X)
        try:
            return self.scorer_(y, y_pred, X)
        except Exception as e:            
            print(e)
        

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
        score based upon the scorer object.
        
        """
        y_pred = self.predict(X)        
        try:
            return self.scorer_(y, y_pred, X)    
        except Exception as e:
            print(e)
        

    # ----------------------------------------------------------------------- #    
    def summary(self):  
        """Prints and optimization report. """
        optimization = OptimizationReport(model=self)
        optimization.report()
                    
# =========================================================================== #
#                        GRADIENT DESCENT REGRESSOR                           #
# =========================================================================== # 
class GDRegressor(GDEstimator):
    """Gradient descent regression class."""
    def __init__(self, eta0=0.01, learning_rate=None, epochs=1000, 
                 objective=MSE(), batch_size=None, theta_init=None, 
                 optimizer=GradientDescentOptimizer(),  
                 early_stop=None, scorer=R2(), val_size=0.3,
                 verbose=False, random_state=None):
        super(GDRegressor, self).__init__(
            eta0 = eta0,
            learning_rate=learning_rate,
            epochs = epochs,
            objective = objective,
            batch_size = batch_size,
            theta_init = theta_init,
            optimizer = optimizer,      
            early_stop=early_stop,
            scorer = scorer,
            val_size = val_size,            
            verbose = verbose,
            random_state = random_state                
        )

    # ----------------------------------------------------------------------- #    
    def _obtain_implicit_dependencies(self, log=None):
        super(GDRegressor, self)._obtain_implicit_dependencies()                
        # Set the application that will be computing the output and predictions. 
        self._application = LinearRegression()

        # Instantiates the data processor for regression
        self._data_processor = RegressionDataProcessor(val_size=self.val_size,
                                random_state=self.random_state)          

# =========================================================================== #
#                        GRADIENT DESCENT CLASSIFIER                          #
# =========================================================================== # 
class GDClassifier(GDEstimator):
    """Gradient descent classification class."""
    def __init__(self, eta0=0.01, learning_rate=None, epochs=1000, 
                 objective=CrossEntropy(), batch_size=None, theta_init=None, 
                 optimizer=GradientDescentOptimizer(),  early_stop=None,
                 scorer=Accuracy(), val_size=0.3, verbose=False, 
                 random_state=None):
        super(GDClassifier, self).__init__(
            eta0 = eta0,
            learning_rate=learning_rate,
            epochs = epochs,
            objective = objective,
            batch_size = batch_size,
            theta_init = theta_init,
            optimizer = optimizer,    
            early_stop=early_stop,                    
            scorer = scorer,
            val_size = val_size,            
            verbose = verbose,
            random_state = random_state                
        )

    # ----------------------------------------------------------------------- #
    def _on_train_begin(self, log=None):
        """Initializes all data, objects, and dependencies."""        

        self.n_classes_ = n_classes(log.get('y'))

        if self.n_classes_ > 2:
            self._binary_classification = False
        else:
            self._binary_classification = True
        
        super(GDClassifier, self)._on_train_begin(log=log)

    # ----------------------------------------------------------------------- #    
    def _obtain_implicit_dependencies(self, log=None):
        """Obtain the application and data processor classes."""
        super(GDClassifier, self)._obtain_implicit_dependencies()                

        if self._binary_classification:
            self._application = LogisticRegression()
            self._data_processor = LogisticRegressionDataProcessor(\
                                val_size=self.val_size, 
                                random_state=self.random_state)          
            if not self.objective:  
                self._objective = CrossEntropy()
        else:
            self._application = MultinomialLogisticRegression()        
            self._data_processor = MulticlassDataProcessor(\
                                val_size=self.val_size, 
                                random_state=self.random_state)
            if not self._objective:
                self._objective = CategoricalCrossEntropy()
        

    # ----------------------------------------------------------------------- #                                    
    def _init_weights(self):
        """Initializes weights for a binary and multiclass classification problem."""
        if self._binary_classification:
            super(GDClassifier, self)._init_weights()
        else:
            if self.theta_init is not None:
                assert self.theta_init.shape == (self.n_features_, self.n_classes_), \
                    "Initial parameters theta must have shape (n_features+1, n_classes)."
                self._theta = self.theta_init
            else:
                rng = np.random.RandomState(self.random_state)                
                self._theta = rng.randn(self.n_features_, self.n_classes_)   

    # ----------------------------------------------------------------------- #                                        
    def predict_proba(self, X):
        """Predicts class probabilities."""
        check_is_fitted(self)
        X = check_X(X)
        X = AddBiasTerm().fit_transform(X)        
        return self._application.predict_proba(self.theta_, X)
