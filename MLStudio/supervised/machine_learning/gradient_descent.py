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
from mlstudio.supervised.core.tasks import LinearRegression
from mlstudio.supervised.core.tasks import LogisticRegression
from mlstudio.supervised.core.tasks import MultinomialLogisticRegression
from mlstudio.supervised.observers.base import Observer, ObserverList
from mlstudio.supervised.observers.monitor import BlackBox, Performance, Progress
from mlstudio.supervised.observers.learning_rate import LearningRateSchedule
from mlstudio.utils.data_manager import AddBiasTerm, unpack_parameters
from mlstudio.utils.data_manager import RegressionDataProcessor
from mlstudio.utils.data_manager import LogisticRegressionDataProcessor
from mlstudio.utils.data_manager import MulticlassDataProcessor, batch_iterator
from mlstudio.utils.validation import check_X, check_X_y, check_is_fitted
from mlstudio.visual.text import OptimizationReport
# =========================================================================== #
#                       GRADIENT DESCENT ABSTRACT                             #
# =========================================================================== #        
class GradientDescentAbstract(ABC,BaseEstimator):
    """Gradient Descent abstract base class."""

    def __init__(self, learning_rate=0.01, epochs=1000, 
                 theta_init=None, optimizer=None,  observers=None,
                 verbose=False, random_state=None):

        self.learning_rate = learning_rate
        self.epochs = epochs
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
    def _validate_params(self):
        """Performs parameter validation."""
        # Learning Rate
        if not isinstance(self.learning_rate, LearningRateSchedule):                    
            validate_range(param=self.learning_rate, minimum=0,
                           maximum=1, param_name='learning_rate',
                           left="open", right="open")        
        # Epochs
        validate_int(param=self.epochs, param_name='epochs')
        # Theta Init
        if self.theta_init:
            validate_array_like(param=self.theta_init, param_name='theta_init')
        # Optimizer
        if self.optimizer:
            validate_optimizer(self.optimizer)
        # Observers 
        if self.observers:
            validate_observers(self.observers)
        # Verbose 
        validate_bool(param=self.verbose, param_name='verbose')
        # Random State
        if self.random_state:
            validate_int(param=self.random_state, param_name='random_state')

    # ----------------------------------------------------------------------- #
    def _copy_mutable_parameters(self):
        """Makes deepcopies of mutable parameters and makes them private members."""

        # Learning Rate
        if isinstance(self.learning_rate, LearningRateSchedule):
            self._learning_rate = copy.deepcopy(self.learning_rate)
        else:
            self._learning_rate = self.learning_rate

        # Observers. Note we pull of the val_size from the Performance
        # observer if it exists. The val_size will be used by the data
        # preparation method.           
        self._val_size = 0.0     
        self._observers = {}
        if self.observers:
            for name, observer in self.observers.items():                
                self._observers[name] = copy.deepcopy(observer)     
                if isinstance(observer, Performance):
                    self._val_size = observer.val_size       

        # The Optimizer algorithm
        if self.optimizer:
            self._optimizer = copy.deepcopy(self.optimizer)
        else:
            self._optimizer = GradientDescentOptimizer()

    # ----------------------------------------------------------------------- #
    def _create_observer_list(self):
        """Adds all observers to the observer list that gets notified."""
        # Add any additional default observers to observer dictionary
        if self.verbose:
            self._observers['progress'] = Progress()

        if isinstance(self._learning_rate, Observer):
            self._observers['learning_rate'] = self._learning_rate

        self._observer_list = ObserverList()                
        for observer in self._observers.values():
            self._observer_list.append(observer)

        # Publish model parameters and instance on observer objects.
        self._observer_list.set_params(self.get_params())
        self._observer_list.set_model(self)            

    # ----------------------------------------------------------------------- #
    def _compile(self):        
        """Obtains, initializes object dependencies and registers observers."""

        self._set_objective()        
        self._copy_mutable_parameters()
        self._obtain_implicit_dependencies()
        self._create_observer_list()

    # ----------------------------------------------------------------------- #
    def _on_train_begin(self, log=None):
        """Initializes all data, objects, and dependencies.
        
        Parameters
        ----------
        log : dict
            Data relevant this part of the process. 
        """
        log = log or {}
        self._compile()    
        self._epoch = 0      
        self._batch = 0 
        self._theta = None
        self._theta_new = None
        self._gradient = None
        self._current_state = {}
        self._converged = False    
        self._final_result = None
        self._best_result = None
        if log:            
            self._prepare_training_data(log.get('X'), log.get('y'))
        self._init_weights()
        try:
            self._eta = self.learning_rate.initial_learning_rate
        except:
            self._eta = self.learning_rate
        
        self._observer_list.on_train_begin()

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
        self._theta = self._theta_new        
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
    def _set_objective(self):
        pass
    # ----------------------------------------------------------------------- #
    @abstractmethod
    def _obtain_implicit_dependencies(self):
        pass    
    # ----------------------------------------------------------------------- #
    @abstractmethod
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""
        pass           
    # ----------------------------------------------------------------------- #
    def _prepare_training_data(self):
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
        # Determine if best or final weights should be stored.
        borf = 'final'
        for observer in self._observers.values():
            if isinstance(observer, Performance):
                borf = observer.best_or_final
        
        # Format results
        if borf == 'best':
            self.final_theta_ = self._theta
            self.final_intercept_, self.final_coef_ = unpack_parameters(self.final_theta_)
            self.theta_ = self.best_results_.get('theta')
            self.intercept_, self.coef_ = unpack_parameters(self.theta_)
        else:
            self.theta_ = self._theta
            self.intercept_, self.coef_ = unpack_parameters(self.theta_)


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
            theta_init = theta_init,
            optimizer = optimizer,
            observers = observers,
            verbose = verbose,
            random_state = random_state
        )
        self.objective = objective

    # ----------------------------------------------------------------------- #        
    def _set_objective(self):
        self._objective = self.objective

    # ----------------------------------------------------------------------- #        
    def _obtain_implicit_dependencies(self):
        pass

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
        s['learning_rate'] = self._eta
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

            self._theta_new, self._gradient = self._optimizer(gradient=self._objective.gradient, \
                    learning_rate=self._eta, theta=copy.deepcopy(self._theta))                    

            self._on_epoch_end()
            

        self._on_train_end()
        return self   
        
    # ----------------------------------------------------------------------- #            
    def _get_results(self):
        self.theta_ = self._theta
        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]

        

# =========================================================================== #
#                        GRADIENT DESCENT ESTIMATOR                           #
# =========================================================================== # 
class GradientDescentEstimator(GradientDescentAbstract):
    """Gradient descent base class for all estimators.
    
    Performs gradient descent optimization to estimate the parameters theta
    that best fit the data.

    Parameters
    ----------
    learning_rate : float or LearningRateSchedule observer
        This can be a float in (0,1) or a LearningRateSchedule object

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

    theta_init : array_like
        Contains the initial values for the parameters theta. Should include
        the bias parameter in addition to the feature parameters.

    optimizer : An Optimizer object or None
        The optimization algorithm to use. If None, the generic 
        GradientDescentOptimizer will be used.

    observers : dict
        A dictionary of observers. The observers will be added to the 
        estimator as attributes. The keys in the observers dictionary will 
        serve as the attribute names for future reference. 

    verbose : Bool or Int
        If False, the parameter is ignored. If an integer is provided, this 
        will be the number of epochs between progress reports.

    random_state : int or None
        If an int, this will be the random state used anywhere pseudo-randomization
        occurs.
    
    """
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=None, 
                 theta_init=None, optimizer=None, observers=None, 
                 verbose=False, random_state=None):
        super(GradientDescentEstimator, self).__init__(
            learning_rate = learning_rate,
            epochs = epochs,
            theta_init = theta_init,
            optimizer = optimizer,
            observers = observers,
            verbose = verbose,
            random_state = random_state    
        )
        self.batch_size = batch_size               

    # ----------------------------------------------------------------------- #                
    @property
    def task(self):
        return self._task

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
        """Returns the estimator description."""                   
        task = self._task.name 

        # Optimizer Title
        optimizer_title = ""       
        if not isinstance(self._optimizer, GradientDescentOptimizer):
            optimizer_title = " (" + self._optimizer.__class__.__name__\
                + " Optimization)"
        
        # Regularizer Title        
        try: 
            regularizer_title = " with " + self._objective.regularizer_name
        except:
            regularizer_title = ""
        
        return task + ' by ' + self.variant + optimizer_title + \
            regularizer_title

    # ----------------------------------------------------------------------- #    
    def _prepare_training_data(self, X, y):
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

    # ----------------------------------------------------------------------- #
    def _init_weights(self):
        """Initializes parameters."""
        if self.theta_init is not None:
            assert self.theta_init.shape == (self.X_train_.shape[1],), \
                "Initial parameters theta must have shape (n_features+1,)."
            self._theta = self.theta_init
        else:
            # Random normal initialization for weights.
            rng = np.random.RandomState(self.random_state)                
            self._theta = rng.randn(self.X_train_.shape[1]) 
            # Set the bias initialization to zero
            self._theta[0] = 0

    # ----------------------------------------------------------------------- #
    def _set_current_state(self):
        """Takes snapshot of current state and performance."""
        s= {}
        s['epoch'] = self._epoch      
        s['learning_rate'] = self._eta    
        s['theta'] = self._theta 
        
        # Compute training costs 
        y_out = self._task.compute_output(self._theta, self.X_train_)
        s['train_cost'] = self._objective(self._theta, self.y_train_, y_out)
        # If there is a scoring object, compute scores
        if self.scorer:
            y_pred = self._task.predict(self._theta, self.X_train_)
            s['train_score'] = self._scorer(self.y_train_, y_pred)

        # If we have a validation set, compute validation error and score
        if hasattr(self, 'X_val_'):
            if self.X_val_.shape[0] > 0:
                y_out_val = self._task.compute_output(self._theta, self.X_val_)
                s['val_cost'] = self._objective(self._theta, self.y_val_, y_out_val)        
                if self.scorer:
                    y_pred_val = self._task.predict(self._theta, self.X_val_)
                    s['val_score'] = self._scorer(self.y_val_, y_pred_val)

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
                y_out = self._task.compute_output(self._theta, X_batch)     

                # Compute costs
                cost = self._objective(self._theta, y_batch, y_out)

                # Format batch log
                log = {'batch': self._batch,'theta': self._theta, 
                       'train_cost': cost}

                # Compute gradient and update parameters 
                self._theta_new, self._gradient = self._optimizer(gradient=self._objective.gradient, \
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
        return self._task.predict(self.theta_, X)    

    # ----------------------------------------------------------------------- #    
    def _score(self, X, y):
        """Calculates scores during as the beginning of each training epoch."""        
        y_pred = self._task.predict(self.theta_, X)
        if self._scorer is None:
            raise Exception("Unable to compute score. No scorer object provided.")
        return self._scorer(y, y_pred)

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
        if self._scorer is None:
            raise Exception("Unable to compute score. No scorer object provided.")        
        return self._scorer(y, y_pred)    

    # ----------------------------------------------------------------------- #    
    def summary(self, reports=None, features=None):  
        """Prints and optimization report. 

        Parameters
        ----------
        reports : list default=['summary', 'performance', 'critical_points',
                                'features', 'hyperparameters']
            The reports in the order to be rendered. The valid report names are:
                'summary' : prints summary data for optimzation
                'hyperparameters' : prints the hyperparameters used for the optimization
                'performance' : prints performance in terms of cost and scores
                'critical_points' : prints cost and scores at critical points 
                    during the optimization
                'features' : prints the best or final intercept and coeficients 
                    by feature name if feature names are available. Best results 
                    are printed if the Performance observer is used and the 
                    'best_or_final' parameter = 'best'. Otherwise, final results
                    will be printed.

        features : list of str
            A list containing the names of the features in the data set. 
        """
        optimization = OptimizationReport(model=self, reports=reports, features=features)
        optimization.report()
                    
# =========================================================================== #
#                        GRADIENT DESCENT REGRESSOR                           #
# =========================================================================== # 
class GradientDescentRegressor(GradientDescentEstimator):
    """Gradient descent base class for all estimators."""
    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=None, 
                 objective=None, theta_init=None, optimizer=None,  
                 scorer=None, observers=None, verbose=False, 
                 random_state=None):
        super(GradientDescentRegressor, self).__init__(
            learning_rate = learning_rate,
            epochs = epochs,
            theta_init = theta_init,
            optimizer = optimizer,            
            observers = observers,
            verbose = verbose,
            random_state = random_state,    
            batch_size=batch_size
        )
        self.scorer = scorer
        self.objective  = objective

    # ----------------------------------------------------------------------- #
    def _set_objective(self):        
        """Defaults to mean squared error loss with L2 regularization."""
        self._objective = self.objective if self.objective else \
            MSE()

    # ----------------------------------------------------------------------- #    
    def _copy_mutable_parameters(self, log=None):
        super(GradientDescentEstimator, self)._copy_mutable_parameters()
        self._scorer = copy.deepcopy(self.scorer) if self.scorer else \
            self.scorer

    # ----------------------------------------------------------------------- #    
    def _obtain_implicit_dependencies(self):
        super(GradientDescentRegressor, self)._obtain_implicit_dependencies()                
        # Set the task that will be computing the output and predictions. 
        self._task = LinearRegression()

        # Instantiates the regression data processor for the 
        # _prepare_training_data method 
        self._data_processor = RegressionDataProcessor(val_size=self._val_size, 
                                            random_state=self.random_state)          


        