#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : base_regression.py                                                #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 14th 2020, 10:27:39 pm                        #
# Last Modified : Thursday, May 14th 2020, 10:27:39 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Base class for all regression algorithms."""
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
# --------------------------------------------------------------------------  #
class GradientDescent(ABC, BaseEstimator):
    """Base class gradient descent estimator."""

    def __init__(self):
        raise NotImplementedError("GradientDescent base class can not be\
             instantiated.")

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

    @abstractmethod
    def _prepare_data(self, X, y):
        """Prepares X and y data for training."""
        self.X_train_ = self.X_val_ = self.y_train_ = self.y_val_ = None
        # Validate inputs
        self.X_train_, self.y_train_ = check_X_y(X, y)
        # Add a column of ones to create the X design matrix
        self.X_train_ = np.insert(self.X_train_, 0, 1.0, axis=1)      

    @abstractmethod
    def _init_weights(self):
        """Initializes weights"""       
        pass

    @abstractmethod
    def _evaluate_epoch(self, log=None):
        """Computes training (and validation) costs and scores for each epoch."""
        pass
    
    def _compile(self):
        # Initialize callback list
        self._cbks = CallbackList()        
        # BlackBox callback
        self.blackbox_ = BlackBox()
        self._cbks.append(self.blackbox_)
        # Learning rate
        self._cbks.append(copy.copy(self.learning_rate))
        # Progress callback
        if self.verbose:
            self._cbks.append(Progress())
        # Add early stop if object injected.
        if self.early_stop:
            self._cbks.append(copy.copy(self.early_stop))
        # Add gradient checking if object injected.
        if self.gradient_check:
            self._cbks.append(copy.copy(self.gradient_check))        
        # Initialize all callbacks.
        self._cbks.set_params(self.get_params())
        self._cbks.set_model(self)

        # Copy all mutable parameters for sklearn compliance
        self._scorer = copy.copy(self.scorer)

    def _begin_training(self, log=None):
        """Performs initializations required at the beginning of training."""
        self._epoch = 0
        self._batch = 0        
        self._converged = False
        self.is_fitted_ = False                
        self._prepare_data(log.get("X"),log.get("y"))
        self._init_weights()            
        self._compile()
        self._cbks.on_train_begin(log)
        
    def _end_training(self, log=None):
        """Closes history callout and assign final and best weights."""
        self._cbks.on_train_end()
        self.intercept_ = self.theta_[0]
        self.coef_ = self.theta_[1:]
        self.n_iter_ = self._epoch
        self.is_fitted_ = True

    def _begin_epoch(self, log=None):
        """Increment the epoch count and shuffle the data."""
        self._epoch += 1
        self.X_train_, self.y_train_ = shuffle_data(self.X_train_, self.y_train_) 
        self._cbks.on_epoch_begin(self._epoch, log)

    def _end_epoch(self, log=None):        
        """Performs end-of-epoch evaluation and scoring."""
        log = log or {}
        # Update log with current learning rate and parameters theta
        log['epoch'] = self._epoch.copy()
        log['learning_rate'] = self._eta.copy()
        log['theta'] = self.theta_.copy()     
        # Compute performance statistics for epoch and post to history
        log = self._evaluate_epoch(log)
        # Call 'on_epoch_end' methods on callbacks.
        self._cbks.on_epoch_end(self._epoch, log)

    def _begin_batch(self, log=None):
        self._batch += 1
        self._cbks.on_batch_begin(self._batch, log)

    def _end_batch(self, log=None):
        self._cbks.on_batch_end(self._batch, log)

    @abstractmethod
    def _compute_output(self, X, theta):
        """Computes output based upon inputs and model parameters.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The model inputs. Note the number of features includes the coefficient
            for the bias term

        theta : array of shape [n_features,] or [n_features, n_classes]
            Model parameters

        Returns
        -------
        output : Model output            
        
        """        
        pass

    @abstractmethod
    def _compute_cost(self, y, y_out, theta):
        """Computes the mean squared error cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """        
        pass

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        pass

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

        while (self._epoch < self.epochs and not self._converged):

            self._begin_epoch()

            for X_batch, y_batch in batch_iterator(self.X_train_, self.y_train_, batch_size=self.batch_size):

                self._begin_batch()
                
                # Compute model output
                y_out = self.compute_output(X_batch, self.theta_)

                # Compute costs
                J = self.compute_cost(y_batch, y_out, self.theta_)                
                
                # Format batch log with weights and cost
                batch_log = {'batch': self._batch, 'batch_size': X_batch.shape[0],
                             'theta': self.theta_.copy(), 'train_cost': J}

                # Compute gradient
                gradient = self.compute_gradient(X_batch, y_batch, y_out, self.theta_)

                # Update parameters.
                self.theta_ = self.optimizer(self.theta_, gradient, self._eta)

                # Update batch log
                self._end_batch(batch_log)

            # Wrap up epoch
            self._end_epoch()

        self._end_training()
        return self       

    @abstractmethod
    def _predict(self, X, theta):
        """Abstract method overriden by subclasses with specific predict functions."""
        pass 

    def predict(self, X):
        """Computes prediction on final trained weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : prediction
        """
        X = check_array(X, accept_sparse=True, accept_large_sparse=True, estimator=self)        
        return self._predict(X, self.theta_)
    
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
        return self._scorer(y, y_pred)

    def summary(self, features=None):
        summary(self.blackbox_, features)

















class BaseRegressor(ABC, BaseEstimator):
    """Base class for all regression subclasses."""

    @abstractmethod
    def __init__(self):      
        raise Exception("Instantiation of the BaseRegressor class is prohibited.")  

    @property
    def task(self):
        return "Regression"
        
    def _validate_hyperparam(self, p):
        """Validates a parameter. Used for validating regularization parameters."""
        assert isinstance(p, (int,float)), "Regularization hyperparameter must be numeric."
        assert p >= 0 and p <= 1, "Regularization parameter must be between zero and 1."

    def predict(self, X, theta):
        """Computes the prediction as linear combination of inputs and parameters.        

        Parameter
        ---------
        X : array of shape [n_samples, n_features]
            The model inputs. 

        theta : array of shape [n_features,] 
            Model parameters

        Note: n_features may or may not include the bias term added prior to 
        training, so we will need to accommodate X of either dimension.

        Returns
        -------
        prediction : Linear combination of inputs.

        """         
        if X.shape[1] == len(theta):
            y_pred = X.dot(theta)
        else:
            y_pred = theta[0] + X.dot(theta[1:])
        return y_pred

    def compute_output(self, X, theta):
        """Computes output based upon inputs and model parameters.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The model inputs. Note the number of features includes the coefficient
            for the bias term

        theta : array of shape [n_features,] or [n_features, n_classes]
            Model parameters

        Returns
        -------
        output : Model output            
        
        """
        return X.dot(theta)
        

    @abstractmethod
    def compute_cost(self, y, y_pred, theta):
        """Implements the cost function.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,) or (n_features, n_classes)        
            The model parameters            

        Returns
        -------
        cost : Computed cost of the objective function. 

        """
        pass

    @abstractmethod
    def compute_gradient(self, X, y, y_pred, theta):
        """Computes the gradient of cost function.

        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions         

        theta : array of shape (n_features,) or (n_features, n_classes)           

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        pass
