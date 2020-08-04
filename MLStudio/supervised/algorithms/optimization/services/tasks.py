#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : task.py                                                           #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Tuesday, May 19th 2020, 10:00:13 pm                         #
# Last Modified : Tuesday, May 19th 2020, 10:00:13 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Defines linear, logistic, and multiclass classification tasks."""
from abc import ABC, abstractmethod, abstractproperty 
from collections import OrderedDict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import DataConversionWarning
from mlstudio.supervised.algorithms.optimization.services.activations import Sigmoid, Softmax
from mlstudio.supervised.algorithms.optimization.services.loss import Quadratic 
from mlstudio.supervised.algorithms.optimization.services.loss import CrossEntropy
from mlstudio.supervised.algorithms.optimization.services.loss import CategoricalCrossEntropy 
from mlstudio.utils import validation
# --------------------------------------------------------------------------  #
class Task(ABC, BaseEstimator):
    """Defines the base class for all tasks."""

    def __init__(self, loss, data_processor, activation=None, 
                 random_state=None):
        self.loss = loss        
        self.data_processor = data_processor
        self.activation = activation        
        self.random_state = random_state
        self._n_features_out = None         
        self._n_classes = None
        self._data_prepared = False

    @abstractproperty
    def name(self):
        pass

    @abstractproperty
    def loss(self):
        pass

    @abstractproperty
    def data_processor(self):
        pass

    def prepare_train_data(self, X, y=None):
        """Prepares training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The independent variables from the training set.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            The dependent variable from the training set.

        Returns
        -------
        X_train : array-like of shape (n_samples, n_features) 
        y_train : array-like of shape (n_samples,) or (n_samples, n_classes)

        """

        data = self._data_processor.process_train_data(X, y, random_state)
        self._n_features_out = data['X_train']['metadata']['processed']['n_features']

        self._data_prepared = True

        return data

    def prepare_train_val_data(self, X, y=None, val_size=None, random_state=None):
        """Prepares training data.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The independent variables from the training set.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            The dependent variable from the training set.

        val_size : float in [0,1)
            The proportion of data to allocate to the validation set.

        random_state : int or None (default=None)
            Seed for pseudo-randomization

        Returns
        -------
        X_train : array-like of shape (n_samples, n_features) 
        y_train : array-like of shape (n_samples,) or (n_samples, n_classes)
        X_test : array-like of shape (n_samples, n_features) 
        y_test : array-like of shape (n_samples,) or (n_samples, n_classes)        

        """

        data = self._data_processor.process_train_val_data(X, y, val_size, 
                                                            random_state)
        self._n_features_out = data['X_train']['metadata']['processed']['n_features']
        self._n_classes = data['y_train']['metadata']['orig']['n_classes']
        self._data_prepared = True
        return data


    def init_weights(self, theta_init=None):
        """Initializes parameters to theta_init or to random values.
        
        Parameters
        ----------
        theta_init : array-like of shape (n_features,) or (n_features, n_classes) Optional
            Optional initial values for the model parameters.

        Raises
        ------
        Exception if data has not been processed

        Returns
        ------        
        theta : array-like of shape (n_features,) or (n_features, n_classes)
        """
        if not self._data_prepared:
            raise Exception("Data must be prepared before weights are initialized.")

        if theta_init is not None:
            assert theta_init.shape == (self._n_features_out,),\
                "Initial parameters theta must have shape (n_features,)."
            theta = theta_init
        else:
            # Random initialization of weights
            rng = np.random.RandomState(self.random_state)                
            theta = rng.randn(self._n_features_out) 
            # Set the bias initialization to zero
            theta[0] = 0
        return theta

    def compute_loss(self, theta, y, y_out):
        """Computes average loss

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        y : array of shape (n_samples) or (n_samples, n_classes)
            The target variable 

        y_out : array of shape (n_samples) or (n_samples, n_classes)
            The computed output based on current parameters theta.             
        
        Returns
        -------
        J : average loss
        """
        return self._loss(theta, y, y_out)

    def compute_output(self, theta, X):
        """Computes output as a linear combination of parameters and inputs.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data
        
        Returns
        -------
        y_out : output
        """
        return np.array(X.dot(theta), dtype=np.float32)

    def gradient(self, theta, X, y, y_out):
        """Computes the gradient."""

        return self._loss.gradient(theta, X, y, y_out)

    def _check_X(self, X, theta):
        """Checks X to ensure that it has been processed for training/prediction."""
        X = validation.check_X(X)        
        if X.shape[1] != theta.shape[0]:                
            X = self._data_processor.process_X_test_data(X)                    
        return X

    def _check_y_pred(self, y_pred):
        if y_pred.ndim > 1:
            msg = self.__class__.__name__ + " doesn't support multioutput."
            warnings.warn(msg, DataConversionWarning)        
        else:
            return y_pred

    @abstractmethod
    def predict(self, X, theta):
        """Computes prediction on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        theta : array-like of shape (n_features) or (n_features, n_classes)
            The model parameters
        
        Returns
        -------
        y_pred : prediction
        """
        pass
    
# --------------------------------------------------------------------------  #
class LinearRegression(Task):
    """Defines the linear regression task."""

    @property
    def name(self):
        return "Linear Regression"

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_regression_loss(x)
        self._loss = x

    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_regression_data_processor(x)
        self._data_processor = x        

    def predict(self, X, theta):
        X = self._check_X(X, theta)
        y_pred = self.compute_output(theta, X)
        y_pred = self._check_y_pred(y_pred)
        return y_pred

    def predict_proba(self, theta, X):
        raise NotImplementedError("predict_proba is not implemented for the LinearRegression task.")

# --------------------------------------------------------------------------  #
class BinaryClassification(Task):
    """Defines the logistic regression task."""

    @property
    def name(self):
        return "Logistic Regression"    

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_binaryclass_loss(x)
        self._loss = x
    
    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_binaryclass_data_processor(x)
        self._data_processor = x        

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, x):
        validation.validate_binaryclass_activation(x)
        self._activation = x                
    
    def compute_output(self, theta, X):
        """Computes output as a probability of the positive class.

        The logit or linear combination of inputs and parameters is passed
        through a sigmoid function to create the probability of the 
        positive class.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data
        
        Returns
        -------
        y_out
        """
        z = super(BinaryClassification, self).compute_output(theta, X)        
        return self._activation(z)

    def _check_y(self, y):
        """Confirms y has been encoded."""
        if not validation.is_binary(y):
            data = self._data_processor.process_y_test_data(y)
            return data['y_test']['data']
        return y
        
    def predict(self, X, theta):
        """Computes prediction on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        Returns
        -------
        y_pred : Predicted class
        """        
        o = self.predict_proba(X, theta)
        y_pred = np.round(o).astype(int)
        y_pred = self._check_y_pred(y_pred)
        return y_pred

    def predict_proba(self, X, theta):
        """Predicts the probability of the positive class

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data

        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        Returns
        -------
        y_pred : Predicted class probability
        """
        X = self._check_X(X, theta)
        y_pred = self.compute_output(theta, X)     
        y_pred = self._check_y_pred(y_pred)
        return y_pred        

# --------------------------------------------------------------------------  #
class MultiClassification(Task):
    """Defines the multiclass classification task."""

    @property
    def name(self):
        return "Multiclass Classification"    

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_multiclass_loss(x)
        self._loss = x

    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_multiclass_data_processor(x)
        self._data_processor = x        

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, x):
        validation.validate_multiclass_activation(x)
        self._activation = x                        

    def init_weights(self, theta_init=None):
        """Initializes parameters to theta_init or to random values.
        
        Parameters
        ----------
        theta_init : array-like of shape (n_features,) or (n_features, n_classes) Optional
            Optional initial values for the model parameters.

        Raises
        ------
        Exception if data has not been processed

        Returns
        ------        
        theta : array-like of shape (n_features,) or (n_features, n_classes)
        """
        if not self._data_prepared:
            raise Exception("Data must be prepared before weights are initialized.")

        if theta_init is not None:
            assert theta_init.shape == (self._n_features_out, self._n_classes),\
                "Initial parameters theta must have shape (n_features,n_classes)."
            theta = theta_init
        else:
            # Random initialization of weights
            rng = np.random.RandomState(self.random_state)                
            theta = rng.randn(self._n_features_out, self._n_classes) 
            # Set the bias initialization to zero
            theta[0] = 0
        return theta          

    def compute_output(self, theta, X):
        """Computes output as a vector of class probabilities.

        The unnormalized linear combination of inputs and parameters is passed
        through a softmax function to create a vector of probabilities.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data
        
        Returns
        -------
        y_out
        """      
        z = super(MultiClassification, self).compute_output(theta, X)
        return self._activation(z)        

    def _check_y(self, y):
        """Confirms y has been one-hot encoded."""
        if not validation.is_one_hot(y):
            data = self._data_processor.process_y_test_data(y)
            y = data['y_test']['data'] 
        return y

    def predict(self, X, theta):
        """Computes prediction on test data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data        

        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        Returns
        -------
        y_pred : Predicted class
        """
                   
        o = self.predict_proba(X, theta)
        return o.argmax(axis=1)

    def predict_proba(self, X, theta):
        """Predicts the class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input data        

        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        Returns
        -------
        y_pred : Predicted class probability
        """              
        X = self._check_X(X, theta)
        return self.compute_output(theta, X)            
