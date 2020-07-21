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

import numpy as np
from sklearn.base import BaseEstimator
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
        self._n_features = None         # Note: This count includes the bias term
        self._n_classes = None
        self._data_prepared = False

    @abstractproperty
    def name(self):
        pass

    def prepare_data(self, X, y=None, val_size=None):
        """Prepares data for training.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The independent variables from the training set.

        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            The dependent variable from the training set.

        val_size : float in (0,1) or None
            Proportion of training data to allocate to validation set.

        Returns 
        -------
        data : dictionary 
            Dict containing training and optionally validation data
        """
        data = self._data_processor.fit_transform(X, y, val_size)
        self._n_features = data.get('n_features_')
        self._n_classes = data.get('n_classes_')
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
            assert theta_init.shape == (self._n_features),\
                "Initial parameters theta must have shape (n_features,)."
            theta = theta_init
        else:
            # Random initialization of weights
            rng = np.random.RandomState(self.random_state)                
            theta = rng.randn(self._n_features) 
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

    def predict(self, theta, X):
        """Computes prediction on test data.

        Parameters
        ----------
        theta : array-like of shape (n_features) or (n_features, n_classes)
            The model parameters

        X : array-like of shape (n_samples, n_features)
            The input data
        
        Returns
        -------
        y_pred : prediction
        """
        X = validation.check_X(X)        
        return self.compute_output(theta, X)
    
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

    def predict_proba(self, theta, X):
        raise NotImplementedError("predict_proba is not implemented for the LinearRegression task.")


# --------------------------------------------------------------------------  #
class LogisticRegression(Task):
    """Defines the logistic regression task."""


    @property
    def name(self):
        return "Logistic Regression"    

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_binary_classification_loss(x)
        self._loss = x

    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_binary_classification_data_processor(x)
        self._data_processor = x        

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, x):
        validation.validate_binary_classification_activation(x)
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
        z = super(LogisticRegression, self).compute_output(theta, X)        
        return self._activation(z)

    def predict(self, theta, X):
        """Computes prediction on test data.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : Predicted class
        """        
        o = self.compute_output(theta, X)
        return np.round(o).astype(int)

    def predict_proba(self, theta, X):
        """Predicts the probability of the positive class

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : Predicted class probability
        """               
        return self.compute_output(theta, X)        

# --------------------------------------------------------------------------  #
class MulticlassClassification(Task):
    """Defines the multiclass classification task."""

    def __init__(self, loss, data_processor, activation=None):
        super(MulticlassClassification, self).__init__(loss=loss, 
                                                 data_processor=data_processor,
                                                 activation=activation)
        

    @property
    def name(self):
        return "Multiclass Classification"    

    @property
    def loss(self):
        return self._loss

    @loss.setter
    def loss(self, x):
        validation.validate_multiclass_classification_loss(x)
        self._loss = x

    @property
    def data_processor(self):
        return self._data_processor

    @data_processor.setter
    def data_processor(self, x):
        validation.validate_multiclass_classification_data_processor(x)
        self._data_processor = x        

    @property
    def activation(self):
        return self._activation

    @activation.setter
    def activation(self, x):
        validation.validate_multiclass_classification_activation(x)
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
            assert theta_init.shape == (self._n_features, self._n_classes),\
                "Initial parameters theta must have shape (n_features,n_classes)."
            theta = theta_init
        else:
            # Random initialization of weights
            rng = np.random.RandomState(self.random_state)                
            theta = rng.randn(self._n_features, self._n_classes) 
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
        z = super(MulticlassClassification, self).compute_output(theta, X)
        return self._activation(z)        

    def predict(self, theta, X):
        """Computes prediction on test data.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : Predicted class
        """           
        o = self.compute_output(theta, X)        
        return o.argmax(axis=1)

    def predict_proba(self, theta, X):
        """Predicts the class probabilities.

        Parameters
        ----------
        theta : array_like of shape (n_features,) or (n_features, n_classes)
            The current learned parameters of the model.

        X : array-like of shape (n_samples, n_features)
            The input data

        Returns
        -------
        y_pred : Predicted class probability
        """              
        return self.compute_output(theta, X)                