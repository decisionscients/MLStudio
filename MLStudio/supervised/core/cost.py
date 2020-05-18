#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : cost.py                                                           #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, May 16th 2020, 10:10:13 pm                        #
# Last Modified : Saturday, May 16th 2020, 10:10:14 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Cost functions and their gradients."""
from abc import ABC, abstractmethod

import numpy as np

from mlstudio.supervised.core.regularization import Nill
from mlstudio.utils.data_manager import Normalize
# --------------------------------------------------------------------------  #
class Cost(ABC):
    """Base class for all cost classes."""
    
    def __init__(self, regularization=None,  clip_threshold=1e-10):        
        """Initialize regularization, and gradient clipping.
        
        Initializes the regularization object, a normalization object,
        and sets the gradient clipping threshold as the exponent to which
        one should be raised.

        Parameters
        ----------
        clip_threshold : int (default=10)
            The power of 1 representing the absolute value of the 
            lower bound on the magnitudes of the gradients.  

        regularization : Regularization class
            Either None, L1, L2, or L1_l2 regularization.        
        
        """

        self.clip_threshold = clip_threshold
        
        if not regularization:
            self.regularization = Nill()
        else:
            self.regularization = regularization

        self._normalizer = Normalize()

    def _check_gradient(self, X):
        """Checks the gradient for underflow and normalizes it if necessary."""        
        r_x = np.linalg.norm(X) 
        if r_x < self.clip_threshold or r_x > self.clip_threshold ** -1:
            return self._normalizer.fit_transform(X)
        return X

    @abstractmethod
    def __call__(self, y, y_out, theta):
        pass

    @abstractmethod
    def gradient(self, X, y, y_out, theta):
        pass

# --------------------------------------------------------------------------  #
class MSE(Cost):

    def __init__(self, regularization=None,  clip_threshold=1e-10):      
        super(MSE, self).__init__(regularization, clip_threshold)

    def __call__(self, y, y_out, theta):
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
        n_samples = y.shape[0]
        J = 0.5 * np.mean((y-y_out)**2) 
        # Add regularization of weights
        J += self.regularization(theta)  / n_samples
        return J

    def gradient(self, X, y, y_out, theta):
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
        n_samples = X.shape[0]
        dZ = y_out-y
        dW = float(1. / n_samples) * X.T.dot(dZ) 
        # Check gradient before normalizing it with n_samples
        dW = self._check_gradient(dW)
        # Add the gradient of regularization of weights 
        dW += self.regularization.gradient(theta) / n_samples        
        return(dW)        

# --------------------------------------------------------------------------  #
class CrossEntropy(Cost):

    def __init__(self, regularization=None,  clip_threshold=1e-10):      
        super(CrossEntropy, self).__init__(regularization, clip_threshold)

    def __call__(self, y, y_out, theta):
        """Computes cross entropy cost.

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
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)        
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_out)) + \
            np.multiply(1-y, np.log(1-y_out))) 
        # Add regularization of weights 
        J += self.regularization(theta) / n_samples        
        return J   

    def gradient(self, X, y, y_out, theta):
        """Computes cross entropy cost  gradient with respect to weights.
        
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
        n_samples = X.shape[0]
        dZ = y_out-y
        dW = float(1./n_samples) * X.T.dot(dZ)         
        # Check gradient before normalizing it with n_samples
        dW = self._check_gradient(dW)
        dW += self.regularization.gradient(theta) / n_samples        
        return(dW)          

# --------------------------------------------------------------------------  #
class CategoricalCrossEntropy(Cost):

    def __init__(self, regularization=None,  clip_threshold=1e-10):      
        super(CategoricalCrossEntropy, self).__init__(regularization, clip_threshold)

    def __call__(self, y, y_out, theta):
        """Computes categorical cross entropy cost.

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
        
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)    
        # Obtain unregularized cost
        J = np.mean(-np.sum(np.log(y_out) * y, axis=1))
        # Add regularization of weights 
        J += self.regularization(theta) / n_samples
        return J 

    def gradient(self, X, y, y_out, theta):
        """Computes gradient of cross-entropy cost with respect to weights.
        
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
        n_samples = y.shape[0]
        dZ =y_out-y
        dW = 1/n_samples * X.T.dot(dZ)
        # Check gradient before normalizing it with n_samples
        dW = self._check_gradient(dW)        
        # Add regularization of weights 
        dW += self.regularization.gradient(theta) / n_samples        
        return(dW)                  