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

import dill as pickle
import numpy as np

from mlstudio.supervised.core.regularization import L0
# --------------------------------------------------------------------------  #
class Cost(ABC):
    """Base class for all cost classes."""
    
    @abstractmethod
    def __init__(self, regularization=None):        
        if not regularization:
            self.regularization = L0()
        else:
            self.regularization = regularization

    @abstractmethod
    def __call__(self, y, y_out, theta):
        pass

    @abstractmethod
    def gradient(self, X, y, y_out, theta):
        pass

# --------------------------------------------------------------------------  #
class MSE(Cost):

    def __init__(self, regularization=None):
        super(MSE, self).__init__(regularization)

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
        J = np.mean(0.5 * (y-y_out)**2) 
        # Add regularization of weights (not bias)
        J += self.regularization(theta)
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
        dW = float(1./n_samples) * X.T.dot(dZ) 
        # Add the gradient of regularization of weights (not bias)
        dW += self.regularization.gradient(theta)
        return(dW)        

# --------------------------------------------------------------------------  #
class CrossEntropy(Cost):

    def __init__(self, regularization=None):
        super(CrossEntropy, self).__init__(regularization)

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
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)        
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_out)) + \
            np.multiply(1-y, np.log(1-y_out))) 
        # Add regularization of weights (not bias)
        J += self.regularization(theta)
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
        dW = float(1./n_samples) * X.T.dot(dZ) 
        # Add the gradient of regularization of weights (not bias)
        dW += self.regularization.gradient(theta)
        return(dW)          

# --------------------------------------------------------------------------  #
class CategoricalCrossEntropy(Cost):

    def __init__(self, regularization=None):
        super(CategoricalCrossEntropy, self).__init__(regularization)

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
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)    
        # Obtain unregularized cost
        J = np.mean(-np.sum(np.log(y_out) * y, axis=1))
        # Add regularization of weights (not bias)
        J = J + self.regularization(theta[1])
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
        n_samples = y.shape[0]
        dZ =y_out-y
        dW = 1/n_samples * X.T.dot(dZ)
        # Add regularization of weights (not bias)
        dW += self.regularization.gradient(theta)
        return(dW)                  