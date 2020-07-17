# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \objectives copy.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 2:34:50 am                        #
# Last Modified : Thursday, July 16th 2020, 2:34:50 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Loss functions and their gradients."""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator

from mlstudio.utils.data_manager import GradientScaler    
# --------------------------------------------------------------------------  #
#                           LOSS BASE CLASS                                   #
# --------------------------------------------------------------------------  #
class Loss(ABC, BaseEstimator):
    """Base class for all loss functions.
    
    Parameters
    ----------
    regularizer : a Regularizer object
        Object to perform L1, L2, or L1_L2 regularization of the gradient

    gradient_scaler : a GradientScaler object or None
        Defaults to a GradientScaler object with default lower 
        threshold = 1e-10 and upper threshold = 1.
    """

    def __init__(self, regularizer=None, gradient_scaler=None):
        self.regularizer = regularizer
        self.gradient_scaler = gradient_scaler
        
    def _validation(self):
        from mlstudio.utils.validation import validate_regularizer                
        validate_regularizer(self.regularizer)
        if self.gradient_scaler:
            validate_gradient_scaler(self.gradient_scaler)

    def _check_gradient_scale(self, gradient):
        if self.gradient_scaler:
            gradient = self.gradient_scaler.fit_transform(gradient)
        return gradient

    @property
    def turn_off_gradient_scaling(self):
        self.gradient_scaler = False

    @abstractmethod
    def __call__(self, theta, **kwargs):
        """Computes the objective function.

        Parameters
        ----------
        theta : array-like
            The parameters from the model

        kwargs : Arbitrary keyword arguments.

        Returns
        -------
        real number
        """
        pass

    @abstractmethod
    def gradient(self, theta, **kwargs):
        """Computes the derivative of the objective function w.r.t. theta.
        
        Parameters
        ----------
        theta : array-like
            The parameters from the model

        kwargs : Arbitrary keyword arguments.

        Returns
        -------
        gradient : array-like in the shape of theta 
            The gradient of the objective function
        """
        pass
         
# --------------------------------------------------------------------------  #
class Quadratic(Loss):

    def __init__(self, regularizer=None, gradient_scaler=None):
        super(Quadratic, self).__init__(regularizer=regularizer, 
                                  gradient_scaler=gradient_scaler)

        self.name = "Quadratic Loss"        
        self.type = "Regression"

    def __call__(self, theta, y, y_out):
        """Computes the quadratic mean squared error cost.

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
        # Number of samples in the dataset     
        m = y.shape[0]     
        # Compute unregularized cost
        J = 0.5 * np.mean((y_out-y)**2)        
        # Apply regularization
        if self.regularizer:
            J += self.regularizer(theta, m)
        # Normalize cost and regularization by 
        return J

    def gradient(self, theta, X, y, y_out):
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
        # Number of samples in the dataset
        m = y.shape[0]
        # Compute unregularized gradient
        gradient = (1/m) * X.T.dot(y_out-y)                         
        # Apply regularization to the weights (not bias) in gradient
        if self.regularizer:
            gradient += self.regularizer.gradient(theta, m)              
        # Check any vanishing or exploding gradients          
        gradient = self._check_gradient_scale(gradient)                            
        return gradient        

# --------------------------------------------------------------------------  #
class CrossEntropy(Loss):

    def __init__(self, regularizer=None, gradient_scaler=None):
        super(CrossEntropy, self).__init__(regularizer=regularizer, 
                                  gradient_scaler=gradient_scaler)

        self.name = "Cross Entropy Loss"        
        self.type = "Binary Classification"

    def __call__(self, theta, y, y_out):
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
        # Number of samples in the dataset
        m = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)    
        # Compute unregularized cost
        J = -np.mean(y * np.log(y_out) + (1-y) * np.log(1-y_out)) 
        # Apply regularization to the weights (not bias) in gradient
        if self.regularizer: 
            J += self.regularizer(theta, m)
        # Return cost as average of cross entropy
        return J 

    def gradient(self, theta, X, y, y_out):
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
        # Number of samples in the dataset
        m = X.shape[0]        
        # Compute unregularized gradient
        gradient = (1/m) * X.T.dot(y_out-y)         
        # Apply regularization to the weights (not bias) in gradient
        if self.regularizer:
            gradient += self.regularizer.gradient(theta, m)                
        # Check any vanishing or exploding gradients
        gradient = self._check_gradient_scale(gradient)            
        return gradient          

# --------------------------------------------------------------------------  #
class CategoricalCrossEntropy(Loss):


    def __init__(self, regularizer=None, gradient_scaler=None):
        super(CategoricalCrossEntropy, self).__init__(regularizer=regularizer, 
                                  gradient_scaler=gradient_scaler)

        self.name = "Categorical Cross Entropy Loss"        
        self.type = "Multiclass Classification"

    def __call__(self, theta, y, y_out):
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
        # Number of samples in the dataset
        m = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)    
        # Compute unregularized cost
        J = -np.mean(np.sum(y * np.log(y_out), axis=1))
        # Add regularizer of weights 
        if self.regularizer:
            J += self.regularizer(theta, m) 
        # Return cost as average of cross-entropy
        return J

    def gradient(self, theta, X, y, y_out):
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
        gradient = {}
        # Number of samples in the dataset
        m = y.shape[0]
        # Compute unregularized gradient
        gradient = -(1/m) * (X.T.dot(y-y_out))
        # Add regularizer of weights 
        if self.regularizer:
            gradient += self.regularizer.gradient(theta, m)     
        # Check gradient scale before applying regularization
        gradient = self._check_gradient_scale(gradient)                    
        return gradient                  
