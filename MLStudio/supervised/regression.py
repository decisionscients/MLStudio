#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : regression.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                      #
# Last Modified : Monday, March 23rd 2020, 10:31:37 am                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Regression algorithms.

This class encapsulates the core behaviors for regression classes. Currently,
the following regression classes are supported.
    
    * Linear Regression
    * Lasso Regression
    * Ridge Regression
    * ElasticNet Regression

The core behaviors exposed for each class include:

    * predict : Predicts outputs as linear combination of inputs and weights.
    * compute_cost : Computes cost associated with predictions
    * compute_gradient : Computes the derivative of loss w.r.t. to weights

"""
from abc import ABC, abstractmethod
import numpy as np

from mlstudio.supervised.estimator.regularizers import NullRegularizer
from mlstudio.supervised.estimator.regularizers import L1, L2, ElasticNet
from mlstudio.utils.data_manager import decode

# --------------------------------------------------------------------------- #
#                          REGRESSION ALGORITHM                               #
# --------------------------------------------------------------------------- #
class Regression(ABC):
    """Base class for regression subclasses."""

    def __init__(self):      
        raise Exception("Instantiation of the Regression base class is prohibited.")  
        
    def _validate_hyperparam(self, p):
        assert isinstance(p, (int,float)), "Regularization hyperparameter must be numeric."
        assert p >= 0 and p <= 1, "Regularization parameter must be between zero and 1."

    def predict(self, X, theta):
        """Computes the prediction.

        Parameter
        ---------
        X : array of shape (m_observations, n_features)
            Input data

        theta : array of shape (n_features,)  
            The model parameters

        Returns
        -------
        prediction : Linear combination of inputs.

        Raises
        ------
        Value error if X and theta have incompatible shapes.
        """
        if X.shape[1] == len(theta):
            y_pred = X.dot(theta)
        elif X.shape[1] == len(theta) - 1:
            y_pred = theta[0] + X.dot(theta[1:])
        else:
            raise ValueError("X and parameters theta have incompatible shapes.\
                 X.shape = {xshape}, theta.shape = {thetashape}.".format(
                     xshape=X.shape, thetashape = theta.shape))
        return y_pred

    @abstractmethod
    def compute_cost(self, y, y_pred, theta):
        """Computes the mean squared error cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        pass

    @abstractmethod
    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        pass
   

# --------------------------------------------------------------------------- #
#                          LINEAR REGRESSION                                  #
# --------------------------------------------------------------------------- #    
class LinearRegression(Regression):
    """Linear Regression algorithm."""
    
    def __init__(self):
        self.name = "Linear Regression"

    def compute_cost(self, y, y_pred, theta):
        """Computes the mean squared error cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        J = np.mean(0.5 * (y-y_pred)**2)
        return J

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        m = X.shape[0]
        dZ = y_pred-y
        dW = float(1./m) * X.T.dot(dZ) 
        return(dW)           

# --------------------------------------------------------------------------- #
#                          LASSO REGRESSION                                   #
# --------------------------------------------------------------------------- #    
class LassoRegression(Regression):
    """Lasso Regression algorithm."""
    
    def __init__(self, alpha=1):
        self.alpha = alpha
        self.name = "Lasso Regression"

    def compute_cost(self, y, y_pred, theta):
        """Computes the mean squared error cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        self._validate_hyperparam(self.alpha)
        m = y.shape[0]
        J_reg = (self.alpha / m) * np.linalg.norm(theta, ord=1)
        J = np.mean(0.5 * (y-y_pred)**2) + J_reg
        return J

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        m = X.shape[0]
        dZ = y_pred-y
        dW = 1/m * (X.T.dot(dZ) + self.alpha * np.sign(theta))
        return(dW)           

# --------------------------------------------------------------------------- #
#                          RIDGE REGRESSION                                   #
# --------------------------------------------------------------------------- #            
class RidgeRegression(Regression):
    """Ridge Regression algorithm."""
    
    def __init__(self, alpha=1):
        self.alpha=alpha
        self.name = "Ridge Regression"    

    def compute_cost(self, y, y_pred, theta):
        """Computes the mean squared error cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        self._validate_hyperparam(self.alpha)
        m = y.shape[0]
        J_reg = (self.alpha / (2*m)) * np.linalg.norm(theta)**2
        J = np.mean(0.5 * (y-y_pred)**2) + J_reg
        return J

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        m = X.shape[0]
        dZ = y_pred-y
        dW = 1/m * (X.T.dot(dZ) + self.alpha * theta)
        return(dW)                         

# --------------------------------------------------------------------------- #
#                        ELASTIC NET REGRESSION                               #
# --------------------------------------------------------------------------- #            
class ElasticNetRegression(Regression):
    """Elastic Net Regression algorithm."""
    
    def __init__(self, alpha=1, ratio=0.5):
        self.alpha=alpha
        self.ratio=ratio
        self.name = "ElasticNet Regression"           

    def compute_cost(self, y, y_pred, theta):
        """Computes the mean squared error cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        m = y.shape[0]
        self._validate_hyperparam(self.alpha)
        self._validate_hyperparam(self.ratio)
        l1_contr = self.ratio * np.linalg.norm(theta, ord=1)
        l2_contr = (1 - self.ratio) * 0.5 * np.linalg.norm(theta)**2        
        J_reg = float(1./m) * self.alpha * (l1_contr + l2_contr)
        J = np.mean(0.5 * (y-y_pred)**2) + J_reg
        return J

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        m = X.shape[0]
        l1_contr = self.ratio * np.sign(theta)
        l2_contr = (1 - self.ratio) * theta
        alpha = np.asarray(self.alpha, dtype='float64')         
        dZ = y_pred-y
        dW = 1/m * (X.T.dot(dZ) + np.multiply(alpha, np.add(l1_contr, l2_contr)))
        return(dW)               