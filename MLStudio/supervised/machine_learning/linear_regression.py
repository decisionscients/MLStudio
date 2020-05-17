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
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array

from mlstudio.supervised.machine_learning.base import BaseRegressor
# --------------------------------------------------------------------------- #
#                          LINEAR REGRESSION                                  #
# --------------------------------------------------------------------------- #    
class LinearRegression(BaseRegressor, RegressorMixin):
    """Linear Regression algorithm."""

    def __init__(self):
        pass

    @property
    def name(self):
        return "Linear Regression"    

    @property
    def task(self):
        return "Regression"

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

    def compute_cost(self, y, y_out, theta):
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
        return J

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
        n_samples = X.shape[0]
        dZ = y_out-y
        dW = float(1./n_samples) * X.T.dot(dZ) 
        return(dW)           

# --------------------------------------------------------------------------- #
#                          LASSO REGRESSION                                   #
# --------------------------------------------------------------------------- #    
class LassoRegression(LinearRegression):
    """Lasso Regression algorithm."""
    
    def __init__(self, lambda_reg=0.0001):
        self.lambda_reg = lambda_reg

    @property
    def name(self):
        return "Lasso Regression"        

    def compute_cost(self, y, y_out, theta):
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
        self._validate_hyperparam(self.lambda_reg)
        n_samples = y.shape[0]
        J_reg = (self.lambda_reg / n_samples) * np.linalg.norm(theta, ord=1)
        J = np.mean(0.5 * (y-y_out)**2) + J_reg
        return J

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
        n_samples = X.shape[0]
        dZ = y_out-y
        dW = 1/n_samples  * (X.T.dot(dZ) + self.lambda_reg * np.sign(theta))
        return(dW)           

# --------------------------------------------------------------------------- #
#                          RIDGE REGRESSION                                   #
# --------------------------------------------------------------------------- #            
class RidgeRegression(LinearRegression):
    """Ridge Regression algorithm."""
    
    def __init__(self, lambda_reg=0.0001):
        self.lambda_reg=lambda_reg

    @property
    def name(self):
        return "Ridge Regression"

    def compute_cost(self, y, y_out, theta):
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
        self._validate_hyperparam(self.lambda_reg)
        n_samples = y.shape[0]
        J_reg = (self.lambda_reg / (2*n_samples)) * np.linalg.norm(theta)**2
        J = np.mean(0.5 * (y-y_out)**2) + J_reg
        return J

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
        n_samples = X.shape[0]
        dZ = y_out-y
        dW = 1/n_samples  * (X.T.dot(dZ) + self.lambda_reg * theta)
        return(dW)                         

# --------------------------------------------------------------------------- #
#                        ELASTIC NET REGRESSION                               #
# --------------------------------------------------------------------------- #            
class ElasticNetRegression(LinearRegression):
    """Elastic Net Regression algorithm."""
    
    def __init__(self, lambda_reg=0.0001, ratio=0.15):
        self.lambda_reg=lambda_reg
        self.ratio=ratio

    @property
    def name(self):
        return "ElasticNet Regression"               

    def compute_cost(self, y, y_out, theta):
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
        self._validate_hyperparam(self.lambda_reg)
        self._validate_hyperparam(self.ratio)
        l1_contr = self.ratio * np.linalg.norm(theta, ord=1)
        l2_contr = (1 - self.ratio) * 0.5 * np.linalg.norm(theta)**2        
        J_reg = float(1./n_samples) * self.lambda_reg * (l1_contr + l2_contr)
        J = np.mean(0.5 * (y-y_out)**2) + J_reg
        return J

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
        n_samples = X.shape[0]
        l1_contr = self.ratio * np.sign(theta)
        l2_contr = (1 - self.ratio) * theta
        lambda_reg = np.asarray(self.lambda_reg, dtype='float64')         
        dZ = y_out-y
        dW = 1/n_samples  * (X.T.dot(dZ) + np.multiply(lambda_reg, np.add(l1_contr, l2_contr)))
        return(dW)               