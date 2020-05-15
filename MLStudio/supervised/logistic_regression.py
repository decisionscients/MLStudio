#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : logistic_regression.py                                            #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                      #
# Last Modified : Friday, April 10th 2020, 9:54:12 am                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes supporting binary classification ."""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import ClassifierMixin

from mlstudio.supervised.base_model import BaseModel
# --------------------------------------------------------------------------- #
#                          LOGISTIC REGRESSION                                #
# --------------------------------------------------------------------------- #            
class LogisticRegression(BaseModel, ClassifierMixin):
    """Logistic Regression Algorithm"""

    def __init__(self):
        pass

    @property
    def name(self):
        return "Logistic Regression"    

    @property
    def task(self):
        return "Classification"        

    def _sigmoid(self, Z):
        """Uses sigmoid to predict the probability of a positive response.""" 
        s = 1.0 / (1 + np.exp(-Z))
        return s

    def compute_output(self, X, theta):
        """Computes output from the sigmoid function.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The model inputs. Note the number of features includes the coefficient
            for the bias term

        theta : array of shape [n_features,] or [n_features, n_classes]
            Model parameters
        
        """
        return self._sigmoid(X.dot(theta))

    def predict(self, X, theta):
        """Computes the prediction logistic regression prediction.

        Parameter
        ---------
        X : array of shape (n_samples, n_features)
            Input data

        theta : array of shape (n_features,)  
            The model parameters

        Note: n_features may or may not include the bias term added prior to 
        training, so we will need to accommodate X of either dimension.            

        Returns
        -------
        prediction : Binary classification

        Raises
        ------
        Value error if X and theta have incompatible shapes.
        """    
        if X.shape[1] == len(theta):
            z = X.dot(theta)
        else:
            z = theta[0] + X.dot(theta[1:])
        a = self._sigmoid(z)        
        return np.round(a).astype(int)        

    def compute_cost(self, y, y_out, theta=None):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Model output.
            
        Returns
        -------
        cost : The binary cross-entropy cost 

        """
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)        
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_out)) + np.multiply(1-y, np.log(1-y_out)))
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes logistic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Predictions                    

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """  
        n_samples = y.shape[0]
        dZ = y_out-y
        dW = 1/n_samples * X.T.dot(dZ)
        return(dW)             

# --------------------------------------------------------------------------- #
#                          LASSO LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #            
class LassoLogisticRegression(LogisticRegression):
    """Logistic Regression Algorithm with Lasso Regularization"""

    def __init__(self, alpha=1):
        self.alpha = alpha

    @property
    def name(self):
        return "Logistic Regression with Lasso Regularization"    

    @property
    def task(self):
        return "Classification"        

    def compute_cost(self, y, y_out, theta):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Ground truth target values

        y_out : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters              

        Returns
        -------
        cost : The binary cross-entropy cost 

        """
        self._validate_hyperparam(self.alpha)
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)        
        # Compute regularization
        J_reg = (self.alpha / n_samples) * np.linalg.norm(theta, ord=1)
        # Compute lasso regularized cost
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_out)) + \
            np.multiply(1-y, np.log(1-y_out))) + J_reg
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features+1)
            Input data

        y : array of shape (n_samples,)
            Ground truth target values

        y_out : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        dZ = y_out-y
        dW = 1/n_samples * (X.T.dot(dZ) + self.alpha * np.sign(theta))
        return(dW)                     

# --------------------------------------------------------------------------- #
#                          RIDGE LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #            
class RidgeLogisticRegression(LogisticRegression):
    """Logistic Regression Algorithm with Ridge Regularization"""

    def __init__(self, alpha=1):
        self.alpha = alpha

    @property
    def name(self):
        return "Logistic Regression with Ridge Regularization"    

    @property
    def task(self):
        return "Classification"              

    def compute_cost(self, y, y_out, theta):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Ground truth target values

        y_out : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters              

        Returns
        -------
        cost : The binary cross-entropy cost 

        """
        self._validate_hyperparam(self.alpha)
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)        
        # Compute regularization
        J_reg = (self.alpha / (2*n_samples)) * np.linalg.norm(theta)**2
        # Compute ridge regularized cost
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_out)) + \
            np.multiply(1-y, np.log(1-y_out))) + J_reg
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features+1)
            Input data

        y : array of shape (n_samples,)
            Ground truth target values

        y_out : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        dZ = y_out-y
        dW = 1/n_samples * (X.T.dot(dZ) + self.alpha * theta)
        return(dW)                             

# --------------------------------------------------------------------------- #
#                       ELASTIC NET LOGISTIC REGRESSION                       #
# --------------------------------------------------------------------------- #            
class ElasticNetLogisticRegression(LogisticRegression):
    """Logistic Regression Algorithm with ElasticNet Regularization"""

    def __init__(self, alpha=1, ratio=0.5):
        self.alpha=alpha
        self.ratio=ratio

    @property
    def name(self):
        return "Logistic Regression with ElasticNet Regularization"    

    @property
    def task(self):
        return "Classification"      

    def compute_cost(self, y, y_out, theta):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Ground truth target values

        y_out : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters              

        Returns
        -------
        cost : The binary cross-entropy cost 

        """
        self._validate_hyperparam(self.alpha)
        self._validate_hyperparam(self.ratio)

        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)        
        # Compute regularization
        l1_contr = self.ratio * np.linalg.norm(theta, ord=1)
        l2_contr = (1 - self.ratio) * 0.5 * np.linalg.norm(theta)**2        
        J_reg = float(1./n_samples) * self.alpha * (l1_contr + l2_contr)
        # Compute elasticnet regularized cost
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_out)) + \
            np.multiply(1-y, np.log(1-y_out))) + J_reg
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features+1)
            Input data

        y : array of shape (n_samples,)
            Ground truth target values

        y_out : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        l1_contr = self.ratio * np.sign(theta)
        l2_contr = (1 - self.ratio) * theta        
        alpha = np.asarray(self.alpha, dtype='float64')     
        dZ = y_out-y
        dW = 1/n_samples  * (X.T.dot(dZ) + np.multiply(alpha, np.add(l1_contr, l2_contr)))
        return(dW)                       