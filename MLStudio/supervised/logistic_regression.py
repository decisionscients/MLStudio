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
"""Classes supporting binary and multinomial classification ."""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from mlstudio.utils.data_manager import data_split, one_hot

# --------------------------------------------------------------------------- #
#                          LOGISTIC REGRESSION                                #
# --------------------------------------------------------------------------- #            
class LogisticRegression(ABC):
    """Logistic Regression Algorithm"""
    _DEFAULT_METRIC = 'accuracy'
    _TASK = "Logistic Regression"

    def __init__(self):
        self.name = "Logistic Regression"

    def _validate_hyperparam(self, p):
        assert isinstance(p, (int,float)), "Regularization hyperparameter must be numeric."
        assert p >= 0 and p <= 1, "Regularization parameter must be between zero and 1."        

    def _sigmoid(self, Z):
        """Uses sigmoid to predict the probability of a positive response.""" 
        s = 1.0 / (1 + np.exp(-Z))
        return s

    def hypothesis(self, X, theta):
        """Computes the hypothesis using an input design matrix with bias term.

        Parameter
        ---------
        X : array of shape (m_observations, n_features+1)
            Input data

        theta : array of shape (n_features+1,)  
            The model parameters

        Returns
        -------
        hypothesis : Linear combination of inputs.
        """
        return self._sigmoid(X.dot(theta))        

    def predict(self, X, theta):
        """Computes the prediction logistic regression prediction.

        Parameter
        ---------
        X : array of shape (m_observations, n_features)
            Input data

        theta : array of shape (n_features+1,)  
            The model parameters

        Returns
        -------
        prediction : Linear combination of inputs.

        Raises
        ------
        Value error if X and theta have incompatible shapes.
        """    
        X = np.array(X)
        check_array(X)        

        if X.shape[1] == len(theta) - 1:
            X = np.insert(X, 0, 1.0, axis=1)   
        h = self.hypothesis(X, theta)            
        y_pred = np.round(h).astype(int)
        return y_pred

    def compute_cost(self, y, y_pred, theta=None):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        Returns
        -------
        cost : The binary cross-entropy cost 

        """
        n_samples = y.shape[0]
        # Prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_pred)) + np.multiply(1-y, np.log(1-y_pred)))
        return J        

    def compute_gradient(self, X, y, y_pred, theta=None):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions                    

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """  
        n_samples = y.shape[0]
        dW = 1/n_samples * (y_pred-y).dot(X)
        return(dW)             

# --------------------------------------------------------------------------- #
#                          LASSO LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #            
class LassoLogisticRegression(LogisticRegression):
    """Logistic Regression Algorithm"""
    _DEFAULT_METRIC = 'accuracy'
    _TASK = "Lasso Logistic Regression"

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.name = "Lasso Logistic Regression"

    def compute_cost(self, y, y_pred, theta):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Ground truth target values

        y_pred : array of shape (n_samples,)
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
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        # Compute regularization
        J_reg = (self.alpha / n_samples) * np.linalg.norm(theta, ord=1)
        # Compute lasso regularized cost
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_pred)) + \
            np.multiply(1-y, np.log(1-y_pred))) + J_reg
        return J        

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features+1)
            Input data

        y : array of shape (n_samples,)
            Ground truth target values

        y_pred : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        dZ = y_pred-y
        dW = 1/n_samples * (X.T.dot(dZ) + self.alpha * np.sign(theta))
        return(dW)                     

# --------------------------------------------------------------------------- #
#                          RIDGE LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #            
class RidgeLogisticRegression(LogisticRegression):
    """Logistic Regression Algorithm"""
    _DEFAULT_METRIC = 'accuracy'
    _TASK = "Ridge Logistic Regression"

    def __init__(self, alpha=1):
        self.alpha = alpha
        self.name = "Ridge Logistic Regression"

    def compute_cost(self, y, y_pred, theta):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Ground truth target values

        y_pred : array of shape (n_samples,)
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
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        # Compute regularization
        J_reg = (self.alpha / (2*n_samples)) * np.linalg.norm(theta)**2
        # Compute lasso regularized cost
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_pred)) + \
            np.multiply(1-y, np.log(1-y_pred))) + J_reg
        return J        

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features+1)
            Input data

        y : array of shape (n_samples,)
            Ground truth target values

        y_pred : array of shape (n_samples,)
            Predictions 

        theta : array of shape (n_features+1,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        dZ = y_pred-y
        dW = 1/n_samples * (X.T.dot(dZ) + self.alpha * theta)
        return(dW)                             

# --------------------------------------------------------------------------- #
#                       ELASTIC NET LOGISTIC REGRESSION                       #
# --------------------------------------------------------------------------- #            
class ElasticNetLogisticRegression(LogisticRegression):
    """Logistic Regression Algorithm"""
    _DEFAULT_METRIC = 'accuracy'
    _TASK = "ElasticNet Logistic Regression"

    def __init__(self, alpha=1, ratio=0.5):
        self.alpha=alpha
        self.ratio=ratio
        self.name = "ElasticNet Logistic Regression" 

    def compute_cost(self, y, y_pred, theta):
        """Computes the binary cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples,)
            Ground truth target values

        y_pred : array of shape (n_samples,)
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
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        # Compute regularization
        l1_contr = self.ratio * np.linalg.norm(theta, ord=1)
        l2_contr = (1 - self.ratio) * 0.5 * np.linalg.norm(theta)**2        
        J_reg = float(1./n_samples) * self.alpha * (l1_contr + l2_contr)
        # Compute elasticnet regularized cost
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_pred)) + \
            np.multiply(1-y, np.log(1-y_pred))) + J_reg
        return J        

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features+1)
            Input data

        y : array of shape (n_samples,)
            Ground truth target values

        y_pred : array of shape (n_samples,)
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
        dZ = y_pred-y
        dW = 1/n_samples  * (X.T.dot(dZ) + np.multiply(alpha, np.add(l1_contr, l2_contr)))
        return(dW)                       