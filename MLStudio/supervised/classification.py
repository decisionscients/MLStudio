#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : classification.py                                                 #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                      #
# Last Modified : Saturday, March 21st 2020, 5:47:08 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes supporting binary and multinomial classification ."""
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from mlstudio.utils.data_manager import data_split, one_hot

# --------------------------------------------------------------------------- #
#                          LOGISTIC REGRESSION                                #
# --------------------------------------------------------------------------- #            
class LogisticRegression:
    """Logistic Regression Algorithm"""
    _DEFAULT_METRIC = 'accuracy'
    _TASK = "Logistic Regression"

    def __init__(self):
        pass

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
        y = np.atleast_2d(y).reshape(-1,1)
        y_pred = np.atleast_2d(y_pred).reshape(-1,1)
        n_samples = y.shape[0]
        dW = 1/n_samples * X.T.dot(y_pred-y)
        return(dW)             