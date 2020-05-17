#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : softmax_regression copy.py                                        #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 11:42:42 am                        #
# Last Modified : Friday, April 10th 2020, 11:43:04 am                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes supporting multinomial / softmax classification ."""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_array

from mlstudio.supervised.machine_learning.base import BaseRegressor

# --------------------------------------------------------------------------- #
#                           SOFTMAX REGRESSION                                #
# --------------------------------------------------------------------------- #            
class SoftmaxRegression(BaseRegressor, ClassifierMixin):
    """Softmax Regression Algorithm"""

    def __init__(self):
        pass

    @property
    def name(self):
        return "Softmax Regression"

    @property
    def task(self):
        return "Classification"


    def _softmax(self, Z):
        """Uses softmax to predict the probability of the classes.""" 
        s = (np.exp(Z.T) / np.sum(np.exp(Z), axis=1)).T
        return s

    def compute_output(self, X, theta):
        """Computes output from the softmax function.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The model inputs. Note the number of features includes the coefficient
            for the bias term

        theta : array of shape (n_features, n_classes)  
            Model parameters
        
        """
        return self._softmax(X.dot(theta)) 

    def predict(self, X, theta):
        """Predicts the probabilities or class label, based upon input.

        Computes the softmax probabilities if the method is called from within
        the class. Otherwise, it computes the class label.

        Parameter
        ---------
        X : array of shape [n_samples, n_features]
            The model inputs.

        theta : array of shape (n_features, n_classes)  
            The model parameters

        Note: n_features may or may not include the bias term added prior to 
        training, so we will need to accommodate X of either dimension.               

        Returns
        -------
        prediction : Softmax prediction

        Raises
        ------
        Value error if X and theta have incompatible shapes.
        """    
        if X.shape[1] == theta.shape[0]:
            z = X.dot(theta)
        else:
            z = theta[0] + X.dot(theta[1:])
        a = self._softmax(z)        
        return a.argmax(axis=1)                

    def compute_cost(self, y, y_out, theta):
        """Computes the softmax cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples, y_classes)
            One-hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output   

        theta : array of shape (n_features, n_classes)  
            The model parameters  

        Returns
        -------
        cost : The softmax cross-entropy cost 

        """
        J = np.mean(-np.sum(np.log(y_out) * y, axis=1))
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to parameters.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data

        y : array of shape (n_samples, y_classes)
            One hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output     

        theta : array of shape (n_features, n_classes)  
            The model parameters                             

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """  
        n_samples = y.shape[0]
        dZ =y_out-y
        dW = 1/n_samples * X.T.dot(dZ)
        return(dW)             

# --------------------------------------------------------------------------- #
#                          LASSO LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #            
class LassoSoftmaxRegression(SoftmaxRegression):
    """Softmax Regression Algorithm"""

    def __init__(self, lambda_reg=0.0001):
        self.lambda_reg = lambda_reg        

    @property
    def name(self):
        return "Softmax Regression with Lasso Regularization"                 

    def compute_cost(self, y, y_out, theta):
        """Computes the softmax cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples, y_classes)
            One-hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output   

        theta : array of shape (n_features, n_classes)  
            The model parameters  

        Returns
        -------
        cost : The softmax cross-entropy cost 

        """
        self._validate_hyperparam(self.lambda_reg)
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)    
        # Obtain unregularized cost
        J = super(LassoSoftmaxRegression, self).compute_cost(y, y_out, theta)    
        # Compute regularization
        J_reg = (self.lambda_reg / n_samples) * np.linalg.norm(theta, ord=1)
        # Compute lasso regularized cost
        J = J + J_reg
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to parameters.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data

        y : array of shape (n_samples, y_classes)
            One hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output     

        theta : array of shape (n_features, n_classes)  
            The model parameters                             

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        dZ =y_out-y
        dW = 1/n_samples * (X.T.dot(dZ) + self.lambda_reg * np.sign(theta))
        return(dW)                     

# --------------------------------------------------------------------------- #
#                          RIDGE LOGISTIC REGRESSION                          #
# --------------------------------------------------------------------------- #            
class RidgeSoftmaxRegression(SoftmaxRegression):
    """Softmax Regression Algorithm"""

    def __init__(self, lambda_reg=0.0001):
        self.lambda_reg = lambda_reg
        
    @property
    def name(self):
        return "Softmax Regression with Ridge Regularization"         

    def compute_cost(self, y, y_out, theta):
        """Computes the softmax cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples, y_classes)
            One-hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output   

        theta : array of shape (n_features, n_classes)  
            The model parameters  

        Returns
        -------
        cost : The softmax cross-entropy cost 

        """
        self._validate_hyperparam(self.lambda_reg)
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)
        # Compute unregularized cost.
        J = super(RidgeSoftmaxRegression, self).compute_cost(y, y_out, theta)             
        # Compute regularization
        J_reg = (self.lambda_reg / (2*n_samples)) * np.linalg.norm(theta)**2
        # Compute ridge regularized cost
        J = J + J_reg
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to parameters.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data

        y : array of shape (n_samples, y_classes)
            One hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output     

        theta : array of shape (n_features, n_classes)  
            The model parameters                             

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        dZ =y_out-y
        dW = 1/n_samples * (X.T.dot(dZ) + self.lambda_reg * theta)
        return(dW)                             

# --------------------------------------------------------------------------- #
#                       ELASTIC NET LOGISTIC REGRESSION                       #
# --------------------------------------------------------------------------- #            
class ElasticNetSoftmaxRegression(SoftmaxRegression):
    """Softmax Regression Algorithm"""

    def __init__(self, lambda_reg=0.0001, ratio=0.5):
        self.lambda_reg=lambda_reg
        self.ratio=ratio

    @property
    def name(self):
        return "Softmax Regression with ElasticNet Regularization"     

    def compute_cost(self, y, y_out, theta):
        """Computes the softmax cross-entropy cost.

        Parameters
        ----------
        y : array of shape (n_samples, y_classes)
            One-hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output   

        theta : array of shape (n_features, n_classes)  
            The model parameters  

        Returns
        -------
        cost : The softmax cross-entropy cost 

        """
        self._validate_hyperparam(self.lambda_reg)
        self._validate_hyperparam(self.ratio)

        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)   
        # Compute unregularized cost.
        J = super(ElasticNetSoftmaxRegression, self).compute_cost(y, y_out, theta)                  
        # Compute regularization
        l1_contr = self.ratio * np.linalg.norm(theta, ord=1)
        l2_contr = (1 - self.ratio) * 0.5 * np.linalg.norm(theta)**2        
        J_reg = float(1./n_samples) * self.lambda_reg * (l1_contr + l2_contr)
        # Compute elasticnet regularized cost
        J = J + J_reg
        return J        

    def compute_gradient(self, X, y, y_out, theta):
        """Computes quadratic costs gradient with respect to parameters.
        
        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            Input data

        y : array of shape (n_samples, y_classes)
            One hot encoded ground truth values

        y_out : array of shape (n_samples, y_classes)
            Model output     

        theta : array of shape (n_features, n_classes)  
            The model parameters                             

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        l1_contr = self.ratio * np.sign(theta)
        l2_contr = (1 - self.ratio) * theta        
        lambda_reg = np.asarray(self.lambda_reg, dtype='float64')     
        dZ =y_out-y
        dW = 1/n_samples  * (X.T.dot(dZ) + np.multiply(lambda_reg, np.add(l1_contr, l2_contr)))
        return(dW)                       