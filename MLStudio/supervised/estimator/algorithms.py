#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : algorithms.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, March 15th 2020, 6:46:12 pm                         #
# Last Modified : Saturday, March 21st 2020, 11:04:56 pm                      #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Algorithms that perform forward and backward propagation.

Each class exposes the methods to perform forward and backward propagation in
regression, binary classification and multi-classification settings. The
essential methods are:
    propagate_forward : Computes the prediction and cost.
    propagate_backward : Computes the gradient and updates the parameters.

The parameter update is delegated to an optimizer class that controls
the parameter update.

The three classes included herein are:
    Regression : Forward and backward propagation for regression problems.
    BinaryClassification : Forward and backward propagation for 
        logistic regression.
    MultiClassification : Forward and backward propagation for 
        multi-class regression.

"""
from abc import ABC, abstractmethod
import numpy as np

from mlstudio.supervised.estimator.early_stop import EarlyStop
from mlstudio.supervised.estimator.optimizers import Standard
from mlstudio.supervised.estimator.regularizers import NullRegularizer
from mlstudio.utils.data_manager import decode

class Algorithm(ABC):

    @abstractmethod
    def predict(self, X, theta):
        pass

    @abstractmethod
    def compute_cost(self, y, y_pred):
        pass

    @abstractmethod
    def compute_gradient(self, X, y, y_pred):   
        pass

    @abstractmethod
    def propagate_forward(self, X, y, theta):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def propagate_backward(self, X, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")


# --------------------------------------------------------------------------- #
#                          REGRESSION ALGORITHM                               #
# --------------------------------------------------------------------------- #
class Regression(Algorithm):
    """Computes cost."""

    def __init__(self, optimizer=Standard()):        
        self.optimizer = optimizer
        self.regularizer = NullRegularizer()

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
        J = 1/2 * np.mean(y-y_pred)**2 + self.regularizer(theta)
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
        dW = 1/m * X.T.dot(dZ)
        dW = dW + self.regularizer.gradient(theta)
        return(dW)   

    def propagate_forward(self, X, y, theta):
        """Performs forward propagation.

        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        theta : array of shape (n_features,)  
            The model parameters          

        Returns
        -------
        prediction : hypothesis of target given weights and data
        cost : the cost associated with the prediction        

        """
        y_pred = self.predict(X, theta)
        J = self.compute_cost(y, y_pred, theta)
        return y_pred, J

    def propagate_backward(self, X, y, y_pred, theta, learning_rate):
        """Performs backpropagation of errors through the parameters.
        
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

        learning_rate : float
            The learning rate for the optimization

        """

        gradient = self.compute_gradient(X, y, y_pred, theta)
        theta = self.optimizer.update(learning_rate=learning_rate, 
                                     gradient=gradient, theta=theta)
        return gradient, theta


# --------------------------------------------------------------------------- #
#                          BINARY CLASSIFICATION                              #
# --------------------------------------------------------------------------- #
class BinaryClassification(Algorithm):
    """Computes cost."""

    def __init__(self, optimizer=Standard(), regularizer=NullRegularizer()):        
        super(BinaryClassification, self).__init__(optimizer=optimizer,
                                         regularizer=regularizer)

    def _sigmoid(self, z):
        """Computes the sigmoid cumulative distribution function.

        Parameters
        ----------
        z : float
            The results from the decision function.

        Returns
        -------
        Predicted probability associated with the decision function.

        """
        s = 1.0/(1.0+np.exp(-z))
        return s

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
            z = X.dot(theta)
        elif X.shape[1] == len(theta) - 1:
            z = theta[0] + X.dot(theta[1:])
        else:
            raise ValueError("X and parameters theta have incompatible shapes.\
                 X.shape = {xshape}, theta.shape = {thetashape}.".format(
                     xshape=X.shape, thetashape = theta.shape))            
        y_pred = self._sigmoid(z)
        return y_pred

    def compute_cost(self, y, y_pred, theta):
        """Computes the cross-entropy loss.

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
        # Prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)     
        # Compute cross-entropy loss with regularization   
        J = -1*(1/m) * np.sum(np.multiply(y, np.log(y_pred)) + \
            np.multiply(1-y, np.log(1-y_pred))) + self.regularizer(theta)
        return J

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes gradient of cross-entropy function with sigmoid.
        
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
        m = y.shape[0]
        dW = 1/m * X.T.dot(y_pred-y)
        dW = dW + self.regularizer.gradient(theta)

        return(dW) 

    def propagate_forward(self, X, y, theta):
        """Performs forward propagation.

        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        theta : array of shape (n_features,)  
            The model parameters          

        Returns
        -------
        prediction : hypothesis of target given weights and data
        cost : the cost associated with the prediction        

        """
        y_pred = self.predict(X, theta)
        J = self.compute_cost(y, y_pred, theta)
        return y_pred, J

    def propagate_backward(self, X, y, y_pred, theta, learning_rate):
        """Performs backpropagation of errors through the parameters.
        
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

        learning_rate : float
            The learning rate for the optimization

        """        
        gradient = self.compute_gradient(X, y, y_pred, theta)
        theta = self.optimizer.update(learning_rate, gradient, theta)
        return gradient, theta


# --------------------------------------------------------------------------- #
#                          MULTI CLASSIFICATION                               #
# --------------------------------------------------------------------------- #
class MultiClassification(Algorithm):
    """Computes cost."""

    def __init__(self, optimizer=Standard(), regularizer=NullRegularizer()):        
        super(MultiClassification, self).__init__(optimizer=optimizer,
                                         regularizer=regularizer)

    def _softmax(self, z):
        """Computes the softmax distribution function.

        Parameters
        ----------
        z : float
            The results from the decision function.

        Returns
        -------
        Predicted probabilities for each class. 

        """
        s = np.exp(-z) / np.exp(-z).sum()
        return s

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
            z = X.dot(theta)
        elif X.shape[1] == len(theta) - 1:
            z = theta[0] + X.dot(theta[1:])
        else:
            raise ValueError("X and parameters theta have incompatible shapes.\
                 X.shape = {xshape}, theta.shape = {thetashape}.".format(
                     xshape=X.shape, thetashape = theta.shape))              
        y_pred = self._softmax(z)
        return y_pred

    def compute_cost(self, y, y_pred, theta):
        """Computes the cross-entropy loss with Softmax.

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
        n_samples = y.shape[0]
        # Convert y to integer if one-hot encoded
        if len(y.shape)>1:
            y = decode(y, axis=1)
        # Prevent division by zero. Note y is NOT one-hot encoded
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        log_likelihood = -np.log(y_pred[range(n_samples),y])
        J = np.sum(log_likelihood) / n_samples + self.regularizer(theta)        
        return(J)

    def compute_gradient(self, X, y, y_pred, theta):
        """Computes gradient of cross-entropy function with sigmoid.
        
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
        m = y.shape[0] 
        # Convert y to integer if one-hot encoded
        if len(y.shape) > 1:
            y = decode(y, axis=1)
        # Prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        y_pred[range(m), y] -= 1
        dy_pred = y_pred/m
        dW = X.T.dot(dy_pred)              
        dW = dW + self.regularizer.gradient(theta)
        return dW   

    def propagate_forward(self, X, y, theta):
        """Performs forward propagation.

        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        theta : array of shape (n_features,)  
            The model parameters          

        Returns
        -------
        prediction : hypothesis of target given weights and data
        cost : the cost associated with the prediction        

        """
        y_pred = self.predict(X, theta)
        J = self.compute_cost(y, y_pred, theta)
        return y_pred, J

    def propagate_backward(self, X, y, y_pred, theta, learning_rate):
        """Performs backpropagation of errors through the parameters.
        
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

        learning_rate : float
            The learning rate for the optimization

        """        
        gradient = self.compute_gradient(X, y, y_pred, theta)
        theta = self.optimizer.update(learning_rate, gradient, theta)
        return gradient, theta

