#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : cost.py                                                            #
# Python  : 3.8.1                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 6:46:12 pm                          #
# Last Modified : Sunday, March 15th 2020, 6:46:12 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Cost functions and gradient computations."""
from abc import ABC, abstractmethod
import numpy as np

from mlstudio.utils.data_manager import decode

class Cost(ABC):

    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def gradient(self, X, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

# --------------------------------------------------------------------------- #
#                      REGRESSION COST FUNCTIONS                              #
# --------------------------------------------------------------------------- #
class Quadratic(Cost):
    """Computes cost."""

    def __init__(self):        
        self.name = "Mean Squared Error"

    def __call__(self, y, y_pred):
        """Computes quadratic costs e.g. squared error cost"""
        e = y_pred - y 
        J = 1/2 * np.mean(e**2)
        return(J)

    def gradient(self, X, y, y_pred):
        """Computes quadratic costs gradient with respect to weights"""
        n_samples = y.shape[0]
        y = np.atleast_2d(y).reshape(-1,1)
        y_pred = np.atleast_2d(y_pred).reshape(-1,1)
        dW = 1/n_samples * X.T.dot(y_pred-y)
        return(dW)   

class RegressionCostFactory():
    """Returns the requested cost class."""

    def __call__(self,cost='quadratic'):

        dispatcher = {'quadratic': Quadratic()}
        return(dispatcher.get(cost, None))

# --------------------------------------------------------------------------- #
#               BINARY CLASSIFICATION COST FUNCTIONS                          #
# --------------------------------------------------------------------------- #
class BinaryCrossEntropy(Cost):
    """Computes cost and gradient w.r.t. weights and bias."""
    
    def __init__(self):        
        self.name = "Binary Cross Entropy Loss"        

    def __call__(self, y, y_pred):
        """Computes binary cross entropy (w/sigmoid) costs"""
        n_samples = y.shape[0]
        # Prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_pred)) + np.multiply(1-y, np.log(1-y_pred)))
        return J

    def gradient(self, X, y, y_pred):
        """Computes binary cross entropy (w/sigmoid) gradient w.r.t. weights."""
        y = np.atleast_2d(y).reshape(-1,1)
        y_pred = np.atleast_2d(y_pred).reshape(-1,1)
        n_samples = y.shape[0]
        dW = 1/n_samples * X.T.dot(y_pred-y)
        return(dW)

class BinaryClassificationCostFactory():
    """Returns the requested cost class."""

    def __call__(self,cost='binary_cross_entropy'):

        dispatcher = {'binary_cross_entropy': BinaryCrossEntropy()}
        return(dispatcher.get(cost, None))        

# --------------------------------------------------------------------------- #
#                MULTI CLASSIFICATION COST FUNCTIONS                          #
# --------------------------------------------------------------------------- #
class CategoricalCrossEntropy(Cost):
    """Computes softmax cross entropy (w/softmax) cost and gradient w.r.t. parameters."""
    
    def __init__(self):        
        self.name = "Categorical Cross Entropy Loss"

    def __call__(self, y, y_pred):
        """Computes cross entropy (w/softmax) costs"""
        n_samples = y.shape[0]
        # Convert y to integer if one-hot encoded
        if len(y.shape)>1:
            y = decode(y, axis=1)
        # Prevent division by zero. Note y is NOT one-hot encoded
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)        
        log_likelihood = -np.log(y_pred[range(n_samples),y])
        J = np.sum(log_likelihood) / n_samples        
        return(J)

    def gradient(self, X, y, y_pred):
        """Computes cross entropy (w/softmax) gradient w.r.t. parameters."""       
        n_samples = y.shape[0] 
        # Convert y to integer if one-hot encoded
        if len(y.shape) > 1:
            y = decode(y, axis=1)
        # Prevent division by zero
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)
        y_pred[range(n_samples), y] -= 1
        dy_pred = y_pred/n_samples
        dW = X.T.dot(dy_pred)              
        return dW     
        
class MultinomialClassificationCostFactory():
    """Returns the requested cost class."""

    def __call__(self,cost='categorical_cross_entropy'):

        dispatcher = {'categorical_cross_entropy': CategoricalCrossEntropy()}
        return(dispatcher.get(cost, None))                