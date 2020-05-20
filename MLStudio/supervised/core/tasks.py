#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : task.py                                                           #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Tuesday, May 19th 2020, 10:00:13 pm                         #
# Last Modified : Tuesday, May 19th 2020, 10:00:13 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Defines linear, logistic, and multinomial logistic regression classes."""
from abc import ABC, abstractmethod 

import numpy as np
from sklearn.base import BaseEstimator

from mlstudio.supervised.core.activation import Sigmoid, Softmax
from mlstudio.supervised.core.objectives import MSE, CrossEntropy, CategoricalCrossEntropy 
# --------------------------------------------------------------------------  #
class Task(ABC, BaseEstimator):
    """Defines the base class for all tasks."""

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def compute_output(self, theta, X):
        """Computes output for the task."""
        pass

    @abstractmethod
    def predict(self, theta, X):
        """Computes prediction."""
        pass

# --------------------------------------------------------------------------  #
class LinearRegression(Task):
    """Defines the linear regression task."""

    @property
    def name(self):
        return "Linear Regression"
    
    def compute_output(self, theta, X):
        """Computes linear regression output."""        
        return X.dot(theta)

    def predict(self, theta, X):
        return self.compute_output(theta, X)

# --------------------------------------------------------------------------  #
class LogisticRegression(Task):
    """Defines the logistic regression task."""

    @property
    def name(self):
        return "Logistic Regression"    

    def compute_output(self, theta, X):
        """Computes logistic regression output."""        
        z = X.dot(theta)
        return self._sigmoid(z)

    def predict(self, theta, X):
        o = self.compute_output(theta, X)        
        return np.round(o).astype(int)


# --------------------------------------------------------------------------  #
class MultinomialLogisticRegression(Task):
    """Defines the multinomial logistic regression task."""

    @property
    def name(self):
        return "Multinomial Logistic Regression"    

    def compute_output(self, theta, X):
        """Computes multinomial logistic regression output."""        
        z = X.dot(theta)
        return self._softmax(z)        

    def predict(self, theta, X):
        o = self.compute_output(theta, X)        
        return o.argmax(axis=1)