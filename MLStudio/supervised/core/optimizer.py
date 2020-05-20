#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : gradient_descent_optimizers.py                                    #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, May 16th 2020, 9:13:15 pm                         #
# Last Modified : Saturday, May 16th 2020, 9:13:16 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Gradient descent optimization algorithms."""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator  
# --------------------------------------------------------------------------  #
class Optimizer(ABC, BaseEstimator):
    """Base class for all optimizers."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This base class is not implemented.")

    @abstractmethod
    def update(self, gradient, theta, learning_rate, **kwargs):
        """Updates the parameters theta and returns theta and gradient."""
        pass

# --------------------------------------------------------------------------  #
class Standard(Optimizer):
    """Standard gradient descent optimizer."""

    def __init__(self):
        pass
    
    def update(self, gradient, theta, learning_rate, **kwargs):
        grad = gradient(theta, **kwargs)
        theta = theta - learning_rate * grad
        return theta, grad

# --------------------------------------------------------------------------  #
class Momentum(Optimizer):
    """Standard gradient descent optimizer."""

    def __init__(self, gamma=0.9):
        self.gamma = gamma
        self._velocity = 0
    
    def __call__(self, theta, gradient, learning_rate, **kwargs):
        self._velocity = gamma * self._velocity + learning_rate * gradient
        return theta - self._velocity

# --------------------------------------------------------------------------  #
# class Nesterov(Optimizer):
#     """Standard gradient descent optimizer."""

#     def __init__(self, gamma=0.9):
#         self.gamma = gamma
#         self._velocity = 0
    
#     def __call__(self, theta, gradient, learning_rate, **kwargs):
#         self._velocity = gamma * self._velocity + learning_rate * (gradient
#         return theta - self._velocity