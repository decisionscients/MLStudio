#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : optimizer.py                                                      #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, March 21st 2020, 10:32:59 pm                      #
# Last Modified : Saturday, March 21st 2020, 10:33:00 pm                      #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes containing standard and neural network optimization methods.

The classes in this module define the various optimization algorithms
used to update model parameters in machine learning and neural networks.

The optimization algorithms include:
* Momentum
* Nesterov accelerated gradient
* Adagrad
* Adadelta
* RMSprop
* Adam
* AdaMax
* Nadam
* AMSGrad
* AdamW
* QHAdam
* AggMo

See https://ruder.io/optimizing-gradient-descent/index.html#otherrecentoptimizers

"""
from abc import ABC, abstractmethod
import numpy as np
# --------------------------------------------------------------------------  #
#                                OPTIMIZER                                    #            
# --------------------------------------------------------------------------  #
class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self):
        pass
    
    @abstractmethod
    def update(self, learning_rate, gradient, theta):
        pass

class Standard(Optimizer):
    """The standard Gradient Descent optimizer."""
    def __init__(self):
        pass

    def update(self, learning_rate, gradient, theta):
        theta = theta - learning_rate * gradient 
        return theta