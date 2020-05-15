#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : optimizers.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 14th 2020, 6:45:45 pm                         #
# Last Modified : Thursday, May 14th 2020, 6:45:45 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Gradient Descent optimizers for updating model parameters.

Module includes the standard gradient descent parameter update class as
well as algorithms such as Momentum, RMSProp, Nesterov accelerated 
gradient, Adagrad, Adadelta and others.

"""
from abc import ABC, abstractmethod, ABCMeta
import numpy as np

# --------------------------------------------------------------------------  #
class Optimizers(ABC):

    def __init__(self):
        raise NotImplementedError("This abstract base class can not be instantiated.")

    @property
    def name(self):
        return "Optimizer Base Class"

    @abstractmethod
    def update_parameters(self, params, gradient, learning_rate):
        pass

# --------------------------------------------------------------------------  #
class Standard(Optimizers):

    def __init__(self):
        pass

    @property
    def name(self):
        return "Standard Gradient Descent Optimizer"

    def update_parameters(self, params, gradient, learning_rate):
        return params - learning_rate * gradient

