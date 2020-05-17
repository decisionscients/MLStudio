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
# --------------------------------------------------------------------------  #
class Standard(ABC):
    """Standard gradient descent optimizer."""
    def __call__(self, theta, gradient, learning_rate, **kwargs):
        return theta - learning_rate * gradient

