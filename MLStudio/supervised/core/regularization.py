#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : regularization.py                                                 #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, May 16th 2020, 11:17:15 pm                        #
# Last Modified : Saturday, May 16th 2020, 11:17:15 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes used to regularize cost and gradient computations."""
import numpy as np
# --------------------------------------------------------------------------  #
class Nill:
    """ No Regularization """
    def __init__(self):
        self.name = "No Regularization"
    
    def __call__(self, theta):
        return 0        
    
    def gradient(self, theta):
        return np.zeros(theta.shape)
# --------------------------------------------------------------------------  #
class L1:
    """ Regularization for Lasso Regression """
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.name = "Lasso (L1) Regularization"
    
    def __call__(self, theta):
        return self.alpha * np.linalg.norm(theta, 1)

    def gradient(self, theta):
        return self.alpha * np.sign(theta)
    
# --------------------------------------------------------------------------  #
class L2:
    """ Regularization for Ridge Regression """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.name = "Ridge (L2) Regularization"
    
    def __call__(self, theta):
        return self.alpha * 0.5 * theta.T.dot(theta)

    def gradient(self, theta):
        return self.alpha * theta
# --------------------------------------------------------------------------  #
class L1_L2:
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha=0.01, ratio=0.5):
        self.alpha = alpha
        self.ratio = ratio
        self.name = "Elasticnet (L1_L2) Regularization"

    def __call__(self, theta):
        l1_contr = self.ratio * np.linalg.norm(theta, 1)
        l2_contr = (1 - self.ratio) * 0.5 * theta.T.dot(theta)
        return self.alpha * (l1_contr + l2_contr)

    def gradient(self, theta):
        l1_contr = self.ratio * np.sign(theta)
        l2_contr = (1 - self.ratio) * theta
        return self.alpha * (l1_contr + l2_contr) 