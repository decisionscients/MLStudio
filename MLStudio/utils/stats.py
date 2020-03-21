#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : stats.py                                                          #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, March 20th 2020, 9:49:55 pm                         #
# Last Modified : Friday, March 20th 2020, 9:49:56 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
""" Statistics functions."""
import numpy as np

# --------------------------------------------------------------------------- #
#                        STANDARDIZED RESIDUALS                               #
# --------------------------------------------------------------------------- #
def standardized_residuals(residuals):
    """Computes standardized residuals."""
    residuals = residuals.ravel()
    return residuals/np.std(residuals)  

# --------------------------------------------------------------------------- #
#                   Normal Probability Order Statistics                       #
# --------------------------------------------------------------------------- #
def uniform_order_stat(x):
    """Estimates uniform order statistics medians for the normal distribution."""
    positions = np.arange(1, len(x)+1)
    n = len(positions)
    u_i = (positions-0.375)/(n+0.25)
    return u_i
# --------------------------------------------------------------------------- #
#                                  Z-Score                                    #
# --------------------------------------------------------------------------- #
def z_score(x):
    """Computes z-scores for a series of values."""
    mu = np.mean(x)
    std = np.std(x)
    z = (x-mu)/std
    return z

# --------------------------------------------------------------------------- #
#                             Theoretical Quantiles                           #
# --------------------------------------------------------------------------- #    
def theoretical_quantiles(x):
    """Computes the theoretical quantiles for a vector x."""
    u_i =  uniform_order_stat(x)
    q = z_score(u_i)
    return q

def sample_quantiles(x):
    """Computes the sample quantiles for a vector x."""
    x_sorted = np.sort(x)
    q = z_score(x_sorted)
    return q
