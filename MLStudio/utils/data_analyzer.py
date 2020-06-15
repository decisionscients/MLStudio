#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : data_analyzer.py                                                  #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, March 23rd 2020, 9:17:20 am                         #
# Last Modified : Monday, March 23rd 2020, 9:17:21 am                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Data analysis helper functions."""
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
# --------------------------------------------------------------------------- #
def compute_gradient_norm(X):
    """Computes norm of gradient with separate bias and weights."""
    g = np.concatenate(X['bias'], X['weights'])
    return np.linalg.norm(g)
# --------------------------------------------------------------------------- #
def standardized_residuals(residuals):
    """Computes standardized residuals."""
    residuals = residuals.ravel()
    return residuals/np.std(residuals)  

# --------------------------------------------------------------------------- #
def uniform_order_stat(x):
    """Estimates uniform order statistics medians for the normal distribution."""
    positions = np.arange(1, len(x)+1)
    n = len(positions)
    u_i = (positions-0.375)/(n+0.25)
    return u_i
# --------------------------------------------------------------------------- #
def z_score(x):
    """Computes z-scores for a series of values."""
    mu = np.mean(x)
    std = np.std(x)
    z = (x-mu)/std
    return z

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
    
# --------------------------------------------------------------------------  #
def cosine(a,b):
    """Returns the cosine similarity between two vectors."""
    numerator = a.dot(b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    return numerator / denominator

# --------------------------------------------------------------------------  #
def describe_numeric_array(x, fmt='dict'):
    """Returns descriptive statistics for a numeric array."""
    d = {}
    d['count'] = len(x)
    d['min'] = np.min(x)
    d['max'] = np.max(x)
    d['mean'] = np.mean(x)
    d['std'] = np.std(x)
    percentiles = [25, 50, 75]
    for p in percentiles:
        key = str(p) + 'th percentile' 
        d[key] = np.percentile(x, p)
    d['skew'] = skew(x, axis=None)
    d['kurtosis'] = kurtosis(x, axis=None)
    if fmt != 'dict':
        d = pd.DataFrame(d, index=[0])
    return d

def describe_categorical_array(x, fmt='dict'):
    """Returns descriptive statistics for a categorical array."""
    d = {}
    unique, pos = np.unique(x, return_inverse=True)
    counts = np.bincount(pos)
    maxpos = counts.argmax()
    d['count'] = len(x)
    d['unique'] = np.unique(x)
    d['top'] = unique[maxpos]
    d['freq'] = counts[maxpos]
    if fmt != 'dict':
        d = pd.DataFrame(d, index=[0])    
    return d






    
