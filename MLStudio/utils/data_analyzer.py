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






    
