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
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, ttest_1samp, t
from sklearn.utils import check_array, check_X_y

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
def one_sample_ttest(x, mu=0):
    "Use scipy.stats ttest_1samp to compute t_statistic and p_value"    
    return ttest_1samp(x, mu)

# --------------------------------------------------------------------------- #
def critical_values(x, df, alpha=0.05):
    """Uses scipy.stats inverse survival function to compute critical values."""
    return t.isf([alpha], [df])

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
# --------------------------------------------------------------------------  #
#                             DATA CHECKS                                     #
# --------------------------------------------------------------------------  #
def n_classes(y):
    """Returns the number of classes for a classification dataset."""
    y = check_array(y,accept_sparse=True, accept_large_sparse=True,
                    ensure_2d=False)    
    if y.ndim < 2:
        return len(np.unique(y))
    else:
        return y.shape[1]

def n_features(X):
    """Returns the number of features in a dataset."""
    X = check_array(X,accept_sparse=True, accept_large_sparse=True,
                    ensure_2d=False)
    if X.ndim == 1:
        return 1
    else:
        return X.shape[1]

def get_features(X):
    """Attempts to retrieve the feature names from the dataset."""
    features = OrderedDict()
    if isinstance(X, pd.DataFrame):
        features = X.columns
    elif isinstance(X, (np.ndarray, np.generic)):
        features = X.dtype.names
    if features is None:
        features = ['X_' + str(i) for i in range(X.shape[1])]
    return features

def get_target_info(y):
    """Obtains target data type and class information for classification data."""
    
    if isinstance(y, (pd.DataFrame, pd.Series)):
        y = y.to_numpy()        
    elif isinstance(y, list):
        y = np.array(y)
    
    if 'float' in str(y.dtype):
        data_type = 'Continuous'
        classes = None
        n_classes = None
    else:
        classes = np.unique(y)
        n_classes = len(classes)
        if n_classes == 2:
            data_type = 'Binary'
        else:
            data_type = 'Nominal'        
    return data_type, classes, n_classes
    

    



        





    
