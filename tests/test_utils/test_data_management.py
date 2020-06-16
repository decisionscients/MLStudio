#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_data_management.py                                           #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, May 11th 2020, 8:33:38 pm                           #
# Last Modified : Monday, May 11th 2020, 8:33:38 pm                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests data management utilities."""
#%%
import numpy as np
import pytest
from pytest import mark
import scipy.sparse as sp

from mlstudio.datasets import load_urls
from mlstudio.utils.data_manager import MinMaxScaler, data_split, GradientScaler
from mlstudio.utils.data_manager import encode_labels, add_bias_term
from mlstudio.utils.data_manager import pack_weights_bias, unpack_weights_bias

# --------------------------------------------------------------------------  #
#                       TEST DATA CHECKS AND PREP                             #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.data_checks
@mark.encode_labels
def test_encode_labels(get_data_management_data):
    d = get_data_management_data
    for k, y in d.items():
        classes = np.unique(y)
        n_classes = len(classes)
        encoded_classes = np.arange(n_classes)
        y_new = encode_labels(y)
        y_new_classes = np.sort(np.unique(y_new))
        msg = "Encoding of " + k + " didn't work."
        assert np.array_equal(encoded_classes, y_new_classes), msg
  

# --------------------------------------------------------------------------  #
#                        TEST GRADIENT SCALER                                 #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.gradient_scaler
@mark.gradient_scaler_1d
def test_gradient_scaler_1d():            
    lower_threshold = 1e-10
    upper_threshold = 1e10
    lows = [1e-20, 1e15, 1] 
    highs = [1e-10, 1e20, 5]
    for g in zip(lows, highs):    
        X = np.random.default_rng().uniform(low=g[0], high=g[1], size=20)                
        X_orig_norm = np.linalg.norm(X)        
        theta = unpack_weights_bias(X)
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        theta_new = scaler.fit_transform(theta)
        X_new = pack_weights_bias(theta_new)
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        theta_old = scaler.inverse_transform(theta_new)
        X_old = pack_weights_bias(theta_old)
        X_old_norm = np.linalg.norm(X_old)

        assert np.isclose(X_orig_norm, X_old_norm), \
            "Reverse transform didn't work\
                \nX_orig_norm = {n1}\nX_old_norm={n2}".format(n1=str(X_orig_norm),
                n2=str(X_old_norm))
        
@mark.utils
@mark.data_manager
@mark.gradient_scaler
@mark.gradient_scaler_2d
def test_gradient_scaler_2d():            
    lower_threshold = 1e-10
    upper_threshold = 1e10
    lows = [1e-20, 1e15, 1] 
    highs = [1e-10, 1e20, 5]
    for g in zip(lows, highs):    
        X = np.random.default_rng().uniform(low=g[0], high=g[1], size=(20,4))                
        X_orig_norm = (np.linalg.norm(X))        
        theta = unpack_weights_bias(X)
        print(theta['bias'])
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        theta_new = scaler.fit_transform(theta)
        X_new = pack_weights_bias(theta_new)
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        theta_old = scaler.inverse_transform(theta_new)
        X_old = pack_weights_bias(theta_old)
        X_old_norm = np.linalg.norm(X_old)

        assert np.allclose(X_orig_norm, X_old_norm), \
            "Reverse transform didn't work\
                \nX_orig_norm = {n1}\nX_old_norm={n2}".format(n1=str(X_orig_norm),
                n2=str(X_old_norm))

# --------------------------------------------------------------------------  #
#                       TEST MINMAX SCALER                                    #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.minmax
def test_minmax_scaler():
    x = np.array([[0,0,22],
                [0,1,17],
                [0,1,2]], dtype=float)
    x_new = np.array([[0,0,1],
                    [0,1,15/20],
                    [0,1,0]], dtype=float)
    scaler = MinMaxScaler()
    x_t = scaler.fit_transform(x)
    assert np.array_equal(x_new, x_t), "Minmax scaler not working"    
# --------------------------------------------------------------------------  #
#                        TEST DATA SPLIT                                      #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.data_split  
def test_data_split():
    X, y = load_urls()
    X_train, X_test, y_train, y_test = data_split(X,y, stratify=True)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    train_values, train_counts = np.unique(y_train, return_counts=True)
    test_values, test_counts = np.unique(y_test, return_counts=True)
    train_proportions = train_counts / n_train
    test_proportions = test_counts / n_test
    assert np.allclose(train_proportions, test_proportions, rtol=1e-2), "Data split stratification problem "







# %%
