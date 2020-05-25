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
from mlstudio.utils.validation import check_X_y, check_X, check_is_fitted, is_multilabel
from mlstudio.utils.validation import is_multilabel

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
@mark.utils
@mark.data_manager
@mark.data_checks
@mark.is_one_hot
def test_is_one_hot(get_data_management_data):
    d = get_data_management_data
    for k, y in d.items():        
        msg = "Is one-hot of " + k + " didn't work."
        if k == 'one_hot':
            assert is_one_hot(y), msg
        else:
            assert not is_one_hot(y), msg

# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.data_checks
@mark.is_multilabel
def test_is_multilabel(get_data_management_data):
    d = get_data_management_data
    for k, y in d.items():        
        msg = "Is multilabel of " + k + " didn't work."
        if 'multilabel' in k:
            assert is_multilabel(y), msg
        else:
            assert not is_multilabel(y), msg


# --------------------------------------------------------------------------  #
#                        TEST VECTOR SCALER                                   #
# --------------------------------------------------------------------------  #  
clip_norm = [1,2,10]
low = [1e-16, 1e15, 1e-16]
high = [1e-15, 1e16, 1e-15] 
@mark.utils
@mark.data_manager
@mark.vector_scaler
def test_vector_scaler_normalize():            
    for s in zip(clip_norm, low, high):
        X = np.random.default_rng().uniform(low=s[1], high=s[2], size=20)
        scaler = GradientScaler(method="n", lower_threshold=s[1], upper_threshold=s[2], clip_norm=s[0])
        X_new = scaler.fit_transform(X)
        assert np.isclose(np.linalg.norm(X_new),s[0]), "Normalization didn't work"
        X_old = scaler.inverse_transform(X_new)
        assert np.allclose(X, X_old), "Denormalization didn't work"

@mark.utils
@mark.data_manager
@mark.vector_scaler
def test_vector_scaler_clip():      
    for s in zip(clip_norm, low, high):
        X = np.random.default_rng().uniform(low=s[1], high=s[2], size=20)
        scaler = GradientScaler(method="c", lower_threshold=s[1], upper_threshold=s[2], clip_norm=s[0])
        X_new = scaler.fit_transform(X)
        assert len(X_new[X_new < s[1]])==0, "Clipping didn't work"
        assert len(X_new[X_new > s[2]])==0, "Clipping didn't work"


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
    print(train_proportions)
    print(test_proportions)







# %%
