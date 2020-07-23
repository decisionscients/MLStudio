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
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from mlstudio.utils.data_manager import MinMaxScaler, data_split, GradientScaler
from mlstudio.utils.data_manager import AddBiasTerm, ZeroBiasTerm, unpack_parameters
from mlstudio.factories.data_processor import DataProcessors
# --------------------------------------------------------------------------  #
#                       TEST ADD BIAS TERM TRANSFORMER                        #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.add_bias_term
def test_add_bias_term_np():
    X = np.random.rand(5,5)
    xformer = AddBiasTerm()
    X = xformer.fit_transform(X)
    assert X.shape[1] == 6, "Bias term not added."
    assert np.all(X[:,0] == 1.0), "Column zero not ones."
    # Inverse transform
    X = xformer.inverse_transform(X)
    assert X.shape[1] == 5, "Bias term not removed."
    

@mark.utils
@mark.data_manager
@mark.add_bias_term
def test_add_bias_term_csr():
    X = np.random.rand(5,5)
    X = csr_matrix(X)
    xformer = AddBiasTerm()
    X = xformer.fit_transform(X)
    assert X.shape[1] == 6, "Bias term not added."
    assert np.all(X.toarray()[:,0] == 1.0), "Column zero not ones."   
    # Inverse transform
    X = xformer.inverse_transform(X)
    assert X.shape[1] == 5, "Bias term not removed."    

# --------------------------------------------------------------------------  #
#                       TEST ZERO BIAS TERM TRANSFORMER                       #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.zero_bias_term
def test_zero_bias_term():
    X = np.random.rand(5)
    xformer = ZeroBiasTerm()
    X = xformer.fit_transform(X)    
    assert X[0] == 0.0, "Bias not zeroed out ."
    X = np.random.rand(5,5)
    xformer = ZeroBiasTerm()
    X = xformer.fit_transform(X)    
    assert np.array_equal(X[0,:], np.zeros(shape=X.shape[1])), "Bias not zeroed out ."


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
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        X_new = scaler.fit_transform(X)        
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        X_old = scaler.inverse_transform(X_new)
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
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        X_new = scaler.fit_transform(X)        
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        X_old = scaler.inverse_transform(X_new)
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
    X, y = make_classification(n_classes=4, n_informative=3)
    X_train, X_test, y_train, y_test = data_split(X,y, stratify=True)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    train_values, train_counts = np.unique(y_train, return_counts=True)
    test_values, test_counts = np.unique(y_test, return_counts=True)
    train_proportions = train_counts / n_train
    test_proportions = test_counts / n_test
    assert np.allclose(train_proportions, test_proportions, rtol=0.1), "Data split stratification problem "

# --------------------------------------------------------------------------  #
#                        TEST UNPACK PARAMETERS                               #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.unpack_parameters
def test_unpack_parameters():
    X = np.random.rand(5)
    bias, weights = unpack_parameters(X)
    assert bias == X[0], "Bias not unpacked correctly"
    assert np.array_equal(weights, X[1:]), "weights not unpacked correctly"
    X = np.random.rand(5,3)
    bias, weights = unpack_parameters(X)
    assert bias.shape == (3,), "Bias not unpacked correctly"
    assert np.array_equal(bias, X[0,:]), "Bias not unpacked correctly "
    assert weights.shape == (4,3), "Weights not unpacked correctly"
    assert np.array_equal(weights, X[1:,:]), "Weights not unpacked correctly"


# --------------------------------------------------------------------------  #
#                        TEST DATA PROCESSORS                                 #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_processors
@mark.regression_data_processor
class RegressionDataProcessorTests:

    def test_regression_data_processor_X(self, get_regression_data):
        X, y = get_regression_data
        dp = DataProcessors.regression()
        # No random state no shuffle
        data_1 = dp.fit_transform(X).copy()
        assert data_1['X'].shape[1] == X.shape[1]+1, "Regression data processor: X.shape[1] incorrect."
        # Check deterministic
        data_2 = dp.fit_transform(X).copy()        
        assert data_2['X'].shape[1] == X.shape[1]+1, "Regression data processor: X.shape[1] incorrect."        
        assert np.array_equal(data_1['X'], data_2['X']), "Regression data processor: not deterministic"
        # Check shuffle
        data_3 = dp.fit_transform(X, shuffle=True).copy()
        assert data_3['X'].shape[1] == X.shape[1]+1, "Regression data processor: X.shape[1] incorrect."        
        assert not np.array_equal(data_2['X'], data_3['X']), "Regression data processor: shuffle didn't work."        
        # Check random state
        data_4 = dp.fit_transform(X, random_state=5).copy()
        assert data_3['X'].shape[1] == X.shape[1]+1, "Regression data processor: X.shape[1] incorrect."        
        assert np.array_equal(data_4['X'], data_2['X']), "Regression data processor: state not consistent."        


    def test_regression_data_processor_no_split(self, get_regression_data):
        X, y = get_regression_data
        dp = DataProcessors.regression()        
        data_1 = dp.fit_transform(X, y).copy()
        # No shuffle no random state
        assert data_1['X_train_'].shape[1] == X.shape[1]+1, "Regression data processor: X.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] == X.shape[0], "Regression data processor: y_train_ shape incorrect."
        # Check deterministic
        data_2 = dp.fit_transform(X, y).copy()
        assert np.array_equal(data_1['X_train_'], data_2['X_train_']), "Regression data processor: not deterministic"
        assert np.array_equal(data_1['y_train_'], data_2['y_train_']), "Regression data processor: not deterministic"
        assert data_2['X_train_'].shape[0] == X.shape[0], "Regression data processor: X wrong length"
        # Check shuffle
        data_3 = dp.fit_transform(X, y, shuffle=True).copy()
        assert not np.array_equal(data_3['X_train_'], data_2['X_train_']), "Regression data processor: shuffle not working"
        assert not np.array_equal(data_3['y_train_'], data_2['y_train_']), "Regression data processor: shuffle not working"
        assert data_3['X_train_'].shape[0] == X.shape[0], "Regression data processor: X wrong length"
        # Check random state
        data_4 = dp.fit_transform(X, y, random_state=5).copy()
        assert np.array_equal(data_2['X_train_'], data_4['X_train_']), "Regression data processor: random state not working"
        assert np.array_equal(data_2['y_train_'], data_4['y_train_']), "Regression data processor: random state not working"        
        assert data_4['X_train_'].shape[0] == X.shape[0], "Regression data processor: X wrong length"



    def test_regression_data_processor_split(self, get_regression_data):
        X, y = get_regression_data
        dp = DataProcessors.regression()        
        data_1 = dp.fit_transform(X, y, val_size=0.3).copy()
        # No shuffle no random state
        assert data_1['X_train_'].shape[1] == X.shape[1]+1, "Regression data processor: X.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] != X.shape[0], "Regression data processor: y_train_ shape incorrect."
        assert data_1['X_val_'].shape[1] == X.shape[1]+1, "Regression data processor: X_val_.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] == data_1['X_train_'].shape[0], "Regression data processor: train data mismatch length."
        assert data_1['y_val_'].shape[0] == data_1['X_val_'].shape[0], "Regression data processor: Val data mismatch length."
        # Check deterministic
        data_2 = dp.fit_transform(X, y, val_size=0.3).copy()
        assert np.array_equal(data_1['X_train_'], data_2['X_train_']), "Regression data processor: not deterministic"
        assert np.array_equal(data_1['y_train_'], data_2['y_train_']), "Regression data processor: not deterministic"
        assert data_2['X_train_'].shape[0] != X.shape[0], "Regression data processor: X wrong length"
        assert data_2['X_train_'].shape[0] == data_1['X_train_'].shape[0], "Regression data processor: train data mismatch length."
        assert data_2['X_val_'].shape[0] == data_1['X_val_'].shape[0], "Regression data processor: Val data mismatch length."                
        assert data_2['y_train_'].shape[0] == data_1['X_train_'].shape[0], "Regression data processor: train data mismatch length."
        assert data_2['y_val_'].shape[0] == data_1['X_val_'].shape[0], "Regression data processor: Val data mismatch length."        
        # Check shuffle
        data_3 = dp.fit_transform(X, y, val_size=0.3, shuffle=True).copy()
        assert not np.array_equal(data_3['X_train_'], data_2['X_train_']), "Regression data processor: shuffle not working"
        assert not np.array_equal(data_3['y_train_'], data_2['y_train_']), "Regression data processor: shuffle not working"
        assert not np.array_equal(data_3['X_val_'], data_2['X_val_']), "Regression data processor: shuffle not working"
        assert not np.array_equal(data_3['y_val_'], data_2['y_val_']), "Regression data processor: shuffle not working"        
        assert data_3['X_train_'].shape[0] != X.shape[0], "Regression data processor: X wrong length"
        assert data_2['X_train_'].shape[0] == data_3['X_train_'].shape[0], "Regression data processor: train data mismatch length."
        assert data_2['X_val_'].shape[0] == data_3['X_val_'].shape[0], "Regression data processor: Val data mismatch length."                
        assert data_2['y_train_'].shape[0] == data_3['X_train_'].shape[0], "Regression data processor: train data mismatch length."
        assert data_2['y_val_'].shape[0] == data_3['X_val_'].shape[0], "Regression data processor: Val data mismatch length."                
        # Check random state
        data_4 = dp.fit_transform(X, y, val_size=0.3, random_state=5).copy()
        assert np.array_equal(data_2['X_train_'], data_4['X_train_']), "Regression data processor: shuffle not working"
        assert np.array_equal(data_2['y_train_'], data_4['y_train_']), "Regression data processor: shuffle not working"
        assert np.array_equal(data_2['X_val_'], data_4['X_val_']), "Regression data processor: shuffle not working"
        assert np.array_equal(data_2['y_val_'], data_4['y_val_']), "Regression data processor: shuffle not working"                
        assert data_4['X_train_'].shape[0] != X.shape[0], "Regression data processor: X wrong length"

@mark.utils
@mark.data_processors
@mark.logistic_regression_data_processor
class LogisticRegressionDataProcessorTests:
    
    def test_logistic_regression_data_processor_X(self, get_logistic_regression_data):
        X, y = get_logistic_regression_data
        dp = DataProcessors.binary_classification()
        # No random state no shuffle
        data_1 = dp.fit_transform(X=X).copy()
        assert data_1['X'].shape[1] == X.shape[1]+1, "Logistic regression data processor: X.shape[1] incorrect."
        # Check deterministic
        data_2 = dp.fit_transform(X=X).copy()  
        assert data_2['X'].shape[1] == X.shape[1]+1, "Logistic regression data processor: X.shape[1] incorrect."        
        assert np.array_equal(data_1['X'], data_2['X']), "Logistic regression data processor: not deterministic"
        # Check shuffle
        data_3 = dp.fit_transform(X=X, shuffle=True).copy()     
        assert data_3['X'].shape[1] == X.shape[1]+1, "Logistic regression data processor: X.shape[1] incorrect."        
        assert not np.array_equal(data_2['X'], data_3['X']), "Logistic regression data processor: shuffle didn't work."        
        # Check random state
        data_4 = dp.fit_transform(X, random_state=5).copy()
        assert data_4['X'].shape[1] == X.shape[1]+1, "Logistic regression data processor: X.shape[1] incorrect."        
        assert np.array_equal(data_4['X'], data_2['X']), "Logistic regression data processor: state not consistent."        
    
    def test_logistic_regression_data_processor_no_split(self, get_logistic_regression_data):
        X, y = get_logistic_regression_data
        dp = DataProcessors.binary_classification()        
        data_1 = dp.fit_transform(X, y).copy()
        # No shuffle no random state
        assert data_1['X_train_'].shape[1] == X.shape[1]+1, "Logistic regression data processor: X.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] == X.shape[0], "Logistic regression data processor: y_train_ shape incorrect."
        # Check deterministic
        data_2 = dp.fit_transform(X, y).copy()
        assert np.array_equal(data_1['X_train_'], data_2['X_train_']), "Logistic regression data processor: not deterministic"
        assert np.array_equal(data_1['y_train_'], data_2['y_train_']), "Logistic regression data processor: not deterministic"
        assert data_2['X_train_'].shape[0] == X.shape[0], "Logistic regression data processor: X wrong length"
        # Check shuffle
        data_3 = dp.fit_transform(X, y, shuffle=True).copy()
        assert not np.array_equal(data_3['X_train_'], data_2['X_train_']), "Logistic regression data processor: shuffle not working"
        assert not np.array_equal(data_3['y_train_'], data_2['y_train_']), "Logistic regression data processor: shuffle not working"
        assert data_3['X_train_'].shape[0] == X.shape[0], "Logistic regression data processor: X wrong length"
        # Check random state
        data_4 = dp.fit_transform(X, y, random_state=5).copy()
        assert np.array_equal(data_2['X_train_'], data_4['X_train_']), "Logistic regression data processor: random state not working"
        assert np.array_equal(data_2['y_train_'], data_4['y_train_']), "Logistic regression data processor: random state not working"        
        assert data_4['X_train_'].shape[0] == X.shape[0], "Logistic regression data processor: X wrong length"

    @mark.hit   
    def test_logistic_regression_data_processor_split(self, get_logistic_regression_data):
        X, y = get_logistic_regression_data
        dp = DataProcessors.binary_classification()        
        data_1 = dp.fit_transform(X, y, val_size=0.3).copy()
        # No shuffle no random state
        assert data_1['X_train_'].shape[1] == X.shape[1]+1, "Logistic regression data processor: X.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] != X.shape[0], "Logistic regression data processor: y_train_ shape incorrect."
        assert data_1['X_val_'].shape[1] == X.shape[1]+1, "Logistic regression data processor: X_val_.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] == data_1['X_train_'].shape[0], "Logistic regression data processor: train data mismatch length."
        assert data_1['y_val_'].shape[0] == data_1['X_val_'].shape[0], "Logistic regression data processor: Val data mismatch length."
        assert data_1['y_train_'].shape == (data_1['X_train_'].shape[0],), "Logistic regression data processor: y_train misshape."
        assert data_1['y_val_'].shape == (data_1['X_val_'].shape[0],), "Logistic regression data processor: y_val misshape."
        # Check deterministic
        data_2 = dp.fit_transform(X, y, val_size=0.3).copy()
        assert np.array_equal(data_1['X_train_'], data_2['X_train_']), "Logistic regression data processor: not deterministic"
        assert np.array_equal(data_1['y_train_'], data_2['y_train_']), "Logistic regression data processor: not deterministic"
        assert data_2['X_train_'].shape[0] != X.shape[0], "Logistic regression data processor: X wrong length"
        assert data_2['X_train_'].shape[0] == data_1['X_train_'].shape[0], "Logistic regression data processor: train data mismatch length."
        assert data_2['X_val_'].shape[0] == data_1['X_val_'].shape[0], "Logistic regression data processor: Val data mismatch length."                
        assert data_2['y_train_'].shape[0] == data_1['X_train_'].shape[0], "Logistic regression data processor: train data mismatch length."
        assert data_2['y_val_'].shape[0] == data_1['X_val_'].shape[0], "Logistic regression data processor: Val data mismatch length."        
        assert data_2['y_train_'].shape == (data_2['X_train_'].shape[0],), "Logistic regression data processor: y_train misshape."
        assert data_2['y_val_'].shape == (data_2['X_val_'].shape[0],), "Logistic regression data processor: y_val misshape."        
        # Check shuffle
        data_3 = dp.fit_transform(X, y, val_size=0.3, shuffle=True).copy()
        assert not np.array_equal(data_3['X_train_'], data_2['X_train_']), "Logistic regression data processor: shuffle not working"        
        assert not np.array_equal(data_3['X_val_'], data_2['X_val_']), "Logistic regression data processor: shuffle not working"
        assert data_3['X_train_'].shape[0] != X.shape[0], "Logistic regression data processor: X wrong length"
        assert data_2['X_train_'].shape[0] == data_3['X_train_'].shape[0], "Logistic regression data processor: train data mismatch length."
        assert data_2['X_val_'].shape[0] == data_3['X_val_'].shape[0], "Logistic regression data processor: Val data mismatch length."                
        assert data_2['y_train_'].shape[0] == data_3['X_train_'].shape[0], "Logistic regression data processor: train data mismatch length."
        assert data_2['y_val_'].shape[0] == data_3['X_val_'].shape[0], "Logistic regression data processor: Val data mismatch length."                
        assert data_3['y_train_'].shape == (data_3['X_train_'].shape[0],), "Logistic regression data processor: y_train misshape."
        assert data_3['y_val_'].shape == (data_3['X_val_'].shape[0],), "Logistic regression data processor: y_val misshape."        
        # Check random state
        data_4 = dp.fit_transform(X, y, val_size=0.3, random_state=5).copy()
        assert np.array_equal(data_2['X_train_'], data_4['X_train_']), "Logistic regression data processor: shuffle not working"
        assert np.array_equal(data_2['y_train_'], data_4['y_train_']), "Logistic regression data processor: shuffle not working"
        assert np.array_equal(data_2['X_val_'], data_4['X_val_']), "Logistic regression data processor: shuffle not working"
        assert np.array_equal(data_2['y_val_'], data_4['y_val_']), "Logistic regression data processor: shuffle not working"                
        assert data_4['X_train_'].shape[0] != X.shape[0], "Logistic regression data processor: X wrong length"
        assert data_4['y_train_'].shape == (data_4['X_train_'].shape[0],), "Logistic regression data processor: y_train misshape."
        assert data_4['y_val_'].shape == (data_4['X_val_'].shape[0],), "Logistic regression data processor: y_val misshape."

@mark.utils
@mark.data_processors
@mark.multiclass_classification_data_processor
class MulticlassDataProcessorTests:

    def test_multiclass_classification_data_processor_X(self, get_multiclass_classification_data):
        X, y = get_multiclass_classification_data
        dp =  DataProcessors.multiclass_classification()
        # No random state no shuffle
        data_1 = dp.fit_transform(X).copy()
        assert data_1['X'].shape[1] == X.shape[1]+1, "Multiclass classification data processor: X.shape[1] incorrect."
        # Check deterministic
        data_2 = dp.fit_transform(X).copy()
        assert data_2['X'].shape[1] == X.shape[1]+1, "Multiclass classification data processor: X.shape[1] incorrect."        
        assert np.array_equal(data_1['X'], data_2['X']), "Multiclass classification data processor: not deterministic"
        # Check shuffle
        data_3 = dp.fit_transform(X, shuffle=True).copy()
        assert data_3['X'].shape[1] == X.shape[1]+1, "Multiclass classification data processor: X.shape[1] incorrect."        
        assert not np.array_equal(data_2['X'], data_3['X']), "Multiclass classification data processor: shuffle didn't work."        
        # Check random state
        data_4 = dp.fit_transform(X, random_state=5).copy()
        assert data_4['X'].shape[1] == X.shape[1]+1, "Multiclass classification data processor: X.shape[1] incorrect."        
        assert np.array_equal(data_4['X'], data_2['X']), "Multiclass classification data processor: state not consistent."        


    def test_multiclass_classification_data_processor_no_split(self, get_multiclass_classification_data):
        X, y = get_multiclass_classification_data
        dp =  DataProcessors.multiclass_classification()        
        data_1 = dp.fit_transform(X, y).copy()
        # No shuffle no random state
        assert data_1['X_train_'].shape[1] == X.shape[1]+1, "Multiclass classification data processor: X.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] == X.shape[0], "Multiclass classification data processor: y_train_ shape incorrect."
        assert data_1['y_train_'].shape != y.shape[0], "Multiclass classification data processor: y_train_ shape incorrect."
        # Check deterministic
        data_2 = dp.fit_transform(X, y).copy()
        assert np.array_equal(data_1['X_train_'], data_2['X_train_']), "Multiclass classification data processor: not deterministic"
        assert np.array_equal(data_1['y_train_'], data_2['y_train_']), "Multiclass classification data processor: not deterministic"
        assert data_2['X_train_'].shape[0] == X.shape[0], "Multiclass classification data processor: X wrong length"
        assert data_2['y_train_'].shape != y.shape[0], "Multiclass classification data processor: y_train_ shape incorrect."        
        # Check shuffle
        data_3 = dp.fit_transform(X, y, shuffle=True).copy()
        assert not np.array_equal(data_3['X_train_'], data_2['X_train_']), "Multiclass classification data processor: shuffle not working"
        assert data_3['X_train_'].shape[0] == X.shape[0], "Multiclass classification data processor: X wrong length"
        assert data_3['y_train_'].shape != y.shape[0], "Multiclass classification data processor: y_train_ shape incorrect."                
        # Check random state
        data_4 = dp.fit_transform(X, y, random_state=5).copy()
        assert np.array_equal(data_2['X_train_'], data_4['X_train_']), "Multiclass classification data processor: random state not working"
        assert np.array_equal(data_2['y_train_'], data_4['y_train_']), "Multiclass classification data processor: random state not working"        
        assert data_4['X_train_'].shape[0] == X.shape[0], "Multiclass classification data processor: X wrong length"
        assert data_4['y_train_'].shape != y.shape[0], "Multiclass classification data processor: y_train_ shape incorrect."        



    def test_multiclass_classification_data_processor_split(self, get_multiclass_classification_data):
        X, y = get_multiclass_classification_data
        dp =  DataProcessors.multiclass_classification()        
        data_1 = dp.fit_transform(X, y, val_size=0.3).copy()
        # No shuffle no random state
        assert data_1['X_train_'].shape[1] == X.shape[1]+1, "Multiclass classification data processor: X.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] != X.shape[0], "Multiclass classification data processor: y_train_ shape incorrect."
        assert data_1['X_val_'].shape[1] == X.shape[1]+1, "Multiclass classification data processor: X_val_.shape[1] incorrect."
        assert data_1['y_train_'].shape[0] == data_1['X_train_'].shape[0], "Multiclass classification data processor: train data mismatch length."
        assert data_1['y_val_'].shape[0] == data_1['X_val_'].shape[0], "Multiclass classification data processor: Val data mismatch length."
        assert data_1['y_train_'].shape != (data_1['X_train_'].shape[0],), "Multiclass classification data processor: y_train misshape."
        assert data_1['y_val_'].shape != (data_1['X_val_'].shape[0],), "Multiclass classification data processor: y_val misshape."
        # Check deterministic
        data_2 = dp.fit_transform(X, y, val_size=0.3).copy()
        assert np.array_equal(data_1['X_train_'], data_2['X_train_']), "Multiclass classification data processor: not deterministic"
        assert np.array_equal(data_1['y_train_'], data_2['y_train_']), "Multiclass classification data processor: not deterministic"
        assert data_2['X_train_'].shape[0] != X.shape[0], "Multiclass classification data processor: X wrong length"
        assert data_2['X_train_'].shape[0] == data_1['X_train_'].shape[0], "Multiclass classification data processor: train data mismatch length."
        assert data_2['X_val_'].shape[0] == data_1['X_val_'].shape[0], "Multiclass classification data processor: Val data mismatch length."                
        assert data_2['y_train_'].shape[0] == data_1['X_train_'].shape[0], "Multiclass classification data processor: train data mismatch length."
        assert data_2['y_val_'].shape[0] == data_1['X_val_'].shape[0], "Multiclass classification data processor: Val data mismatch length."        
        assert data_2['y_train_'].shape != (data_2['X_train_'].shape[0],), "Multiclass classification data processor: y_train misshape."
        assert data_2['y_val_'].shape != (data_2['X_val_'].shape[0],), "Multiclass classification data processor: y_val misshape."        
        # Check shuffle
        data_3 = dp.fit_transform(X, y,val_size=0.3, shuffle=True).copy()
        assert not np.array_equal(data_3['X_train_'], data_2['X_train_']), "Multiclass classification data processor: shuffle not working"
        assert not np.array_equal(data_3['X_val_'], data_2['X_val_']), "Multiclass classification data processor: shuffle not working"
        assert data_3['X_train_'].shape[0] != X.shape[0], "Multiclass classification data processor: X wrong length"
        assert data_2['X_train_'].shape[0] == data_3['X_train_'].shape[0], "Multiclass classification data processor: train data mismatch length."
        assert data_2['X_val_'].shape[0] == data_3['X_val_'].shape[0], "Multiclass classification data processor: Val data mismatch length."                
        assert data_2['y_train_'].shape[0] == data_3['X_train_'].shape[0], "Multiclass classification data processor: train data mismatch length."
        assert data_2['y_val_'].shape[0] == data_3['X_val_'].shape[0], "Multiclass classification data processor: Val data mismatch length."                
        assert data_3['y_train_'].shape != (data_3['X_train_'].shape[0],), "Multiclass classification data processor: y_train misshape."
        assert data_3['y_val_'].shape != (data_3['X_val_'].shape[0],), "Multiclass classification data processor: y_val misshape."        
        # Check random state
        data_4 = dp.fit_transform(X, y, val_size=0.3, random_state=5).copy()
        assert np.array_equal(data_2['X_train_'], data_4['X_train_']), "Multiclass classification data processor: shuffle not working"
        assert np.array_equal(data_2['y_train_'], data_4['y_train_']), "Multiclass classification data processor: shuffle not working"
        assert np.array_equal(data_2['X_val_'], data_4['X_val_']), "Multiclass classification data processor: shuffle not working"
        assert np.array_equal(data_2['y_val_'], data_4['y_val_']), "Multiclass classification data processor: shuffle not working"                
        assert data_4['X_train_'].shape[0] != X.shape[0], "Multiclass classification data processor: X wrong length"
        assert data_4['y_train_'].shape != (data_4['X_train_'].shape[0],), "Multiclass classification data processor: y_train misshape."
        assert data_4['y_val_'].shape != (data_4['X_val_'].shape[0],), "Multiclass classification data processor: y_val misshape."
