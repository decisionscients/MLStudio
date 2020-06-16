#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_objectives.py                                                #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 3:45:31 pm                          #
# Last Modified : Monday, June 15th 2020, 3:45:31 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
import math
import numpy as np
import pytest
from pytest import mark

from sklearn.metrics import mean_squared_error

from mlstudio.supervised.core.objectives import MSE, CrossEntropy
from mlstudio.supervised.core.objectives import CategoricalCrossEntropy
from mlstudio.supervised.core.regularizers import L1, L2, L1_L2
# --------------------------------------------------------------------------  #
@mark.objectives
@mark.mse
class MSETests:

    def test_mse_cost(self, get_regression_prediction):
        # Get some predictions
        X, y, y_pred = get_regression_prediction
        # Create some parameters
        theta = {}
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=20)
        theta['bias'] = np.random.default_rng().uniform(low=0, high=1, size=1)
        # Compute expected result
        exp_result = 0.5 * mean_squared_error(y_true=y, y_pred=y_pred)
        # Compute unregularized actual result
        obj = MSE()
        act_result = obj(theta=theta, y=y, y_out=y_pred)
        assert np.isclose(exp_result, act_result), "MSE Error (non-regularized)"
        # Compute L1 regularized actual result
        reg = L1()
        obj = MSE(regularizer=reg)
        act_result_l1 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.isclose(exp_result, act_result_l1), "MSE Error (L1)"
        assert np.not_equal(act_result, act_result_l1), "L1 Regularization didn't happen"
        # Compute L2 regularized actual result
        reg = L2()
        obj = MSE(regularizer=reg)
        act_result_l2 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.isclose(exp_result, act_result_l2), "MSE Error (L2)"
        assert np.not_equal(act_result, act_result_l2), "L2 Regularization didn't happen"        
        assert np.not_equal(act_result_l1, act_result_l2), "L1 Regularization didn't happen"        
        # Compute L1_L2 regularized actual result
        reg = L1_L2()
        obj = MSE(regularizer=reg)
        act_result_l1_l2 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.isclose(exp_result, act_result_l1_l2), "MSE Error (L1_L2)"        
        assert np.not_equal(act_result, act_result_l1_l2), "L1_L2 Regularization didn't happen"                
        assert np.not_equal(act_result_l1, act_result_l1_l2), "L1_L2 Regularization didn't happen"                
        assert np.not_equal(act_result_l2, act_result_l1_l2), "L1_L2 Regularization didn't happen"                

    def test_mse_gradient(self, get_regression_prediction):
        # Get some predictions
        X, y, y_pred = get_regression_prediction
        # Create some parameters
        theta = {}
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=X.shape[1])
        theta['bias'] = np.random.default_rng().uniform(low=0, high=1, size=1)
        # Compute gradient w/o regularization
        obj = MSE()
        grad_no_reg = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L1 Regularization
        obj = MSE(L1())
        grad_l1 = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L2 Regularization
        obj = MSE(L2())
        grad_l2 = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L1_L2 Regularization
        obj = MSE(L1_L2())
        grad_l1_l2 = obj.gradient(theta, X=X, y=y, y_out=y_pred)        
        # Evaluate gradients
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l1['weights'])), "Unregularized and L1 weights are the same"
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l2['weights'])), "Unregularized and L2 weights are the same"
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l1_l2['weights'])), "Unregularized and L1_L2 weights are the same"        
        assert np.allclose(grad_no_reg['weights'], grad_l1['weights'], rtol=0.05), "Unregularized and L1 weights are not close"
        assert np.allclose(grad_no_reg['weights'], grad_l2['weights'], rtol=0.05), "Unregularized and L2 weights are not close"
        assert np.allclose(grad_no_reg['weights'], grad_l1_l2['weights'], rtol=0.05), "Unregularized and L1_L2 weights are not close"        


# --------------------------------------------------------------------------  #
@mark.objectives
@mark.cross_entropy
class CrossEntropyTests:

    def test_cross_entropy_cost(self, get_regression_prediction):
        # Get some predictions
        X, y, y_pred = get_regression_prediction
        # Create some parameters
        theta = {}
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=20)
        theta['bias'] = np.random.default_rng().uniform(low=0, high=1, size=1)
        # Compute expected result
        m = X.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)   
        exp_result = 1/m * np.sum(-y*np.log(y_pred)-(1-y)*np.log(1-y_pred))
        # Compute unregularized actual result
        obj = CrossEntropy()
        act_result = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result), "CrossEntropy Error (non-regularized)"
        # Compute L1 regularized actual result
        reg = L1()
        obj = CrossEntropy(regularizer=reg)
        act_result_l1 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result_l1, rtol=0.01), "CrossEntropy Error (L1)"
        assert np.not_equal(act_result, act_result_l1), "L1 Regularization didn't happen"
        # Compute L2 regularized actual result
        reg = L2()
        obj = CrossEntropy(regularizer=reg)
        act_result_l2 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result_l2, rtol=0.01), "CrossEntropy Error (L2)"
        assert np.not_equal(act_result, act_result_l2), "L2 Regularization didn't happen"        
        assert np.not_equal(act_result_l1, act_result_l2), "L1 Regularization didn't happen"        
        # Compute L1_L2 regularized actual result
        reg = L1_L2()
        obj = CrossEntropy(regularizer=reg)
        act_result_l1_l2 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result_l1_l2, rtol=0.01), "CrossEntropy Error (L1_L2)"        
        assert np.not_equal(act_result, act_result_l1_l2), "L1_L2 Regularization didn't happen"                
        assert np.not_equal(act_result_l1, act_result_l1_l2), "L1_L2 Regularization didn't happen"                
        assert np.not_equal(act_result_l2, act_result_l1_l2), "L1_L2 Regularization didn't happen"                

    def test_cross_entropy_gradient(self, get_regression_prediction):
        # Get some predictions
        X, y, y_pred = get_regression_prediction
        # Create some parameters
        theta = {}
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=X.shape[1])
        theta['bias'] = np.random.default_rng().uniform(low=0, high=1, size=1)
        # Compute gradient w/o regularization
        obj = CrossEntropy()
        grad_no_reg = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L1 Regularization
        obj = CrossEntropy(L1())
        grad_l1 = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L2 Regularization
        obj = CrossEntropy(L2())
        grad_l2 = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L1_L2 Regularization
        obj = CrossEntropy(L1_L2())
        grad_l1_l2 = obj.gradient(theta, X=X, y=y, y_out=y_pred)        
        # Evaluate gradients
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l1['weights'])), "Unregularized and L1 weights are the same"
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l2['weights'])), "Unregularized and L2 weights are the same"
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l1_l2['weights'])), "Unregularized and L1_L2 weights are the same"        
        assert np.allclose(grad_no_reg['weights'], grad_l1['weights'], rtol=0.05), "Unregularized and L1 weights are not close"
        assert np.allclose(grad_no_reg['weights'], grad_l2['weights'], rtol=0.05), "Unregularized and L2 weights are not close"
        assert np.allclose(grad_no_reg['weights'], grad_l1_l2['weights'], rtol=0.05), "Unregularized and L1_L2 weights are not close"        


# --------------------------------------------------------------------------  #
@mark.objectives
@mark.categorical_cross_entropy
class CategoricalCrossEntropyTests:

    def test_categorical_cross_entropy_cost(self, get_regression_prediction):
        # Get some predictions
        X, y, y_pred = get_regression_prediction
        # Create some parameters
        theta = {}
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=20)
        theta['bias'] = np.random.default_rng().uniform(low=0, high=1, size=1)
        # Compute expected result
        m = X.shape[0]
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)   
        exp_result = -np.mean(np.log(y_pred) * y)
        # Compute unregularized actual result
        obj = CategoricalCrossEntropy()
        act_result = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result), "CategoricalCrossEntropy Error (non-regularized)"
        # Compute L1 regularized actual result
        reg = L1()
        obj = CategoricalCrossEntropy(regularizer=reg)
        act_result_l1 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result_l1, rtol=0.01), "CategoricalCrossEntropy Error (L1)"
        assert np.not_equal(act_result, act_result_l1), "L1 Regularization didn't happen"
        # Compute L2 regularized actual result
        reg = L2()
        obj = CategoricalCrossEntropy(regularizer=reg)
        act_result_l2 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result_l2, rtol=0.01), "CategoricalCrossEntropy Error (L2)"
        assert np.not_equal(act_result, act_result_l2), "L2 Regularization didn't happen"        
        assert np.not_equal(act_result_l1, act_result_l2), "L1 Regularization didn't happen"        
        # Compute L1_L2 regularized actual result
        reg = L1_L2()
        obj = CategoricalCrossEntropy(regularizer=reg)
        act_result_l1_l2 = obj(theta=theta, y=y, y_out=y_pred)
        assert np.allclose(exp_result, act_result_l1_l2, rtol=0.01), "CategoricalCrossEntropy Error (L1_L2)"        
        assert np.not_equal(act_result, act_result_l1_l2), "L1_L2 Regularization didn't happen"                
        assert np.not_equal(act_result_l1, act_result_l1_l2), "L1_L2 Regularization didn't happen"                
        assert np.not_equal(act_result_l2, act_result_l1_l2), "L1_L2 Regularization didn't happen"                

    def test_categorical_cross_entropy_gradient(self, get_regression_prediction):
        # Get some predictions
        X, y, y_pred = get_regression_prediction
        # Create some parameters
        theta = {}
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=X.shape[1])
        theta['bias'] = np.random.default_rng().uniform(low=0, high=1, size=1)
        # Compute gradient w/o regularization
        obj = CategoricalCrossEntropy()
        grad_no_reg = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L1 Regularization
        obj = CategoricalCrossEntropy(L1())
        grad_l1 = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L2 Regularization
        obj = CategoricalCrossEntropy(L2())
        grad_l2 = obj.gradient(theta, X=X, y=y, y_out=y_pred)
        # Compute gradient w/ L1_L2 Regularization
        obj = CategoricalCrossEntropy(L1_L2())
        grad_l1_l2 = obj.gradient(theta, X=X, y=y, y_out=y_pred)        
        # Evaluate gradients
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l1['weights'])), "Unregularized and L1 weights are the same"
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l2['weights'])), "Unregularized and L2 weights are the same"
        assert np.all(np.not_equal(grad_no_reg['weights'], grad_l1_l2['weights'])), "Unregularized and L1_L2 weights are the same"        
        assert np.allclose(grad_no_reg['weights'], grad_l1['weights'], rtol=0.1), "Unregularized and L1 weights are not close"
        assert np.allclose(grad_no_reg['weights'], grad_l2['weights'], rtol=0.1), "Unregularized and L2 weights are not close"
        assert np.allclose(grad_no_reg['weights'], grad_l1_l2['weights'], rtol=0.1), "Unregularized and L1_L2 weights are not close"        






