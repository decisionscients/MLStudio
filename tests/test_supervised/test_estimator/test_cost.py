#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : test_cost.py                                                       #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 11:22:45 pm                         #
# Last Modified : Sunday, March 15th 2020, 11:22:45 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
import math
import numpy as np
from pytest import mark

from mlstudio.supervised.estimator.cost import Quadratic
from mlstudio.supervised.estimator.cost import BinaryCrossEntropy
from mlstudio.supervised.estimator.cost import CategoricalCrossEntropy

class QuadraticCostTests:

    @mark.cost
    def test_quadratic_cost(self, get_quadratic_y, get_quadratic_y_pred, get_quadratic_cost):
        y = get_quadratic_y
        y_pred = get_quadratic_y_pred
        J = get_quadratic_cost
        J_test = 1/2 * np.mean((y_pred-y)**2)
        assert math.isclose(J, J_test, abs_tol=1)
    
    @mark.cost
    def test_quadratic_cost_gradient(self, get_quadratic_X, get_quadratic_y, get_quadratic_y_pred, get_quadratic_gradient):
        y = get_quadratic_y
        y_pred = get_quadratic_y_pred
        X = get_quadratic_X
        grad = get_quadratic_gradient
        grad_test = 1/y.shape[0] * (y_pred- y).dot(X)        
        for a,b in zip(grad, grad_test):
            assert math.isclose(a, b, abs_tol=1.0)

class BinaryCostTests:

    @mark.cost
    def test_binary_cost(self, get_binary_cost_y, get_binary_cost_y_pred, get_binary_cost):
        y = get_binary_cost_y
        y_pred = get_binary_cost_y_pred
        J = get_binary_cost
        J_test = -1*(1/y.shape[0] * np.sum(np.multiply(y,np.log(y_pred), np.multiply((1-y),np.log(1-y_pred)))))
        assert math.isclose(J, J_test, abs_tol=10**4)

    @mark.cost
    def test_binary_cost_gradient(self, get_binary_cost_X, get_binary_cost_y, get_binary_cost_y_pred, get_binary_cost_gradient):
        X = get_binary_cost_X
        y = get_binary_cost_y
        y_pred = get_binary_cost_y_pred
        grad = get_binary_cost_gradient
        grad_test = X.T.dot(y_pred-y)        
        for a,b in zip(grad, grad_test):
            assert math.isclose(a, b, abs_tol=1.0)

class CategoricalCostTests:

    @mark.cost
    def test_categorical_cost(self, get_categorical_cost_y, get_categorical_cost_y_pred, get_categorical_cost):
        y = get_categorical_cost_y
        y_pred = get_categorical_cost_y_pred
        J = get_categorical_cost
        J_test = -1*(1/y.shape[0] * np.sum(np.multiply(y,np.log(y_pred), np.multiply((1-y),np.log(1-y_pred)))))
        assert math.isclose(J, J_test, abs_tol=10**4)

    @mark.cost
    def test_categorical_cost_gradient(self, get_categorical_cost_X, get_categorical_cost_y, get_categorical_cost_y_pred,
                                       get_categorical_cost_gradient):
        X = get_categorical_cost_X
        y = get_categorical_cost_y
        y_pred = get_categorical_cost_y_pred
        grad = get_categorical_cost_gradient
        grad_test = 1/y.shape[0] * X.T.dot(y_pred-y)        
        for array_a,array_b in zip(grad, grad_test):
            for a, b in zip(array_a, array_b):
                assert math.isclose(a, b, abs_tol=1.0)
