#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_activations.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 10:24:08 pm                         #
# Last Modified : Monday, June 15th 2020, 10:24:25 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Test Activation Functions."""
import math
import numpy as np
import pytest
from pytest import mark

from scipy.special import expit, softmax

from mlstudio.supervised.core.tasks import LinearRegression, LogisticRegression
from mlstudio.supervised.core.tasks import MultinomialLogisticRegression
# --------------------------------------------------------------------------  #
@mark.tasks
@mark.linear_regression
class LinearRegressionTaskTests:

    def test_linear_regression_output(self, get_regression_data_and_weights):
        theta = {}        
        X, y, theta['weights'] = get_regression_data_and_weights
        theta['bias'] = np.array([0])
        t = LinearRegression()
        y_pred = t.compute_output(theta, X)        
        assert np.allclose(y_pred, y), "Compute output inaccurate"
        y_pred = t.predict(theta, X)        
        assert np.allclose(y_pred, y), "Predict inaccurate"

# --------------------------------------------------------------------------  #
@mark.tasks
@mark.logistic_regression
class LogisticRegressionTaskTests:

    def test_logistic_regression_output(self, get_logistic_regression_data):
        theta = {}        
        X, y = get_logistic_regression_data
        theta['weights'] = np.random.default_rng().uniform(low=0, high=20, size=X.shape[1])
        theta['bias'] = np.array([0])
        t = LogisticRegression()
        y_pred = t.compute_output(theta, X)        
        assert y_pred.shape == y.shape, "Compute output wrong shape"
        y_pred = t.predict(theta, X)        
        assert y_pred.shape == y.shape, "Compute predict wrong shape"

# --------------------------------------------------------------------------  #
@mark.tasks
@mark.softmax_regression
class SoftmaxRegressionTaskTests:

    def test_softmax_regression_output(self, get_softmax_regression_data):
        theta = {}        
        X, y = get_softmax_regression_data
        theta['weights'] = np.random.default_rng().uniform(low=0, high=20, size=(X.shape[1],y.shape[1]))
        theta['bias'] = np.random.default_rng().uniform(low=0, high=20, size=y.shape[1])
        t = MultinomialLogisticRegression()
        y_pred = t.compute_output(theta, X)        
        assert y_pred.shape == y.shape, "Compute output wrong shape"
        y_pred = t.predict(theta, X)        
        assert y_pred.shape == (y.shape[0],), "Compute predict wrong shape"

