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
from sklearn.linear_model import LinearRegression as skl_linear_regression
from sklearn.linear_model import LogisticRegression as skl_logistic_regression

from mlstudio.supervised.core.tasks import LinearRegression, LogisticRegression
from mlstudio.supervised.core.tasks import MultinomialLogisticRegression
from mlstudio.utils.data_manager import AddBiasTerm
# --------------------------------------------------------------------------  #
@mark.tasks
@mark.linear_regression
class LinearRegressionTaskTests:

    def test_linear_regression_output(self, make_regression_data):        
        X, y = make_regression_data
        # Obtain sklearn parameters and solution
        skl = skl_linear_regression()
        skl.fit(X,y)
        bias = np.atleast_1d(skl.intercept_)
        coef = skl.coef_    
        theta = np.concatenate((bias, coef))    
        y_pred_skl = skl.predict(X)
        # Add bias term to data
        X = AddBiasTerm().fit_transform(X) 
        # Compute solution
        t = LinearRegression()
        y_pred = t.compute_output(theta, X)        
        assert np.allclose(y_pred_skl, y_pred), "Compute output inaccurate"
        y_pred = t.predict(theta, X)        
        assert np.allclose(y_pred_skl, y_pred), "Predict inaccurate"

# --------------------------------------------------------------------------  #
@mark.tasks
@mark.logistic_regression
class LogisticRegressionTaskTests:

    def test_logistic_regression_output(self, make_classification_data):
        X, y = make_classification_data
        # Obtain sklearn parameters and solution
        skl = skl_logistic_regression(penalty='none')
        skl.fit(X,y)
        bias = np.atleast_1d(skl.intercept_)
        coef = skl.coef_    
        theta = np.concatenate((bias, coef[0]))    
        y_prob_skl = skl.predict_proba(X)[:,1]
        y_pred_skl = skl.predict(X)
        # Add bias term to data
        X = AddBiasTerm().fit_transform(X) 
        # Compute solution
        t = LogisticRegression()
        y_pred = t.compute_output(theta, X)        
        assert np.allclose(y_prob_skl, y_pred), "Compute output inaccurate"
        y_pred = t.predict(theta, X)        
        assert np.allclose(y_pred_skl, y_pred), "Predict inaccurate"

# --------------------------------------------------------------------------  #
@mark.tasks
@mark.softmax_regression
class SoftmaxRegressionTaskTests:

    def test_softmax_regression_output(self, make_multiclass_data):
        X, y = make_multiclass_data
        # Obtain sklearn parameters and solution
        skl = skl_logistic_regression(penalty='none')
        skl.fit(X,y)
        bias = np.atleast_2d(skl.intercept_)
        coef = skl.coef_    
        theta = np.concatenate((bias, coef.T), axis=0)    
        y_prob_skl = skl.predict_proba(X)
        y_pred_skl = skl.predict(X)
        # Add bias term to data
        X = AddBiasTerm().fit_transform(X) 
        # Compute solution
        t = MultinomialLogisticRegression()
        y_pred = t.compute_output(theta, X)        
        assert np.allclose(y_prob_skl, y_pred), "Compute output inaccurate"
        y_pred = t.predict(theta, X)        
        assert np.allclose(y_pred_skl, y_pred), "Predict inaccurate"

