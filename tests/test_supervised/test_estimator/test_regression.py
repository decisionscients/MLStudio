#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : test_regression.py                                                #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, March 22nd 2020, 2:54:17 am                         #
# Last Modified : Monday, March 23rd 2020, 11:44:36 am                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark
import sklearn.linear_model as lm
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator

from sklearn.metrics import zero_one_loss, log_loss, mean_squared_error
from mlstudio.supervised.estimator.callbacks import Callback
from mlstudio.supervised.estimator.debugging import GradientCheck
from mlstudio.supervised.estimator.early_stop import EarlyStop
from mlstudio.supervised.estimator.gradient import GradientDescentRegressor
from mlstudio.supervised.estimator.scorers import MSE
from mlstudio.supervised.ols_regression import OLSRegression
from mlstudio.supervised.regression import LinearRegression
from mlstudio.supervised.regression import LassoRegression
from mlstudio.supervised.regression import RidgeRegression
from mlstudio.supervised.regression import ElasticNetRegression

# --------------------------------------------------------------------------  #
#                       TEST LINEAR REGRESSION OLS                            #
# --------------------------------------------------------------------------  #
@mark.regression
@mark.regression_ols
@parametrize_with_checks([OLSRegression()])
def test_ols_regression(estimator, check):
    check(estimator)

# --------------------------------------------------------------------------  #
#                       BENCHMARK LINEAR REGRESSION OLS                       #
# --------------------------------------------------------------------------  #
@mark.regression
@mark.regression_skl
def test_ols_regression_sklearn(get_regression_data_split):
    X_train, X_test, y_train, y_test = get_regression_data_split
    ols = OLSRegression()
    ols.fit(X_train, y_train)
    ols_score = ols.score(X_test, y_test)
    skl = lm.LinearRegression()
    skl.fit(X_train, y_train)
    skl_score = skl.score(X_test, y_test)
    assert np.isclose(ols_score, skl_score), "Score not close to sklearn score."




# --------------------------------------------------------------------------  #
#                          TEST ALGORITHMS                                    #
# --------------------------------------------------------------------------  #

@mark.regression
@mark.regression_algorithms
@parametrize_with_checks([GradientDescentRegressor(algorithm=LinearRegression()),
                          GradientDescentRegressor(algorithm=LassoRegression()),
                          GradientDescentRegressor(algorithm=RidgeRegression()),
                          GradientDescentRegressor(algorithm=ElasticNetRegression())])
def test_regression_algorithms(estimator, check):
    check(estimator)

# --------------------------------------------------------------------------  #
#                          TEST GRADIENTS                                     #
# --------------------------------------------------------------------------  #
@mark.regression
@mark.gradient_check
@mark.parametrize("algorithm", [LinearRegression(), LassoRegression(),
                                RidgeRegression(), ElasticNetRegression()])
def test_regression_gradients(get_regression_data, algorithm):
    X, y = get_regression_data    
    gradient_check = GradientCheck()
    est = GradientDescentRegressor(algorithm=algorithm, gradient_check=GradientCheck())        
    est.fit(X, y)
    

# --------------------------------------------------------------------------  #
#                              TEST EARLYSTOP                                 #
# --------------------------------------------------------------------------  #        
# @mark.regression
# @mark.regression_early_stop
# @parametrize_with_checks([GradientDescentRegressor(algorithm=LinearRegression(), early_stop=EarlyStop())])
# def test_regression_early_stop(estimator, check):
#     check(estimator)

@mark.regression
@mark.regression_early_stop
def test_regression_early_stop_II(get_regression_data, get_regression_data_features):
    X, y = get_regression_data
    est = GradientDescentRegressor(algorithm=ElasticNetRegression(), early_stop=EarlyStop())
    est.fit(X,y)
    est.summary(features=get_regression_data_features)
    assert est.history_.total_epochs < est.epochs, "Early stop didn't work"

