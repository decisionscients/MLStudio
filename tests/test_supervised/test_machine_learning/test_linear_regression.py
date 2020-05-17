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

from mlstudio.supervised.callbacks.base import Callback
from mlstudio.supervised.callbacks.debugging import GradientCheck
from mlstudio.supervised.callbacks.early_stop import EarlyStop
from mlstudio.supervised.callbacks.learning_rate import Constant, TimeDecay, SqrtTimeDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialSchedule, PowerSchedule
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentRegressor
from mlstudio.supervised.core.scorers import MSE
from mlstudio.supervised.core.cost import MSE
from mlstudio.supervised.core.regularization import L1, L2, L1_L2


# --------------------------------------------------------------------------  #
#                         REGULARIZATION TESTING                              #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentRegressor(cost=MSE()),
    GradientDescentRegressor(cost=MSE(regularization=L1())),
    GradientDescentRegressor(cost=MSE(regularization=L2())),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()))
]
@mark.regression
@mark.regression_regularization
@parametrize_with_checks(scenarios)
def test_regression_regularization(estimator, check):
    check(estimator)

# --------------------------------------------------------------------------  #
#                          TEST GRADIENTS                                     #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentRegressor(cost=MSE(), gradient_check=True),
    GradientDescentRegressor(cost=MSE(regularization=L1()), gradient_check=True),
    GradientDescentRegressor(cost=MSE(regularization=L2()), gradient_check=True),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), gradient_check=True)
]

@mark.regression
@mark.regression_gradients
@parametrize_with_checks(scenarios)
def test_regression_gradients(estimator, check):
    check(estimator)
    

# --------------------------------------------------------------------------  #
#                              TEST EARLYSTOP                                 #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentRegressor(cost=MSE(), early_stop=EarlyStop()),
    GradientDescentRegressor(cost=MSE(regularization=L1()), early_stop=EarlyStop()),
    GradientDescentRegressor(cost=MSE(regularization=L2()), early_stop=EarlyStop()),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), early_stop=EarlyStop())
]        
@mark.regression
@mark.regression_early_stop
@parametrize_with_checks(scenarios)
def test_regression_early_stop(estimator, check):
    check(estimator)

@mark.regression
@mark.regression_early_stop
def test_regression_early_stop_II(get_regression_data, get_regression_data_features):
    X, y = get_regression_data
    for est in scenarios:
        est.fit(X, y)    
        est.summary(features=get_regression_data_features)
        assert est.blackbox_.total_epochs < est.epochs, "Early stop didn't work"

# --------------------------------------------------------------------------  #
#                              TEST LEARNING RATES                            #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentRegressor(cost=MSE(), learning_rate=Constant()),
    GradientDescentRegressor(cost=MSE(regularization=L1()), learning_rate=TimeDecay()),
    GradientDescentRegressor(cost=MSE(regularization=L2()), learning_rate=SqrtTimeDecay()),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), learning_rate=ExponentialDecay()),
    GradientDescentRegressor(cost=MSE(), learning_rate=PolynomialDecay()),
    GradientDescentRegressor(cost=MSE(regularization=L1()), learning_rate=ExponentialSchedule()),
    GradientDescentRegressor(cost=MSE(regularization=L2()), learning_rate=PowerSchedule())    
]        
@mark.regression
@mark.regression_learning_rates
@parametrize_with_checks(scenarios)
def test_regression_learning_rates(estimator, check):
    check(estimator)


# --------------------------------------------------------------------------  #
#                              TEST SGD                                       #
# --------------------------------------------------------------------------  #
@mark.regression
@mark.regression_sgd
@parametrize_with_checks([GradientDescentRegressor(batch_size=1)])
def test_regression_sgd(estimator, check):
    check(estimator)

# --------------------------------------------------------------------------  #
#                              TEST MBGD                                      #
# --------------------------------------------------------------------------  #
@mark.regression
@mark.regression_mbgd
@parametrize_with_checks([GradientDescentRegressor(batch_size=64)])
def test_regression_mbgd(estimator, check):
    check(estimator)    