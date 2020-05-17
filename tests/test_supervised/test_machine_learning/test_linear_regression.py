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
from mlstudio.supervised.callbacks.early_stop import Performance, Stability
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

@mark.regression
@mark.regression_regularization_II
def test_regression_regularization_II(get_regression_data_split, get_regression_data_features):
    X_train, X_val, y_train, y_val = get_regression_data_split
    for est in scenarios:
        est.fit(X_train, y_train)            
        regularization = est.cost.regularization.__class__.__name__        
        msg = "Poor score from " + regularization + ' on ' + str(X_train.shape[0]) + ' observations.'
        score = est.score(X_val, y_val)
        assert score > 0.5, msg

# --------------------------------------------------------------------------  #
#                          TEST GRADIENTS                                     #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentRegressor(cost=MSE(), gradient_check=True)
]

@mark.regression
@mark.regression_gradients
@parametrize_with_checks(scenarios)
def test_regression_gradients(estimator, check):
    check(estimator)
    

# --------------------------------------------------------------------------  #
#                              TEST EARLYSTOP                                 #
# --------------------------------------------------------------------------  #
scenarios_early_stop = [
    GradientDescentRegressor(cost=MSE(), early_stop=Performance(metric='val_cost')),
    GradientDescentRegressor(cost=MSE(regularization=L1()), early_stop=Performance(metric='val_score')),
    GradientDescentRegressor(cost=MSE(regularization=L2()), early_stop=Performance(metric='train_score')),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), early_stop=Stability(metric='train_cost')),
    GradientDescentRegressor(cost=MSE(), early_stop=Stability(metric='gradient')),
    GradientDescentRegressor(cost=MSE(regularization=L1()), early_stop=Stability(metric='theta')),
    GradientDescentRegressor(cost=MSE(regularization=L2()), early_stop=Stability(metric='train_cost')),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), early_stop=Stability(metric='train_score')),
    GradientDescentRegressor(cost=MSE(regularization=L2()), early_stop=Stability(metric='val_cost')),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), early_stop=Stability(metric='val_score')),        
]        
@mark.regression
@mark.regression_early_stop
@parametrize_with_checks(scenarios_early_stop)
def test_regression_early_stop(estimator, check):
    check(estimator)


@mark.regression
@mark.regression_early_stop_II
def test_regression_early_stop_II(get_regression_data, get_regression_data_features):
    X, y = get_regression_data
    for est in scenarios_early_stop:
        est.fit(X, y)    
        est.summary(features=get_regression_data_features)
        early_stop = est.early_stop.__class__.__name__
        metric = est.early_stop.metric
        msg = "Early stop didn't work for " + early_stop + " monitoring " + metric\
            + " with epsilon = " + str(est.early_stop.epsilon) 
        assert est.blackbox_.total_epochs < est.epochs, msg

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

@mark.regression
@mark.regression_learning_rates_II
def test_regression_learning_rates_II(get_regression_data, get_regression_data_features):
    X, y = get_regression_data
    for est in scenarios:
        est.fit(X, y)            
        learning_rate = est.learning_rate.__class__.__name__
        if learning_rate != 'Constant':
            msg = "Learning rate decay didn't work for " + learning_rate
            l0 = est.blackbox_.epoch_log.get('learning_rate')[0]
            l9 = est.blackbox_.epoch_log.get('learning_rate')[-1]
            assert l0 > l9, msg
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