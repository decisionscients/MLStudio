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
import warnings

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
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.learning_rate import Constant, TimeDecay, SqrtTimeDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialSchedule, PowerSchedule
from mlstudio.supervised.callbacks.learning_rate import BottouSchedule
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentRegressor
from mlstudio.supervised.core.scorers import MSE
from mlstudio.supervised.core.cost import MSE
from mlstudio.supervised.core.regularization import L1, L2, L1_L2


# --------------------------------------------------------------------------  #
#                         REGULARIZATION TESTING                              #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentRegressor(cost=MSE(clip_threshold=1e-15)),
    GradientDescentRegressor(cost=MSE(regularization=L1(), clip_threshold=1e-15)),
    GradientDescentRegressor(cost=MSE(regularization=L2(), clip_threshold=1e-15)),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2(), clip_threshold=1e-15))
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
        print("Processing " + est.cost.regularization.__class__.__name__)   
        msg = "Poor score from " + regularization + ' on ' + str(X_train.shape[0]) + ' observations.'
        score = est.score(X_val, y_val)
        assert score > 0.5, msg

# --------------------------------------------------------------------------  #
#                          TEST GRADIENTS                                     #
# --------------------------------------------------------------------------  #

@mark.regression
@mark.regression_gradients
@parametrize_with_checks(scenarios)
def test_regression_gradients(estimator, check):
    check(estimator)
    

# --------------------------------------------------------------------------  #
#                              TEST EARLYSTOP                                 #
# --------------------------------------------------------------------------  #
scenarios_early_stop = [
    GradientDescentRegressor(cost=MSE(), early_stop=Stability()),
    GradientDescentRegressor(cost=MSE(regularization=L1()), early_stop=Stability(metric='val_cost')),
    GradientDescentRegressor(cost=MSE(regularization=L2(alpha=0.0001)), early_stop=Stability(metric='train_score')),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), early_stop=Stability(metric='train_cost')),
    GradientDescentRegressor(cost=MSE(), early_stop=Stability(metric='gradient')),
    GradientDescentRegressor(cost=MSE(regularization=L1()), early_stop=Stability(metric='theta')),
    GradientDescentRegressor(cost=MSE(regularization=L2()), early_stop=Stability(metric='gradient')),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), early_stop=Stability(metric='theta'))
]   


@mark.regression
@mark.regression_early_stop
def test_regression_early_stop(get_regression_data_split, get_regression_data_features):
    X_train, X_test, y_train, y_test = get_regression_data_split
    for est in scenarios_early_stop:
        est.fit(X_train, y_train)    
        est.summary(features=get_regression_data_features)
        score = est.score(X_test, y_test)        
        msg = "Early stop didn't work for linear regression with " + est.cost.regularization.name + \
            " regularization, and " + est.early_stop.__class__.__name__ + \
                " early stopping, monitoring " + est.early_stop.metric +\
                    " with epsilon = " + str(est.early_stop.epsilon) 
        if est.blackbox_.total_epochs == est.epochs:
            warnings.warn(msg)
        msg = "Early stop for linear regression with " + est.cost.regularization.name + \
            " regularization, and " + est.early_stop.__class__.__name__ + \
                " early stopping, monitoring " + est.early_stop.metric +\
                    " with epsilon = " + str(est.early_stop.epsilon) +\
                        " received a poor score of " + str(score)
        if score < 0.5:
            warnings.warn(msg)

# --------------------------------------------------------------------------  #
#                              TEST LEARNING RATES                            #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentRegressor(cost=MSE(), learning_rate=Constant(), epochs=3000),
    GradientDescentRegressor(cost=MSE(regularization=L1()), learning_rate=TimeDecay(), epochs=3000),
    GradientDescentRegressor(cost=MSE(regularization=L2()), learning_rate=SqrtTimeDecay(), epochs=3000),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), learning_rate=ExponentialDecay(), epochs=3000),
    GradientDescentRegressor(cost=MSE(), learning_rate=PolynomialDecay(), epochs=3000),
    GradientDescentRegressor(cost=MSE(regularization=L1()), learning_rate=ExponentialSchedule(), epochs=3000),
    GradientDescentRegressor(cost=MSE(regularization=L2()), learning_rate=PowerSchedule(), epochs=3000),    
    GradientDescentRegressor(cost=MSE(regularization=L2()), learning_rate=BottouSchedule(), epochs=3000)
]        
@mark.regression
@mark.regression_learning_rates
@parametrize_with_checks(scenarios)
def test_regression_learning_rates(estimator, check):
    check(estimator)

@mark.regression
@mark.regression_learning_rates_II
def test_regression_learning_rates_II(get_regression_data_split, get_regression_data_features):
    X_train, X_test, y_train, y_test = get_regression_data_split
    for est in scenarios:
        est.fit(X_train, y_train)            
        score = est.score(X_test, y_test)
        learning_rate = est.learning_rate.__class__.__name__
        if learning_rate != 'Constant':
            msg = "Learning rate decay didn't work for " + learning_rate
            l0 = est.blackbox_.epoch_log.get('learning_rate')[0]
            l9 = est.blackbox_.epoch_log.get('learning_rate')[-1]
            assert l0 > l9, msg
        msg = est.learning_rate.__class__.__name__ + " received a poor score of " + str(score)
        assert score > 0.5, msg
        
# --------------------------------------------------------------------------  #
#                              TEST SGD                                       #
# --------------------------------------------------------------------------  #
scenarios_sgd = [
    GradientDescentRegressor(cost=MSE(), early_stop=Stability(), batch_size=1),
    GradientDescentRegressor(cost=MSE(regularization=L1()), early_stop=Stability(metric='val_score'), batch_size=1),
    GradientDescentRegressor(cost=MSE(regularization=L2()), early_stop=Stability(metric='train_score'), batch_size=1),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), early_stop=Stability(metric='gradient'), batch_size=1),
    GradientDescentRegressor(cost=MSE(regularization=L2()), learning_rate=BottouSchedule(), batch_size=1)    
]   


@mark.regression
@mark.regression_sgd
def test_regression_sgd(get_regression_data_split, get_regression_data_features):
    X_train, X_test, y_train, y_test = get_regression_data_split
    for est in scenarios_sgd:
        est.fit(X_train, y_train)            
        score = est.score(X_test, y_test)
        est.summary(features=get_regression_data_features)
        msg = est.learning_rate.__class__.__name__ + " received a poor score of " + str(score)
        assert score > 0.5, msg

# --------------------------------------------------------------------------  #
#                              TEST SGD                                       #
# --------------------------------------------------------------------------  #
scenarios_MBGD = [
    GradientDescentRegressor(cost=MSE(), batch_size=64, epochs=2000),
    GradientDescentRegressor(cost=MSE(),early_stop=Stability(epsilon=0.0001, patience=100), batch_size=64, epochs=2000),
    GradientDescentRegressor(cost=MSE(regularization=L1()), early_stop=Stability(metric='val_score'), batch_size=64),
    GradientDescentRegressor(cost=MSE(regularization=L2()), early_stop=Stability(metric='train_score'), batch_size=64),
    GradientDescentRegressor(cost=MSE(regularization=L1_L2()), learning_rate=BottouSchedule(), early_stop=Stability(metric='val_cost'), batch_size=64, epochs=2000),
    GradientDescentRegressor(cost=MSE(regularization=L2()), learning_rate=BottouSchedule(), batch_size=64)    
]   


@mark.regression
@mark.regression_mbgd
def test_regression_MBGD(get_regression_data_split, get_regression_data_features):
    X_train, X_test, y_train, y_test = get_regression_data_split
    for est in scenarios_MBGD:
        est.fit(X_train, y_train)            
        score = est.score(X_test, y_test)
        est.summary(features=get_regression_data_features)
        msg = est.cost.regularization.__class__.__name__ + " received a poor score of " + str(score)\
            + " after " + str(est.epochs) + " iterations"
        assert score > 0.5, msg        