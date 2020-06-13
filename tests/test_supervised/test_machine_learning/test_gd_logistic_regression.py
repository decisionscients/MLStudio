#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : test_logistic_regression.py                                                #
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

from mlstudio.supervised.core.tasks import LogisticRegression
from mlstudio.supervised.callbacks.base import Callback
from mlstudio.supervised.callbacks.debugging import GradientCheck
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.learning_rate import StepDecay
from mlstudio.supervised.callbacks.learning_rate import TimeDecay, SqrtTimeDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialStepDecay
from mlstudio.supervised.callbacks.learning_rate import PolynomialStepDecay
from mlstudio.supervised.callbacks.learning_rate import PowerSchedule
from mlstudio.supervised.callbacks.learning_rate import BottouSchedule, Improvement
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentClassifier
from mlstudio.supervised.core.scorers import Accuracy
from mlstudio.supervised.core.objectives import CrossEntropy
from mlstudio.supervised.core.regularizers import L1, L2, L1_L2
from mlstudio.utils.data_manager import GradientScaler


# --------------------------------------------------------------------------  #
#                               Q&D TEST                                      #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentClassifier(objective=CrossEntropy()),
    GradientDescentClassifier(
                              objective=CrossEntropy(gradient_scaler=GradientScaler())),
    GradientDescentClassifier(
                             objective=CrossEntropy(regularizer=L1_L2())),
    GradientDescentClassifier(gradient_check=True),
    GradientDescentClassifier(early_stop=Stability()),                                           
    GradientDescentClassifier(learning_rate=BottouSchedule()),                                           

]

@mark.logistic_regression
@mark.logistic_regression_qnd
@parametrize_with_checks(scenarios)
def test_logistic_regression_qnd(estimator, check):
    check(estimator)
# --------------------------------------------------------------------------  #
#                        REGULARIZATION TESTING                               #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentClassifier(objective=CrossEntropy()),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1())),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2())),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1_L2()))
]

@mark.logistic_regression
@mark.logistic_regression_regularizer
@parametrize_with_checks(scenarios)
def test_logistic_regression_regularizer(estimator, check):
    check(estimator)

@mark.logistic_regression
@mark.logistic_regression_regularizer
@mark.logistic_regression_regularizer_II
def test_logistic_regression_regularizer_II(get_logistic_regression_data_split, get_logistic_regression_data_features):
    X_train, X_val, y_train, y_val = get_logistic_regression_data_split
    for est in scenarios:
        est.fit(X_train, y_train)            
        regularizer = est.objective.regularizer.__class__.__name__     
        msg = "Poor score from " + regularizer + ' on ' + str(X_train.shape[0]) + ' observations.'
        score = est.score(X_val, y_val)
        assert score > 0.5, msg


# --------------------------------------------------------------------------  #
#                          TEST GRADIENTS                                     #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentClassifier(objective=CrossEntropy(gradient_scaler=False), 
                             gradient_check=True)
]
@mark.logistic_regression
@mark.logistic_regression_gradients
@parametrize_with_checks(scenarios)
def test_logistic_regression_gradients(estimator, check):
    check(estimator)
    

# --------------------------------------------------------------------------  #
#                              TEST EARLYSTOP                                 #
# --------------------------------------------------------------------------  #
scenarios_early_stop = [
    GradientDescentClassifier(objective=CrossEntropy(), early_stop=Stability()),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1()), early_stop=Stability(metric='val_cost')),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2(alpha=0.0001)), early_stop=Stability(metric='train_score')),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1_L2()), early_stop=Stability(metric='train_cost')),
    GradientDescentClassifier(objective=CrossEntropy(), early_stop=Stability(metric='gradient')),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1()), early_stop=Stability(metric='theta')),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()), early_stop=Stability(metric='gradient')),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1_L2()), early_stop=Stability(metric='theta'))
]   


@mark.logistic_regression
@mark.logistic_regression_early_stop
def test_logistic_regression_early_stop(get_logistic_regression_data_split, get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_data_split
    for est in scenarios_early_stop:
        est.fit(X_train, y_train)    
        est.summary(features=get_logistic_regression_data_features)
        score = est.score(X_test, y_test)        
        msg = "Early stop didn't work for linear logistic_regression with " + est.objective.regularizer.name + \
            " regularizer, and " + est.early_stop.__class__.__name__ + \
                " early stopping, monitoring " + est.early_stop.metric +\
                    " with epsilon = " + str(est.early_stop.epsilon) 
        if est.blackbox_.total_epochs == est.epochs:
            warnings.warn(msg)
        msg = "Early stop for linear logistic_regression with " + est.objective.regularizer.name + \
            " regularizer, and " + est.early_stop.__class__.__name__ + \
                " early stopping, monitoring " + est.early_stop.metric +\
                    " with epsilon = " + str(est.early_stop.epsilon) +\
                        " received a poor score of " + str(score)
        if score < 0.5:
            warnings.warn(msg)

# --------------------------------------------------------------------------  #
#                              TEST LEARNING RATES                            #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescentClassifier(objective=CrossEntropy(), learning_rate=0.01, epochs=1000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1()), learning_rate=TimeDecay(), epochs=1000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()), learning_rate=SqrtTimeDecay(), epochs=1000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1_L2()), learning_rate=ExponentialDecay(), epochs=1000),
    GradientDescentClassifier(objective=CrossEntropy(), learning_rate=PolynomialDecay(),  epochs=1000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1()), learning_rate=ExponentialStepDecay(), epochs=1000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()), learning_rate=PowerSchedule(), epochs=1000),    
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()), learning_rate=BottouSchedule(), epochs=1000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()), learning_rate=Improvement(), epochs=1000)
]        
@mark.logistic_regression
@mark.logistic_regression_learning_rates
@parametrize_with_checks(scenarios)
def test_logistic_regression_learning_rates(estimator, check):
    check(estimator)

@mark.logistic_regression
@mark.logistic_regression_learning_rates_II
def test_logistic_regression_learning_rates_II(get_logistic_regression_data_split, get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_data_split
    for est in scenarios:
        est.fit(X_train, y_train)            
        score = est.score(X_test, y_test)
        learning_rate = est.schedule.__class__.__name__
        if est.schedule:
            msg = "Learning rate decay didn't work for " + learning_rate
            l0 = est.blackbox_.epoch_log.get('learning_rate')[0]
            l9 = est.blackbox_.epoch_log.get('learning_rate')[-1]
            assert l0 > l9, msg
        est.summary()
        msg = est.schedule.__class__.__name__ + " received a poor score of " + str(score)
        assert score > 0.5, msg
        
# --------------------------------------------------------------------------  #
#                              TEST SGD                                       #
# --------------------------------------------------------------------------  #
scenarios_sgd = [
    GradientDescentClassifier(objective=CrossEntropy(), early_stop=Stability(), batch_size=1),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1()), early_stop=Stability(metric='val_score'), batch_size=1),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()), early_stop=Stability(metric='train_score'), batch_size=1),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1_L2()), early_stop=Stability(metric='gradient'), batch_size=1),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()),learning_rate=BottouSchedule(), batch_size=1)    
]   


@mark.logistic_regression
@mark.logistic_regression_sgd
def test_logistic_regression_sgd(get_logistic_regression_data_split, get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_data_split
    for est in scenarios_sgd:
        est.fit(X_train, y_train)            
        score = est.score(X_test, y_test)
        est.summary(features=get_logistic_regression_data_features)
        msg = est.learning_rate.__class__.__name__ + " received a poor score of " + str(score)
        assert score > 0.5, msg

# --------------------------------------------------------------------------  #
#                              TEST SGD                                       #
# --------------------------------------------------------------------------  #
scenarios_MBGD = [
    GradientDescentClassifier(objective=CrossEntropy(), batch_size=64, epochs=2000),
    GradientDescentClassifier(objective=CrossEntropy(),early_stop=Stability(epsilon=0.0001, patience=100), batch_size=64, epochs=2000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1()), early_stop=Stability(metric='val_score'), batch_size=64),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()), early_stop=Stability(metric='train_score'), batch_size=64),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L1_L2()),learning_rate=BottouSchedule(), early_stop=Stability(metric='val_cost'), batch_size=64, epochs=2000),
    GradientDescentClassifier(objective=CrossEntropy(regularizer=L2()),learning_rate=BottouSchedule(), batch_size=64)    
]   


@mark.logistic_regression
@mark.logistic_regression_mbgd
def test_logistic_regression_MBGD(get_logistic_regression_data_split, get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_data_split
    for est in scenarios_MBGD:
        est.fit(X_train, y_train)            
        score = est.score(X_test, y_test)
        est.summary(features=get_logistic_regression_data_features)
        msg = est.objective.regularizer.__class__.__name__ + " received a poor score of " + str(score)\
            + " after " + str(est.epochs) + " iterations"
        assert score > 0.5, msg        