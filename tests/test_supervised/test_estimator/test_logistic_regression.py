#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : test_logistic_regression.py                                       #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, April 1st 2020, 7:17:06 am                       #
# Last Modified : Friday, April 10th 2020, 10:26:05 am                        #
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
from sklearn.metrics import accuracy_score
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator

from mlstudio.supervised.estimator.callbacks import Callback
from mlstudio.supervised.estimator.debugging import GradientCheck
from mlstudio.supervised.estimator.early_stop import EarlyStop
from mlstudio.supervised.estimator.gradient import GradientDescentClassifier
from mlstudio.supervised.logistic_regression import LogisticRegression
from mlstudio.supervised.logistic_regression import LassoLogisticRegression
from mlstudio.supervised.logistic_regression import RidgeLogisticRegression
from mlstudio.supervised.logistic_regression import ElasticNetLogisticRegression

# --------------------------------------------------------------------------  #
#                          TEST GRADIENTS                                     #
# --------------------------------------------------------------------------  #
@mark.logistic_regression
@mark.logistic_regression_gradient_check
@mark.parametrize("algorithm", [LogisticRegression(), LassoLogisticRegression(),
                                RidgeLogisticRegression(), ElasticNetLogisticRegression()])
def test_regression_gradients(get_logistic_regression_data, algorithm):
    X, y = get_logistic_regression_data    
    gradient_check = GradientCheck()
    est = GradientDescentClassifier(algorithm=algorithm, gradient_check=GradientCheck())        
    est.fit(X, y)

# --------------------------------------------------------------------------  #
#                            TEST ACCURACY                                    #
# --------------------------------------------------------------------------  #

@mark.logistic_regression
def test_logistic_regression_accuracy(get_logistic_regression_split_data,
                                      get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_split_data        
    est = GradientDescentClassifier(algorithm=LogisticRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_logistic_regression_data_features)
    assert est.score(X_test, y_test) > 0.85, "Accuracy less than 0.85"

@mark.logistic_regression
def test_lasso_logistic_regression_accuracy(get_logistic_regression_split_data,
                                      get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_split_data        
    est = GradientDescentClassifier(algorithm=LassoLogisticRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_logistic_regression_data_features)
    assert est.score(X_test, y_test) > 0.90, "Accuracy less than 0.90"    

@mark.logistic_regression
def test_ridge_logistic_regression_accuracy(get_logistic_regression_split_data,
                                      get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_split_data        
    est = GradientDescentClassifier(algorithm=RidgeLogisticRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_logistic_regression_data_features)
    assert est.score(X_test, y_test) > 0.90, "Accuracy less than 0.90"      

@mark.logistic_regression
def test_elasticnet_logistic_regression_accuracy(get_logistic_regression_split_data,
                                      get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_split_data        
    est = GradientDescentClassifier(algorithm=ElasticNetLogisticRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_logistic_regression_data_features)
    assert est.score(X_test, y_test) > 0.90, "Accuracy less than 0.90"      
# --------------------------------------------------------------------------  #
#                            TEST EARLY STOP                                  #
# --------------------------------------------------------------------------  #

@mark.logistic_regression
def test_logistic_regression_early_stop(get_logistic_regression_split_data,
                                      get_logistic_regression_data_features):
    X_train, X_test, y_train, y_test = get_logistic_regression_split_data        
    est = GradientDescentClassifier(algorithm=LogisticRegression(),
                                    early_stop=EarlyStop(precision=1e-12, patience=200))        
    est.fit(X_train, y_train)
    est.summary(features=get_logistic_regression_data_features)
    assert est.score(X_test, y_test) > 0.80, "Accuracy less than 0.80"
