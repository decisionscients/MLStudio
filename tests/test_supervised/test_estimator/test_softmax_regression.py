#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : test_softmax_regression copy.py                                   #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 12:40:50 pm                        #
# Last Modified : Friday, April 10th 2020, 12:41:07 pm                        #
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
from mlstudio.supervised.softmax_regression import SoftmaxRegression
from mlstudio.supervised.softmax_regression import LassoSoftmaxRegression
from mlstudio.supervised.softmax_regression import RidgeSoftmaxRegression
from mlstudio.supervised.softmax_regression import ElasticNetSoftmaxRegression

# --------------------------------------------------------------------------  #
#                            TEST ACCURACY                                    #
# --------------------------------------------------------------------------  #

@mark.softmax_regression
def test_softmax_regression_accuracy(get_softmax_regression_split_data,
                                      get_softmax_regression_data_features):
    X_train, X_test, y_train, y_test = get_softmax_regression_split_data        
    est = GradientDescentClassifier(epochs=4000,algorithm=SoftmaxRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_softmax_regression_data_features)
    assert est.score(X_test, y_test) > 0.90, "Accuracy less than 0.90"

@mark.softmax_regression
def test_lasso_softmax_regression_accuracy(get_softmax_regression_split_data,
                                      get_softmax_regression_data_features):
    X_train, X_test, y_train, y_test = get_softmax_regression_split_data        
    est = GradientDescentClassifier(epochs=4000,algorithm=LassoSoftmaxRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_softmax_regression_data_features)
    assert est.score(X_test, y_test) > 0.90, "Accuracy less than 0.90"    

@mark.softmax_regression
def test_ridge_softmax_regression_accuracy(get_softmax_regression_split_data,
                                      get_softmax_regression_data_features):
    X_train, X_test, y_train, y_test = get_softmax_regression_split_data        
    est = GradientDescentClassifier(epochs=4000,algorithm=RidgeSoftmaxRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_softmax_regression_data_features)
    assert est.score(X_test, y_test) > 0.90, "Accuracy less than 0.90"      

@mark.softmax_regression
def test_elasticnet_softmax_regression_accuracy(get_softmax_regression_split_data,
                                      get_softmax_regression_data_features):
    X_train, X_test, y_train, y_test = get_softmax_regression_split_data        
    est = GradientDescentClassifier(epochs=4000,algorithm=ElasticNetSoftmaxRegression())        
    est.fit(X_train, y_train)
    est.summary(features=get_softmax_regression_data_features)
    assert est.score(X_test, y_test) > 0.90, "Accuracy less than 0.90"      
# --------------------------------------------------------------------------  #
#                            TEST EARLY STOP                                  #
# --------------------------------------------------------------------------  #

@mark.softmax_regression
@mark.softmax_regression_early_stop
def test_softmax_regression_early_stop(get_softmax_regression_split_data,
                                      get_softmax_regression_data_features):
    X_train, X_test, y_train, y_test = get_softmax_regression_split_data        
    est = GradientDescentClassifier(algorithm=SoftmaxRegression(),
                                    early_stop=EarlyStop(precision=1e-9, patience=200))        
    est.fit(X_train, y_train)
    est.summary(features=get_softmax_regression_data_features)
    assert est.score(X_test, y_test) > 0.80, "Accuracy less than 0.80"
