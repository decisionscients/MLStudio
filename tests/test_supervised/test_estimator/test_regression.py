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
from mlstudio.supervised.estimator.early_stop import EarlyStop
from mlstudio.supervised.estimator.gradient import GradientDescentRegressor
from mlstudio.supervised.estimator.optimizers import Standard
from mlstudio.supervised.estimator.scorers import MSE
from mlstudio.supervised.regression import LinearRegression
from mlstudio.supervised.regression import LassoRegression
from mlstudio.supervised.regression import RidgeRegression
from mlstudio.supervised.regression import ElasticNetRegression
from mlstudio.utils.debugging import GradientCheck

# --------------------------------------------------------------------------  #

@mark.regression
@mark.regression_sklearn
@parametrize_with_checks([GradientDescentRegressor(algorithm=LinearRegression()),
                          GradientDescentRegressor(algorithm=LassoRegression()),
                          GradientDescentRegressor(algorithm=RidgeRegression()),
                          GradientDescentRegressor(algorithm=ElasticNetRegression())])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)

class RegressionTests:

    @mark.regression
    def test_regression_gradients(self, get_regression_data):
        X, y = get_regression_data
        optimizer = Standard()        
        regularizer = NullRegularizer()
        algorithm = Regression(optimizer=optimizer, regularizer=regularizer)
        gradient_check = GradientCheck(algorithm)
        est = LinearRegression(algorithm, gradient_check)
        est.fit(X, y)
        failures = est.gradient_check.check_results()
        assert failures == 0, "Got some gradient problems."

        