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
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator

from mlstudio.supervised.estimator.callbacks import Callback
from mlstudio.supervised.estimator.debugging import GradientCheck
from mlstudio.supervised.estimator.early_stop import EarlyStop
from mlstudio.supervised.estimator.gradient import GradientDescentClassifier
from mlstudio.supervised.classification import LogisticRegression

# --------------------------------------------------------------------------  #
#                          TEST ALGORITHMS                                    #
# --------------------------------------------------------------------------  #

# @mark.logistic_regression_check
# @parametrize_with_checks([GradientDescentClassifier])
# def test_classification_algorithm(estimator, check):
#     check(estimator)

# --------------------------------------------------------------------------  #
#                          TEST GRADIENTS                                     #
# --------------------------------------------------------------------------  #
@mark.logistic_regression_gradients
@mark.parametrize("algorithm", [LogisticRegression()])
def test_classification_gradients(get_logistic_regression_data, algorithm):
    X, y = get_logistic_regression_data    
    gradient_check = GradientCheck()
    est = GradientDescentClassifier(algorithm=algorithm, gradient_check=GradientCheck())        
    est.fit(X, y)
    
