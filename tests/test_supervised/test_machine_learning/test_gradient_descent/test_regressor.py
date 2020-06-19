#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_regressor.py                                                 #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, June 19th 2020, 4:29:58 am                          #
# Last Modified : Friday, June 19th 2020, 4:29:58 am                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Integration test for GradientDescentRegressor class."""
import math
import numpy as np
import pytest
from pytest import mark
from sklearn.linear_model import LinearRegression

from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentRegressor
from mlstudio.supervised.observers.learning_rate import TimeDecay, StepDecay
from mlstudio.supervised.observers.learning_rate import ExponentialDecay
from mlstudio.supervised.observers.learning_rate import ExponentialStepDecay
from mlstudio.supervised.observers.learning_rate import PolynomialDecay
from mlstudio.supervised.observers.learning_rate import PolynomialStepDecay
from mlstudio.supervised.observers.learning_rate import PowerSchedule
from mlstudio.supervised.observers.learning_rate import BottouSchedule
from mlstudio.supervised.observers.learning_rate import Improvement
from mlstudio.supervised.observers.monitor import Performance
from mlstudio.supervised.core.optimizers import GradientDescentOptimizer
from mlstudio.supervised.core.optimizers import Momentum
from mlstudio.supervised.core.optimizers import Nesterov
from mlstudio.supervised.core.optimizers import Adagrad
from mlstudio.supervised.core.optimizers import Adadelta
from mlstudio.supervised.core.optimizers import RMSprop
from mlstudio.supervised.core.optimizers import Adam, AdaMax, Nadam
from mlstudio.supervised.core.optimizers import AMSGrad, AdamW, QHAdam
from mlstudio.supervised.core.optimizers import QuasiHyperbolicMomentum
from mlstudio.supervised.core.scorers import MSE
# --------------------------------------------------------------------------  #

@mark.gd
@mark.regressor
class RegressorTests:

    def test_regressor_no_observers(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split
        # Fit the model
        est = GradientDescentRegressor(epochs=5000)
        est.fit(X_train,y_train)
        mls_score = est.score(X_test, y_test)
        # Fit sklearn's model
        skl = LinearRegression()
        skl.fit(X_train, y_train)
        skl_score = skl.score(X_test, y_test)
        msg = "Score is not close to sklearn's score. MLS score = {m}, \
            Sklearn score = {s}".format(m=str(mls_score), s=str(skl_score))
        assert np.isclose(mls_score, skl_score, rtol=0.01), "Score not close to sklearn's score"



