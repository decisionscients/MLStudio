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
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator

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
from mlstudio.supervised.observers.debugging import GradientCheck
from mlstudio.supervised.core.objectives import MSE
from mlstudio.supervised.core.optimizers import GradientDescentOptimizer
from mlstudio.supervised.core.optimizers import Momentum
from mlstudio.supervised.core.optimizers import Nesterov
from mlstudio.supervised.core.optimizers import Adagrad
from mlstudio.supervised.core.optimizers import Adadelta
from mlstudio.supervised.core.optimizers import RMSprop
from mlstudio.supervised.core.optimizers import Adam, AdaMax, Nadam
from mlstudio.supervised.core.optimizers import AMSGrad, AdamW, QHAdam
from mlstudio.supervised.core.optimizers import QuasiHyperbolicMomentum
from mlstudio.supervised.core.regularizers import L1, L2, L1_L2
from mlstudio.supervised.core.scorers import R2
# --------------------------------------------------------------------------  #
count = 0
observers = [{'performance': Performance()}, 
                {'early_stop': Performance(mode='active')}]
scorers = [R2()]
learning_rates = [TimeDecay(), StepDecay(), ExponentialDecay(), ExponentialStepDecay(),
                    PolynomialDecay(), PolynomialStepDecay(), PowerSchedule(),
                    BottouSchedule(), Improvement()]
objectives = [MSE(), MSE(regularizer=L1(alpha=0.01)), 
                        MSE(regularizer=L2(alpha=0.01)), 
                        MSE(regularizer=L1_L2(alpha=0.01, ratio=0.5))]
# optimizers = [
#     GradientDescentOptimizer(), Momentum(), Nesterov(),Adagrad(), Adadelta(),
#     RMSprop(), Adam(), AdaMax(), Nadam(), AMSGrad(), AdamW(), QHAdam(),
#     QuasiHyperbolicMomentum()
# ]

scenarios = [[observer, scorer, learning_rate, objective] for observer in observers
                                                                for scorer in scorers
                                                                for learning_rate in learning_rates
                                                                for objective in objectives
        ]

estimators = [GradientDescentRegressor(observers=scenario[0], scorer=scenario[1],
                                    learning_rate=scenario[2], objective=scenario[3]) for scenario in scenarios]
@mark.gd
@mark.regressor_skl_check
@parametrize_with_checks(estimators)
def test_regression_sklearn(estimator, check):    
    observers = [o.name for o in estimator.observers.values()]
    learning_rate = estimator.learning_rate.name
    objective = estimator.objective.name
    regularizer = estimator.objective.regularizer.name if estimator.objective.regularizer else\
        None
    # optimizer = estimator.optimizer.name
    msg = "Checking scenario : observers : {o}, learning_rate : {l}, objective : {ob},\
            regularizer : {r}".format(
                o=str(observers), l=str(learning_rate),
                ob=str(objective), r=str(regularizer))
    print(msg)        
    check(estimator)

@mark.gd
@mark.regressor
def test_regressor(get_regression_data_split):
    X_train, X_test, y_train, y_test = get_regression_data_split
    for estimator in estimators:
        # Extract scenario options
        observers = [o.name for o in estimator.observers.values()]
        learning_rate = estimator.learning_rate.name
        objective = estimator.objective.name
        regularizer = estimator.objective.regularizer.name if estimator.objective.regularizer else\
            None        
        msg = "Checking scenario: observers : {o}, learning_rate : {l}, objective : {ob},\
            regularizer : {r}".format(o=str(observers), l=str(learning_rate),
            ob=str(objective), r=str(regularizer))
        print(msg)                     
        # Fit the model
        estimator.fit(X_train,y_train)
        mls_score = estimator.score(X_test, y_test)
        # Fit sklearn's model
        skl = LinearRegression()
        skl.fit(X_train, y_train)
        skl_score = skl.score(X_test, y_test)
        if mls_score < skl_score:
            print(msg)
        msg = "Score is significantly worst than sklearn's score. MLS score = {m}, \
            Sklearn score = {s}".format(m=str(mls_score), s=str(skl_score))
        # assert np.isclose(mls_score, skl_score, rtol=0.01) or\
        #     mls_score > skl_score, msg
        estimator.summary()



