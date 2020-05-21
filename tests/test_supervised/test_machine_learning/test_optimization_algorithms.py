#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_optimization_algorithms.py                                   #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 21st 2020, 1:48:51 am                         #
# Last Modified : Thursday, May 21st 2020, 1:48:52 am                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Test optimization algorithms."""
import warnings

import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator
from tabulate import tabulate

from mlstudio.supervised.callbacks.base import Callback
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.learning_rate import TimeDecay, SqrtTimeDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialSchedule, PowerSchedule
from mlstudio.supervised.callbacks.learning_rate import BottouSchedule
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescent
from mlstudio.supervised.core.objectives import Adjiman, BartelsConn, SumSquares
from mlstudio.supervised.core.objectives import ThreeHumpCamel, Himmelblau, Leon
from mlstudio.supervised.core.objectives import Rosenbrock, StyblinskiTank
from mlstudio.supervised.core.optimizers import Momentum, Nesterov, Adagrad
from mlstudio.supervised.core.optimizers import Adadelta, RMSprop, Adam
from mlstudio.supervised.core.optimizers import AdaMax, Nadam, AMSGrad
from mlstudio.supervised.core.regularizers import L1, L2, L1_L2
from mlstudio.utils.data_analyzer import cosine

# --------------------------------------------------------------------------  #
#                       OPTIMIZATION ALGORITHMS W/ BGD                        #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescent(objective=Adjiman(),  optimizer=Momentum(), epochs=5000, theta_init=Adjiman().start),
    GradientDescent(objective=BartelsConn(),  optimizer=Nesterov(), epochs=5000, theta_init=BartelsConn().start),
    GradientDescent(objective=SumSquares(),  optimizer=Adagrad(), epochs=1000, theta_init=SumSquares().start),
    GradientDescent(objective=ThreeHumpCamel(),  optimizer=Adadelta(), epochs=5000, theta_init=ThreeHumpCamel().start),
    GradientDescent(objective=Himmelblau(),  optimizer=RMSprop(), epochs=5000, theta_init=Himmelblau().start),
    GradientDescent(objective=Leon(), optimizer=Adam(), learning_rate=0.001, epochs=5000, theta_init=Leon().start),
    GradientDescent(objective=Rosenbrock(),  optimizer=AdaMax(), epochs=5000, theta_init=Rosenbrock().start),
    GradientDescent(objective=StyblinskiTank(),  optimizer=Nadam(), epochs=5000, theta_init=StyblinskiTank().start),
    GradientDescent(objective=Adjiman(),  optimizer=QHAdam(), epochs=5000, theta_init=Adjiman().start),


    GradientDescent(objective=ThreeHumpCamel(),  optimizer=AMSGrad(), epochs=5000, theta_init=ThreeHumpCamel().start),
    GradientDescent(objective=Himmelblau(),  optimizer=AdamW(), epochs=5000, theta_init=Himmelblau().start),
    
]

@mark.optimization
@mark.optimization_benchmarks
def test_benchmark_gradients():    
    print("\n")
    results = []
    for est in scenarios:
        est.fit()
        diff = round(np.linalg.norm(np.subtract(est.theta_, est.objective.minimum),3))     
        row = [est.optimizer.name, est.objective.name, est.objective.minimum,\
               est.theta_, diff]            
        results.append(row)
        print("\n{o} optimizing {f}:".format(o=est.optimizer.name, f=est.objective.name))
        print("         True minimum: {t}".format(t=str(est.objective.minimum)))
        print("               Result: {r}".format(r=str(est.theta_)))
        print("           Difference: {d}".format(d=str(diff)))

    print(tabulate(results, headers=["Optimizer", "Objective", "True Min.", "Min Hat", "Difference"]))

# @mark.optimization
# @parametrize_with_checks(scenarios)
# def test_regression_qnd(estimator, check):
#     check(estimator)    
        