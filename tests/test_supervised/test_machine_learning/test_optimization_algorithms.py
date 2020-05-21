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

from mlstudio.supervised.callbacks.base import Callback
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.learning_rate import Constant, TimeDecay, SqrtTimeDecay
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
    GradientDescent(objective=Adjiman(), learning_rate=ExponentialDecay(eta0=0.0001), optimizer=Momentum()),
    GradientDescent(objective=BartelsConn(), learning_rate=ExponentialDecay(eta0=0.0001),optimizer=Nesterov()),
    GradientDescent(objective=SumSquares(), learning_rate=ExponentialDecay(eta0=0.0001), optimizer=Adagrad()),
    GradientDescent(objective=ThreeHumpCamel(), learning_rate=ExponentialDecay(eta0=0.0001), optimizer=Adadelta()),
    GradientDescent(objective=Himmelblau(), learning_rate=ExponentialDecay(eta0=0.0001), optimizer=RMSprop()),
    GradientDescent(objective=Leon(),learning_rate=ExponentialDecay(eta0=0.0001), optimizer=Adam()),
    GradientDescent(objective=Rosenbrock(), learning_rate=ExponentialDecay(eta0=0.0001), optimizer=AdaMax()),
    GradientDescent(objective=StyblinskiTank(), learning_rate=ExponentialDecay(eta0=0.0001), optimizer=Nadam()),
    GradientDescent(objective=ThreeHumpCamel(), learning_rate=ExponentialDecay(eta0=0.0001), optimizer=AMSGrad())
    
]

@mark.optimization
@mark.optimization_benchmarks
def test_benchmark_gradients():    
    print("\n")
    results = []
    for est in scenarios:
        est.fit()
        sim=cosine(est.theta_, est.objective.minimum)        
        row = [est.optimizer.name, est.objective.name, est.objective.minimum,\
               est.theta_, sim]            
        results.append(row)
        msg = est.optimizer.name + ' optimizing ' + est.objective.name + \
            ": True minimum: {t}.   Empirical minimum: {e}.   Cosine Sim: {c}".\
            format(t=str(est.objective.minimum), e=str(est.theta_), \
                c=str(sim))
        print(msg) 

    print(tabulate(results, headers=["Optimizer", "Objective", "True Min.", "Min Hat", "Similarity"]))

@mark.optimization
@parametrize_with_checks(scenarios)
def test_regression_qnd(estimator, check):
    check(estimator)    
        