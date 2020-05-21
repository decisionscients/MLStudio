#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_benchmark.py                                                 #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, May 20th 2020, 4:11:11 am                        #
# Last Modified : Wednesday, May 20th 2020, 4:11:11 am                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests benchmark functions."""
import warnings

import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from mlstudio.supervised.callbacks.base import Callback
from mlstudio.supervised.callbacks.debugging import GradientCheck
from mlstudio.supervised.callbacks.early_stop import Stability
from mlstudio.supervised.callbacks.learning_rate import Constant, TimeDecay, SqrtTimeDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialSchedule, PowerSchedule
from mlstudio.supervised.callbacks.learning_rate import BottouSchedule
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescent
from mlstudio.supervised.core.objectives import Adjiman, BartelsConn, SumSquares
from mlstudio.supervised.core.objectives import GoldsteinPrice, Himmelblau, Leon
from mlstudio.supervised.core.objectives import Rosenbrock, StyblinskiTank
from mlstudio.supervised.core.regularizers import L1, L2, L1_L2

# --------------------------------------------------------------------------  #
#                             GRADIENT CHECK                                  #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescent(objective=Adjiman(), epochs=10000, 
                                         learning_rate=TimeDecay(eta0=0.003),
                                         gradient_check=GradientCheck()),
    GradientDescent(objective=BartelsConn(), gradient_check=GradientCheck()),
    GradientDescent(objective=SumSquares(), gradient_check=GradientCheck()),
    GradientDescent(objective=GoldsteinPrice(), gradient_check=GradientCheck()),
    GradientDescent(objective=Himmelblau(), gradient_check=GradientCheck()),
    GradientDescent(objective=Leon(), gradient_check=GradientCheck()),
    GradientDescent(objective=Rosenbrock(), gradient_check=GradientCheck()),
    GradientDescent(objective=StyblinskiTank(), gradient_check=GradientCheck())
    
]

@mark.benchmarks
@mark.benchmarks_gradients
def test_benchmark_gradients():    
    for est in scenarios:
        est.fit()            
        msg = est.objective.name + ' failed convergence. True minimum: {t}.   Empirical minimum: {e}.'.\
            format(t=str(est.objective.minimum), e=str(est.theta_)) 
        assert np.allclose(est.objective.minimum, est.theta_), msg
