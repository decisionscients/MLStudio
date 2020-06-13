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

from mlstudio.supervised.observers.base import Observer
from mlstudio.supervised.observers.performance import Performance
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescent
from mlstudio.supervised.core.objectives import Adjiman, Branin02
from mlstudio.supervised.core.objectives import ThreeHumpCamel, Ursem01
from mlstudio.supervised.core.objectives import StyblinskiTank
from mlstudio.supervised.core.regularizers import L1, L2, L1_L2

# --------------------------------------------------------------------------  #
#                             GRADIENT CHECK                                  #
# --------------------------------------------------------------------------  #
scenarios = [
    GradientDescent(objective=Adjiman(), gradient_check=True),
    GradientDescent(objective=Branin02(), gradient_check=True),    
    GradientDescent(objective=ThreeHumpCamel(), gradient_check=True),
    GradientDescent(objective=Ursem01(), gradient_check=True),
    GradientDescent(objective=StyblinskiTank(), gradient_check=True)
    
]

@mark.benchmarks
@mark.gradients
def test_benchmark_gradients():    
    epsilon = .25
    print("\n")
    for est in scenarios:
        est.fit()    
        pct_diff = abs((np.linalg.norm(est.theta_) - np.linalg.norm(est.objective.minimum)) \
            / np.linalg.norm(est.objective.minimum))
        if pct_diff > epsilon:
            msg = est.optimizer.name + ' on the ' +  est.objective.name + \
                ' function converged to ' + str(est.theta_) + \
                    '.    True minimum: ' + str(est.objective.minimum) + \
                        ' Pct Difference: ' + str(pct_diff)
            print(msg)
        

# --------------------------------------------------------------------------  #
#                                CHECK LOGS                                   #
# --------------------------------------------------------------------------  #
@mark.benchmarks
@mark.logs
def test_benchmark_logs():    
    objective = Adjiman()
    est = GradientDescent(objective=objective)
    est.fit()
    log = est.blackbox_.epoch_log
    # Check lengths
    assert len(log.get('learning_rate')) == est.epochs, "Log error: length of learning_rate array incorrect"
    assert len(log.get('theta')) == est.epochs, "Log error: length of theta array incorrect"
    assert len(log.get('train_cost')) == est.epochs, "Log error: length of train_cost array incorrect"
    assert len(log.get('gradient')) == est.epochs, "Log error: length of gradient array incorrect"
    assert len(log.get('gradient_norm')) == est.epochs, "Log error: length of gradient_norm array incorrect"    
    # Check thetas and gradients
    thetas = log.get('theta')
    gradients = log.get('gradient')
    gradient_norms = log.get('gradient_norm')
    for i in range(len(thetas)):
        theta = thetas[i]
        gradient = objective.gradient(theta)
        assert np.array_equal(gradients[i], gradient), "Log error: Incorrect gradient"
        assert np.isclose(gradient_norms[i], np.linalg.norm(gradients[i])),\
                            "Log error: wrong gradient norm"
