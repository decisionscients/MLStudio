#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_optimizer.py                                                 #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Tuesday, June 16th 2020, 12:02:30 am                        #
# Last Modified : Tuesday, June 16th 2020, 12:02:31 am                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Integration test for gradient descent pure optimizer class."""
import math
import numpy as np
import pytest
from pytest import mark

from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentPureOptimizer
from mlstudio.supervised.observers.learning_rate import TimeDecay, StepDecay
from mlstudio.supervised.observers.learning_rate import ExponentialDecay
from mlstudio.supervised.observers.learning_rate import ExponentialStepDecay
from mlstudio.supervised.observers.learning_rate import PolynomialDecay
from mlstudio.supervised.observers.learning_rate import PolynomialStepDecay
from mlstudio.supervised.observers.learning_rate import PowerSchedule
from mlstudio.supervised.observers.learning_rate import BottouSchedule
from mlstudio.supervised.observers.learning_rate import Improvement
from mlstudio.supervised.observers.monitor import Performance
from mlstudio.supervised.core.objectives import Adjiman, BartelsConn
from mlstudio.supervised.core.objectives import Himmelblau, Leon
from mlstudio.supervised.core.objectives import Rosenbrock, Branin02
from mlstudio.supervised.core.objectives import StyblinskiTank
from mlstudio.supervised.core.objectives import ThreeHumpCamel, Ursem01
from mlstudio.supervised.core.objectives import Wikipedia
# --------------------------------------------------------------------------  #
#                       TEST OBJECTIVE FUNCTIONS                              #
# --------------------------------------------------------------------------  #
@mark.gradient_descent
@mark.optimizer
class PureOptimizerObjectiveTests:

    objectives = [Adjiman(), BartelsConn(), Himmelblau(), Leon(), Rosenbrock(), Branin02(),
                  StyblinskiTank(), ThreeHumpCamel(), Ursem01(), Wikipedia()]
    
    objectives = [Adjiman()]

    def pure_optimizer_core_test(self):
        for objective in objectives:
            est = GradientDescentPureOptimizer(objective=objective)
            est.fit()
    