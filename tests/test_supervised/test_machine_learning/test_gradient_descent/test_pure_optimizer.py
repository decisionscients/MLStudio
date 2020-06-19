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
from mlstudio.supervised.core.optimizers import GradientDescentOptimizer
from mlstudio.supervised.core.optimizers import Momentum
from mlstudio.supervised.core.optimizers import Nesterov
from mlstudio.supervised.core.optimizers import Adagrad
from mlstudio.supervised.core.optimizers import Adadelta
from mlstudio.supervised.core.optimizers import RMSprop
from mlstudio.supervised.core.optimizers import Adam, AdaMax, Nadam
from mlstudio.supervised.core.optimizers import AMSGrad, AdamW, QHAdam
from mlstudio.supervised.core.optimizers import QuasiHyperbolicMomentum
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
@mark.pure_optimizer
class PureOptimizerObjectiveTests:

    objectives = [Adjiman(), BartelsConn(), Himmelblau(), Leon(), Rosenbrock(), Branin02(),
                  StyblinskiTank(), ThreeHumpCamel(), Ursem01(), Wikipedia()]

    optimizers = [GradientDescentOptimizer(), Momentum(), Nesterov(),
                  Adagrad(), Adadelta(), RMSprop(), Adam(), AdaMax(), Nadam(),
                  AMSGrad(), AdamW(), QHAdam(), QuasiHyperbolicMomentum()]

    def test_pure_optimizer_core(self):
        epochs = 500
        for objective in self.objectives:
            objective_min_norm = np.linalg.norm(objective.minimum)
            for optimizer in self.optimizers:
                est = GradientDescentPureOptimizer(epochs=epochs, 
                                                   optimizer=optimizer,
                                                   objective=objective)
                est.fit()
                bb = est.blackbox_
                solution_norm = np.linalg.norm(bb.epoch_log.get('theta')[-1])
                assert len(bb.epoch_log.get('epoch')) == epochs, "Epoch log wrong length"
                assert len(bb.epoch_log.get('theta')) == epochs, "Epoch log wrong length"
                assert len(bb.epoch_log.get('train_cost')) == epochs, "Epoch log wrong length"
                assert len(bb.epoch_log.get('learning_rate')) == epochs, "Epoch log wrong length"
                msg = "\nPoor solution for objective = {o}, optimizer = {p}\n       min_norm = {m}, solution_norm = {s}".format(o = objective.__class__.__name__,
                                                                                  p = optimizer.__class__.__name__,
                                                                                  m = str(objective_min_norm),
                                                                                  s = str(solution_norm))
                if solution_norm - objective_min_norm > 50:
                    print(msg)

    