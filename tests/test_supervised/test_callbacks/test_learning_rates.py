#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_learning_rates.py                                            #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, May 17th 2020, 2:27:44 am                           #
# Last Modified : Sunday, May 17th 2020, 2:34:01 am                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests learning rate classes."""
#%%
import math
import os
from pathlib import Path
import sys
testdir = str(Path(__file__).parents[2])
testdatadir = os.path.join(testdir, 'test_data')
sys.path.append(testdatadir)

import pandas as pd
import pytest
from pytest import mark
import numpy as np

from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentRegressor
from mlstudio.supervised.callbacks.learning_rate import StepDecay, TimeDecay
from mlstudio.supervised.callbacks.learning_rate import SqrtTimeDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.callbacks.learning_rate import ExponentialStepDecay, PowerSchedule
from mlstudio.supervised.callbacks.learning_rate import BottouSchedule, Improvement

@mark.callback
@mark.lrs
@mark.step_decay
class StepDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_step_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = StepDecay(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(initial_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(initial_learning_rate=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = StepDecay(min_learning_rate=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(min_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(min_learning_rate=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = StepDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(decay_factor=2)
            lrs.on_train_begin()       
        # Validate decay_steps
        with pytest.raises(TypeError):
            lrs = StepDecay(decay_steps=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = StepDecay(decay_steps=-1)
            lrs.on_train_begin()
        with pytest.raises(TypeError):
            lrs = StepDecay(decay_steps=np.inf)
            lrs.on_train_begin()      

    def test_step_decay(self, get_regression_data):       
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_step_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule
        lrs = StepDecay(initial_learning_rate=0.1, min_learning_rate=0.01,
                        decay_factor=0.5, decay_steps=5)              
        # Create estimator
        est = GradientDescentRegressor(epochs=10, learning_rate=lrs)
        # Obtain the data and fit the model
        X, y = get_regression_data
        est.fit(X, y)
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        print(epochs)
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg

