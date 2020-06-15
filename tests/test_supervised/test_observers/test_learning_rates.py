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

from mlstudio.supervised.observers.learning_rate import StepDecay, TimeDecay
from mlstudio.supervised.observers.learning_rate import SqrtTimeDecay
from mlstudio.supervised.observers.learning_rate import ExponentialDecay, PolynomialDecay
from mlstudio.supervised.observers.learning_rate import PolynomialStepDecay
from mlstudio.supervised.observers.learning_rate import ExponentialStepDecay, PowerSchedule
from mlstudio.supervised.observers.learning_rate import BottouSchedule, Improvement

@mark.observer
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

    def test_step_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_step_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = StepDecay(initial_learning_rate=0.1, min_learning_rate=0.01,
                        decay_factor=0.5, decay_steps=5)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg

@mark.observer
@mark.lrs
@mark.time_decay
class TimeDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_time_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = TimeDecay(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(initial_learning_rate=0, decay_factor=0.5)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(initial_learning_rate=1, decay_factor=0.5)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = TimeDecay(min_learning_rate=None, decay_factor=0.5)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(min_learning_rate=0, decay_factor=0.5)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(min_learning_rate=1, decay_factor=0.5)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = TimeDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = TimeDecay(decay_factor=2)
            lrs.on_train_begin()       
    

    def test_time_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_time_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = TimeDecay(initial_learning_rate=0.1, min_learning_rate=0.02,
                        decay_factor=0.5)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg

@mark.observer
@mark.lrs
@mark.sqrt_time_decay
class SqrtTimeDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_sqrt_time_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = SqrtTimeDecay(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(initial_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(initial_learning_rate=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = SqrtTimeDecay(min_learning_rate=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(min_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(min_learning_rate=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = SqrtTimeDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = SqrtTimeDecay(decay_factor=2)
            lrs.on_train_begin()       
    

    def test_sqrt_time_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_sqrt_time_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = SqrtTimeDecay(initial_learning_rate=0.1, min_learning_rate=0.045,
                        decay_factor=0.5)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg    

@mark.observer
@mark.lrs
@mark.exponential_decay
class ExponentialDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_exponential_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = ExponentialDecay(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(initial_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(initial_learning_rate=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = ExponentialDecay(min_learning_rate=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(min_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(min_learning_rate=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = ExponentialDecay(decay_factor=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialDecay(decay_factor=2)
            lrs.on_train_begin()       
    
    def test_exponential_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_exponential_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = ExponentialDecay(initial_learning_rate=0.1, min_learning_rate=0.05,
                        decay_factor=0.1)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg                

@mark.observer
@mark.lrs
@mark.exponential_step_decay
class ExponentialStepDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_exponential_step_decay_validation(self, get_mock_estimator):
        est = get_mock_estimator
        # Validate initial learning rate
        with pytest.raises(TypeError):            
            lrs = ExponentialStepDecay(initial_learning_rate='h')
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(initial_learning_rate=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(initial_learning_rate=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = ExponentialStepDecay(min_learning_rate=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(min_learning_rate=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(min_learning_rate=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = ExponentialStepDecay(decay_factor=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(decay_factor=-1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(decay_factor=2)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()       
        # Validate decay_steps
        with pytest.raises(TypeError):
            lrs = ExponentialStepDecay(decay_steps=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(decay_steps=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin() 
    
    def test_exponential_step_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_exponential_step_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = ExponentialStepDecay(initial_learning_rate=0.1, min_learning_rate=0.01,
                        decay_factor=0.5, decay_steps=5)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg          

@mark.observer
@mark.lrs
@mark.exponential_step_decay_staircase
class ExponentialStepDecayStaircaseTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_exponential_step_decay_staircase_validation(self, get_mock_estimator):
        est = get_mock_estimator
        # Validate initial learning rate
        with pytest.raises(TypeError):            
            lrs = ExponentialStepDecay(initial_learning_rate='h')
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(initial_learning_rate=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(initial_learning_rate=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = ExponentialStepDecay(min_learning_rate=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(min_learning_rate=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(min_learning_rate=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = ExponentialStepDecay(decay_factor=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(decay_factor=-1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(decay_factor=2)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()       
        # Validate decay_staircase_steps
        with pytest.raises(TypeError):
            lrs = ExponentialStepDecay(decay_steps=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = ExponentialStepDecay(decay_steps=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin() 
    
    def test_exponential_step_decay_staircase(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_exponential_step_decay_staircase.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = ExponentialStepDecay(initial_learning_rate=0.1, min_learning_rate=0.02,
                        decay_factor=0.5, decay_steps=5, staircase=True)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg          

@mark.observer
@mark.lrs
@mark.polynomial_decay
class PolynomialDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_polynomial_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = PolynomialDecay(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(initial_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(initial_learning_rate=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = PolynomialDecay(min_learning_rate=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(min_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(min_learning_rate=1)
            lrs.on_train_begin()      
        # Validate power
        with pytest.raises(TypeError):
            lrs = PolynomialDecay(power=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(power=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialDecay(power=2)
            lrs.on_train_begin()       
    
    def test_polynomial_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_polynomial_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = PolynomialDecay(initial_learning_rate=0.1, min_learning_rate=0.025,
                        power=0.95)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg           

@mark.observer
@mark.lrs
@mark.polynomial_step_decay
class PolynomialStepDecayTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_polynomial_step_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = PolynomialStepDecay(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialStepDecay(initial_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialStepDecay(initial_learning_rate=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = PolynomialStepDecay(min_learning_rate=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialStepDecay(min_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialStepDecay(min_learning_rate=1)
            lrs.on_train_begin()      
        # Validate power
        with pytest.raises(TypeError):
            lrs = PolynomialStepDecay(power=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialStepDecay(power=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialStepDecay(power=2)
            lrs.on_train_begin()   
        # Validate decay_steps
        with pytest.raises(TypeError):
            lrs = PolynomialStepDecay(decay_steps=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PolynomialStepDecay(decay_steps=0)
            lrs.on_train_begin()                 
    
    def test_polynomial_step_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_polynomial_step_decay.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = PolynomialStepDecay(initial_learning_rate=0.1, min_learning_rate=0.001,
                        power=0.5, decay_steps=5)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg         

@mark.observer
@mark.lrs
@mark.power_decay
class PowerScheduleTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_power_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = PowerSchedule(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(initial_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(initial_learning_rate=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = PowerSchedule(min_learning_rate=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(min_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(min_learning_rate=1)
            lrs.on_train_begin()      
        # Validate power
        with pytest.raises(TypeError):
            lrs = PowerSchedule(power=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(power=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(power=2)
            lrs.on_train_begin()       
        # Validate decay_steps
        with pytest.raises(TypeError):
            lrs = PowerSchedule(decay_steps=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = PowerSchedule(decay_steps=0)
            lrs.on_train_begin()                
    
    def test_power_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_power.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = PowerSchedule(initial_learning_rate=0.1, min_learning_rate=0.04,
                        power=0.95, decay_steps=5)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg       

@mark.observer
@mark.lrs
@mark.bottou_decay
class BottouScheduleTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['lr'],
                                    usecols="B").to_numpy().flatten()

    def test_bottou_decay_validation(self):
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = BottouSchedule(initial_learning_rate='h')
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(initial_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(initial_learning_rate=1)
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = BottouSchedule(min_learning_rate=None)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(min_learning_rate=0)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(min_learning_rate=1)
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = BottouSchedule(decay_factor=None)            
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(decay_factor=-1)
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = BottouSchedule(decay_factor=2)
            lrs.on_train_begin()       
            
    
    def test_bottou_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_bottou.xlsx")
        exp_results = self._get_expected_results(filepath)
        # Instantiate learning rate schedule and create it as an observer
        lrs = BottouSchedule(initial_learning_rate=0.1, min_learning_rate=0.07,
                        decay_factor=0.5)
        observer = {'learning_rate_schedule': lrs}              
        # Instantiate and fit mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)
        est.fit()
        # Extract learning rate history
        epochs = est.blackbox_.epoch_log.get('epoch')
        act_results = est.blackbox_.epoch_log.get('learning_rate')
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg                

@mark.observer
@mark.lrs
@mark.improvement_decay
class ImprovementTests:

    def _get_expected_results(self, filepath):
        return pd.read_excel(filepath, sheet_name='results', header=0, names=['cost','lr'],
                                    usecols="C,H")

    def test_improvement_decay_validation(self, get_mock_estimator):
        est = get_mock_estimator        
        # Validate initial learning rate
        with pytest.raises(TypeError):
            lrs = Improvement(initial_learning_rate='h')
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Improvement(initial_learning_rate=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Improvement(initial_learning_rate=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()            
        # Validate minimum learning rate
        with pytest.raises(TypeError):
            lrs = Improvement(min_learning_rate=None)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Improvement(min_learning_rate=0)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Improvement(min_learning_rate=1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()      
        # Validate decay_factor
        with pytest.raises(TypeError):
            lrs = Improvement(decay_factor=None)            
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Improvement(decay_factor=-1)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()
        with pytest.raises(ValueError):
            lrs = Improvement(decay_factor=2)
            lrs.set_model(est(epochs=10))
            lrs.on_train_begin()   
    
    def test_improvement_decay(self, get_mock_estimator):       
        Estimator = get_mock_estimator
        # Obtain expected results
        filepath = os.path.join(testdatadir, "test_learning_rate_schedules_improvement.xlsx")
        exp_results = self._get_expected_results(filepath)
        cost = exp_results['cost'].values
        exp_results = exp_results['lr'].values
        # Instantiate learning rate schedule and create it as an observer
        lrs = Improvement(initial_learning_rate=0.1, min_learning_rate=0.002,
                        decay_factor=0.5, epsilon=0.01, patience=2)
        observer = {'learning_rate_schedule': lrs}                  
        # Instantiate mock estimator
        est = Estimator(epochs=10, learning_rate=0.1, observers=observer)                                
        lrs.set_model(est)
        # Simulate training 
        lrs.on_train_begin()
        act_results = []
        lr = 0.1
        for i in range(10):
            log = {'train_cost':cost[i], 'learning_rate': lrs.model.eta}    
            lrs.on_epoch_end(epoch=i, log=log)    
            act_results.append(lrs.model.eta)
        # Compare two arrays
        act_res_len = len(act_results)
        exp_res_len = len(exp_results)
        msg = "Expected results length = {e}, actual results length = {a}".format(e=str(exp_res_len),
                                                                                  a=str(act_res_len))
        assert act_res_len == exp_res_len, msg
        msg = "Expected results {e}\nActual Results {a}".format(e=str(exp_results),a=str(act_results))
        assert np.allclose(exp_results, act_results), msg         