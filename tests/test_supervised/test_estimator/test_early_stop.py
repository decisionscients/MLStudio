#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : test_early_stop.py                                                 #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 11:39:20 pm                         #
# Last Modified : Sunday, March 15th 2020, 11:39:20 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
# =========================================================================== #
#                            TEST EARLY STOP                                  #
# =========================================================================== #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark

from mlstudio.supervised.estimator.early_stop import EarlyStop
from mlstudio.supervised.estimator.scorers import RegressionScorerFactory
from mlstudio.supervised.regression import LinearRegression

# --------------------------------------------------------------------------- #
#                           TEST EARLY STOP                                   #
# --------------------------------------------------------------------------- #

class EarlyStopTests:

    @mark.early_stop
    def test_early_stop_improvement_init(self):
        stop = EarlyStop()
        assert stop.early_stop is True, "Early stop not correct"
        assert stop.precision == 0.01, "precision not correct"
        assert stop.monitor == 'val_score', "metric is initiated correctly"
        assert stop.converged_ is False, "converged is not False on instantiation"
        assert stop.best_weights_ is None, "best weights is not None on instantiation"
        

    @mark.early_stop
    def test_early_stop_improvement_validation(self):
        with pytest.raises(ValueError):
            stop = EarlyStop(monitor=9)
            stop.model = LinearRegression(metric=None)
            stop.on_train_begin()
        with pytest.raises(ValueError):
            stop = EarlyStop(monitor='x')
            stop.model = LinearRegression(metric=None)
            stop.on_train_begin()
        with pytest.raises(TypeError):
            stop = EarlyStop(precision='x')
            stop.model = LinearRegression(metric=None)
            stop.on_train_begin()              
        with pytest.raises(TypeError):
            stop = EarlyStop(precision=5)
            stop.model = LinearRegression(metric=None)
            stop.on_train_begin()
        with pytest.raises(TypeError):
            stop = EarlyStop(patience='x')
            stop.model = LinearRegression(metric=None)
            stop.on_train_begin()            
        with pytest.raises(ValueError):
            stop = EarlyStop(monitor='val_score')
            stop.model = LinearRegression(metric=None)
            stop.on_train_begin()                        

    @mark.early_stop
    def test_early_stop_improvement_on_train_begin(self, models_by_metric,
                                               early_stop_monitor):        
        # Test with score        
        stop=EarlyStop(monitor=early_stop_monitor)
        stop.model = models_by_metric
        stop.on_train_begin()
        assert stop.monitor == early_stop_monitor, "metric not set correctly" 
        if 'score' in early_stop_monitor:
            assert stop.best_performance_ == models_by_metric.scorer_.worst, "metric best_performance not set correctly"
            assert stop.precision == abs(stop.precision) * models_by_metric.scorer_.precision_factor, "precision not set correctly"
        else:
            assert stop.best_performance_ == np.Inf, "cost best_performance not set correctly"
            assert stop.precision < 0, "precision not set correctly"

    @mark.early_stop
    def test_early_stop_improvement_on_epoch_end_train_cost(self):        
        stop=EarlyStop(monitor='train_cost', precision=0.1, patience=2)
        stop.model = LinearRegression(metric=None)
        stop.on_train_begin()                
        logs = [{'train_cost': 100}, {'train_cost': 99},{'train_cost': 80},
               {'train_cost': 78},{'train_cost': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged_ == converged[i], "not converging correctly" 

    @mark.early_stop
    def test_early_stop_improvement_on_epoch_end_val_cost(self):
        stop=EarlyStop(monitor='val_cost', precision=0.1, patience=2)
        stop.model = LinearRegression(metric=None)
        stop.on_train_begin()                
        logs = [{'val_cost': 100}, {'val_cost': 99},{'val_cost': 80},
               {'val_cost': 78},{'val_cost': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged_ == converged[i], "not converging correctly"

    @mark.early_stop
    def test_early_stop_improvement_on_epoch_end_train_scores_lower_is_better(self, 
                            model_lower_is_better):
        stop=EarlyStop(monitor='train_score', precision=0.1, patience=2)
        stop.model = model_lower_is_better
        stop.on_train_begin()                
        logs = [{'train_score': 100}, {'train_score': 99},{'train_score': 80},
               {'train_score': 78},{'train_score': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged_ == converged[i], "not converging correctly"             

    @mark.early_stop
    def test_early_stop_improvement_on_epoch_end_train_scores_higher_is_better(self, 
                            model_higher_is_better):
        stop=EarlyStop(monitor='train_score', precision=0.1, patience=2)
        stop.model = model_higher_is_better
        stop.on_train_begin()             
        logs = [{'train_score': 100}, {'train_score': 101},{'train_score': 120},
               {'train_score': 122},{'train_score': 123}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged_ == converged[i], "not converging correctly"                                  
 
    @mark.early_stop
    def test_early_stop_improvement_on_epoch_end_val_scores_lower_is_better(self, 
                            model_lower_is_better):
        stop=EarlyStop(monitor='val_score', precision=0.1, patience=2)
        stop.model = model_lower_is_better
        stop.on_train_begin()                
        logs = [{'val_score': 100}, {'val_score': 99},{'val_score': 80},
               {'val_score': 78},{'val_score': 77}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])
            assert stop.converged_ == converged[i], "not converging correctly"             
 
    @mark.early_stop
    def test_early_stop_improvement_on_epoch_end_val_scores_higher_is_better(self, 
                            model_higher_is_better):
        stop=EarlyStop(precision=0.1, patience=2)
        stop.model = model_higher_is_better
        stop.on_train_begin()             
        logs = [{'val_score': 100}, {'val_score': 101},{'val_score': 120},
               {'val_score': 122},{'val_score': 123}]
        converged = [False, False, False, False, True]
        for i in range(len(logs)):
            stop.on_epoch_end(epoch=i+1, logs=logs[i])            
            assert stop.converged_ == converged[i], "not converging correctly"                      
         
