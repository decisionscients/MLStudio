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
"""Integration test for GDRegressor class."""
#%%
import numpy as np
import pytest
from pytest import mark
from sklearn.linear_model import SGDRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator
from sklearn.linear_model import SGDRegressor, SGDClassifier

from mlstudio.factories import tasks, observers, algorithms
from mlstudio.supervised.algorithms.optimization.observers import base
from mlstudio.supervised.algorithms.optimization.observers import history
from mlstudio.supervised.algorithms.optimization.observers import report
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.supervised.algorithms.optimization.observers import learning_rate
from mlstudio.supervised import metrics 
# --------------------------------------------------------------------------  #

@mark.regressor
class GradientRegressorTests:

    def test_regressor_instantiation(self):        
        est = algorithms.GDRegressor.base()       
        assert isinstance(est.task, tasks.LinearRegression), "Error GradientRegressorTest: task is invalid."        
        assert isinstance(est.optimizer, optimizers.GradientDescentOptimizer), "Error GradientRegressorTest: optimizer is invalid."        
        assert isinstance(est.scorer, (metrics.base.BaseRegressionMetric,
                                       metrics.base.BaseRegressionMetric)), "Error GradientRegressorTest: scorer is invalid."        
        assert isinstance(est.observer_list, base.ObserverList), "Error GradientRegressorTest: observer_list is invalid."        
        assert isinstance(est.progress, report.Progress), "Error GradientRegressorTest: progress is invalid."        
        assert isinstance(est.blackbox, history.BlackBox), "Error GradientRegressorTest: blackbox is invalid."
        assert isinstance(est.summary, report.Summary), "Error GradientRegressorTest: summary is invalid."
        assert est.eta0 == 0.01, "Error GradientRegressorTest: eta0 is invalid."
        assert est.epochs == 1000, "Error GradientRegressorTest: epochs is invalid."
        assert est.batch_size is None, "Error GradientRegressorTest: batch_size is invalid."
        assert est.val_size == 0.3, "Error GradientRegressorTest: val_size is invalid."
        assert est.theta_init is None, "Error GradientRegressorTest: theta_init is invalid."
        assert est.verbose is False, "Error GradientRegressorTest: verbose is invalid."
        assert est.random_state is None, "Error GradientRegressorTest: random_state is invalid."

    # Test learning rates
    learning_rates = \
        [learning_rate.TimeDecay(eta0=0.5), learning_rate.StepDecay(eta0=0.5), 
         learning_rate.SqrtTimeDecay(eta0=0.1),learning_rate.ExponentialDecay(eta0=0.1), 
         learning_rate.ExponentialSchedule(eta0=0.1), learning_rate.PolynomialDecay(eta0=0.1), 
         learning_rate.PowerSchedule(eta0=0.1), learning_rate.BottouSchedule(eta0=0.1), 
         learning_rate.Adaptive(eta0=0.1)]

    scenarios = [[learning_rate] for learning_rate in learning_rates]         
    
    lr_scenarios = [algorithms.GDRegressor.base(learning_rate=scenario[0],
                                                        epochs=1000) 
                                                        for scenario in scenarios]
    @mark.regressor_learning_rates                                                            
    def test_regressor_learning_rates(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split        
        for estimator in GradientRegressorTests.lr_scenarios:            
            est = estimator
            est.fit(X_train, y_train)
            skl = SGDRegressor()
            skl.fit(X_train, y_train)
            # Test data split
            m = X_train.shape[0]
            assert np.isclose(est.X_train.shape[0], (0.7 * m), rtol=1), "Regressor: training data shape incorrect."
            assert np.isclose(est.X_val.shape[0], (0.3 * m), rtol=1), "Regressor: validation data shape incorrect."
            assert est.X_train.shape[0] == len(est.y_train), "Regressor: X_train,y_train shape mismatch "
            assert est.X_val.shape[0] == len(est.y_val), "Regressor: X_val,y_val shape mismatch "
            # Test blackbox
            bb = est.get_blackbox()
            assert bb.total_epochs == 1000, "Regressor: blackbox error, num epochs incorrect"
            assert bb.total_batches == 1000, "Regressor: blackbox error, num batches incorrect"
            
            assert len(bb.epoch_log.get('train_cost')) == 1000, "Regressor: blackbox error, train_cost length != num epochs"
            assert len(bb.epoch_log.get('val_cost')) == 1000, "Regressor: blackbox error, val_cost length != num epochs"
            assert len(bb.epoch_log.get('train_score')) == 1000, "Regressor: blackbox error, train_score length != num epochs"
            assert len(bb.epoch_log.get('val_score')) == 1000, "Regressor: blackbox error, val_score length != num epochs"        
            assert len(bb.epoch_log.get('theta')) == 1000, "Regressor: blackbox error, theta shape incorrect"
            assert len(bb.epoch_log.get('gradient')) == 1000, "Regressor: blackbox error, gradient shape incorrect"
            assert len(bb.epoch_log.get('gradient_norm')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('cpu_time')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('current_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('peak_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            # Evaluate results vis-a-vis sklearn
            skl_scores = skl.score(X_train, y_train)
            est_scores = est.score(X_train, y_train)
            msg = est.learning_rate.name
            d = (skl_scores-est_scores)/skl_scores
            print("\nSKL {k}     MLS {m}     RDIF {d}".format(k=str(skl_scores), m=str(est_scores), d=str(d)))
            est.summarize()
            assert np.isclose(skl.score(X_train,y_train), est.score(X_train, y_train), rtol=1e-1), "Regressor: inaccurate train scores " + msg
            assert np.isclose(skl.score(X_test,y_test), est.score(X_test, y_test), rtol=1e-1), "Regressor: inaccurate test scores " + msg
            # Test scorer
            est.set_scorer(metrics.regression.AdjustedR2())
            est_scores_ar2 = est.score(X_train, y_train)
            assert est_scores != est_scores_ar2, "Regressor: ar2 score not different than r2"
            assert isinstance(est_scores_ar2, float), "Regressor: ar2 score not float"
            est.set_scorer(metrics.regression.MeanAbsolutePercentageError())
            est_scores_mape =  est.score(X_train, y_train)
            assert est_scores_ar2 != est_scores_mape, "Regressor: ar2 score not different than mape"
            assert isinstance(est_scores_mape, float), "Regressor: mape score not float"
            # Test summary
            #est.summarize()
    
    regularizers = \
        [algorithms.GDRegressor.base,
         algorithms.GDRegressor.lasso().task.loss.regularizer(alpha=0.4),
         algorithms.GDRegressor.ridge().task.loss.regularizer(alpha=0.4),
         algorithms.GDRegressor.elasticnet().task.loss.regularizer(alpha=0.4, ratio=0.4)]
    # print(RegressionTasks.lasso().loss.regularizer(alpha=0.4))
    # Test regularizers
    reg_scenarios = [regularizer for regularizer in regularizers]                                                        
    @mark.regressor_regularizers                                                            
    def test_regressor_regularizers(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split        
        for estimator in GradientRegressorTests.reg_scenarios:            
            est = estimator
            est.fit(X_train, y_train)
            skl = SGDRegressor()
            skl.fit(X_train, y_train)
            # Test data split
            m = X_train.shape[0]
            assert np.isclose(est.X_train.shape[0], (0.7 * m), rtol=1), "Regressor: training data shape incorrect."
            assert np.isclose(est.X_val.shape[0], (0.3 * m), rtol=1), "Regressor: validation data shape incorrect."
            assert est.X_train.shape[0] == len(est.y_train), "Regressor: X_train,y_train shape mismatch "
            assert est.X_val.shape[0] == len(est.y_val), "Regressor: X_val,y_val shape mismatch "
            # Test blackbox
            bb = est.get_blackbox()
            assert bb.total_epochs == 1000, "Regressor: blackbox error, num epochs incorrect"
            assert bb.total_batches == 1000, "Regressor: blackbox error, num batches incorrect"
            
            assert len(bb.epoch_log.get('train_cost')) == 1000, "Regressor: blackbox error, train_cost length != num epochs"
            assert len(bb.epoch_log.get('val_cost')) == 1000, "Regressor: blackbox error, val_cost length != num epochs"
            assert len(bb.epoch_log.get('train_score')) == 1000, "Regressor: blackbox error, train_score length != num epochs"
            assert len(bb.epoch_log.get('val_score')) == 1000, "Regressor: blackbox error, val_score length != num epochs"        
            assert len(bb.epoch_log.get('theta')) == 1000, "Regressor: blackbox error, theta shape incorrect"
            assert len(bb.epoch_log.get('gradient')) == 1000, "Regressor: blackbox error, gradient shape incorrect"
            assert len(bb.epoch_log.get('gradient_norm')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('cpu_time')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('current_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            assert len(bb.epoch_log.get('peak_memory')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
            # Evaluate results vis-a-vis sklearn
            skl_scores = skl.score(X_train, y_train)
            est_scores = est.score(X_train, y_train)            
            msg = est.task.loss.regularizer.name if est.task.loss.regularizer else "No Regularizer"
            d = (skl_scores-est_scores)/skl_scores
            print("\nSKL {k}     MLS {m}     RDIF {d}".format(k=str(skl_scores), m=str(est_scores), d=str(d)))
            est.summarize()
            assert np.isclose(skl.score(X_train,y_train), est.score(X_train, y_train), rtol=1e-1), "Regressor: inaccurate train scores " + msg
            assert np.isclose(skl.score(X_test,y_test), est.score(X_test, y_test), rtol=1e-1), "Regressor: inaccurate test scores " + msg
            # Test scorer
            est.set_scorer(metrics.regression.AdjustedR2())
            est_scores_ar2 = est.score(X_train, y_train)
            assert est_scores != est_scores_ar2, "Regressor: ar2 score not different than r2"
            assert isinstance(est_scores_ar2, float), "Regressor: ar2 score not float"
            est.set_scorer(metrics.regression.MeanAbsolutePercentageError())
            est_scores_mape =  est.score(X_train, y_train)
            assert est_scores_ar2 != est_scores_mape, "Regressor: ar2 score not different than mape"
            assert isinstance(est_scores_mape, float), "Regressor: mape score not float"
            # Test summary
            #est.summarize()