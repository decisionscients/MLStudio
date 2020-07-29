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
from mlstudio.supervised.algorithms.optimization.services import tasks
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.supervised.performance import regression
# --------------------------------------------------------------------------  #
@mark.regressor
class GradientRegressorTests:

    def test_regressor_instantiation(self):        
        est = algorithms.GradientDescent.regression()       
        assert isinstance(est.task, tasks.LinearRegression), "Error GradientRegressorTest: task is invalid."        
        assert isinstance(est.optimizer, optimizers.GradientDescentOptimizer), "Error GradientRegressorTest: optimizer is invalid."        
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

    def test_regressor_fit(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split
        est = algorithms.GradientDescent.regression(check_gradient=False, epochs=5000)        
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
        assert bb.total_epochs == 5000, "Regressor: blackbox error, num epochs incorrect"
        assert bb.total_batches == 5000, "Regressor: blackbox error, num batches incorrect"
        
        assert len(bb.epoch_log.get('train_cost')) == 5000, "Regressor: blackbox error, train_cost length != num epochs"
        assert len(bb.epoch_log.get('val_cost')) == 5000, "Regressor: blackbox error, val_cost length != num epochs"
        assert len(bb.epoch_log.get('train_score')) == 5000, "Regressor: blackbox error, train_score length != num epochs"
        assert len(bb.epoch_log.get('val_score')) == 5000, "Regressor: blackbox error, val_score length != num epochs"        
        assert len(bb.epoch_log.get('theta')) == 5000, "Regressor: blackbox error, theta shape incorrect"
        assert len(bb.epoch_log.get('gradient')) == 5000, "Regressor: blackbox error, gradient shape incorrect"
        assert len(bb.epoch_log.get('gradient_norm')) == 5000, "Regressor: blackbox error, gradient_norm length != num epochs"        
        assert len(bb.epoch_log.get('cpu_time')) == 5000, "Regressor: blackbox error, gradient_norm length != num epochs"        
        assert len(bb.epoch_log.get('current_memory')) == 5000, "Regressor: blackbox error, gradient_norm length != num epochs"        
        assert len(bb.epoch_log.get('peak_memory')) == 5000, "Regressor: blackbox error, gradient_norm length != num epochs"        
        skl_scores = skl.score(X_train, y_train)
        est_scores = est.score(X_train, y_train)
        d = (skl_scores-est_scores)/skl_scores
        print("\nSKL {k}     MLS {m}     RDIF {d}".format(k=str(skl_scores), m=str(est_scores), d=str(d)))
        assert np.isclose(skl.score(X_train,y_train), est.score(X_train, y_train), rtol=1e-2), "Regressor: inaccurate train scores "
        assert np.isclose(skl.score(X_test,y_test), est.score(X_test, y_test), rtol=1e-2), "Regressor: inaccurate test scores "
        # Test summary
        est.summarize()




        
