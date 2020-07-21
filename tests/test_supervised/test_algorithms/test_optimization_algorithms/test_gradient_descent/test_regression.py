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
from mlstudio.supervised.algorithms.optimization.observers import base, debug
from mlstudio.supervised.algorithms.optimization.observers import monitor
from mlstudio.supervised.algorithms.optimization.services import loss, tasks
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.supervised.metrics import regression, classification
# --------------------------------------------------------------------------  #
@mark.regressor
class GradientRegressorTests:

    def test_regressor_instantiation(self):        
        est = algorithms.GradientDescent.regressor_factory()       
        assert isinstance(est.task, tasks.LinearRegression), "Error GradientRegressorTest: task is invalid."        
        assert isinstance(est.optimizer, optimizers.GradientDescentOptimizer), "Error GradientRegressorTest: optimizer is invalid."        
        assert isinstance(est.scorer, regression.RegressionScorer), "Error GradientRegressorTest: scorer is invalid."        
        assert isinstance(est.observer_list, base.ObserverList), "Error GradientRegressorTest: observer_list is invalid."        
        assert isinstance(est.progress, monitor.Progress), "Error GradientRegressorTest: progress is invalid."        
        assert isinstance(est.blackbox, monitor.BlackBox), "Error GradientRegressorTest: blackbox is invalid."
        assert isinstance(est.summary, monitor.Summary), "Error GradientRegressorTest: summary is invalid."
        assert est.eta0 == 0.01, "Error GradientRegressorTest: eta0 is invalid."
        assert est.epochs == 1000, "Error GradientRegressorTest: epochs is invalid."
        assert est.batch_size is None, "Error GradientRegressorTest: batch_size is invalid."
        assert est.val_size == 0.3, "Error GradientRegressorTest: val_size is invalid."
        assert est.theta_init is None, "Error GradientRegressorTest: theta_init is invalid."
        assert est.verbose is False, "Error GradientRegressorTest: verbose is invalid."
        assert est.random_state is None, "Error GradientRegressorTest: random_state is invalid."

    def test_regressor_fit(self, get_regression_data_split):
        X_train, X_test, y_train, y_test = get_regression_data_split
        est = algorithms.GradientDescent.regressor_factory(check_gradient=True)        
        est.fit(X_train, y_train)
        skl = SGDRegressor()
        skl.fit(X_train, y_train)
        # Test data split
        m = X_train.shape[0]
        assert np.isclose(est.X_train_.shape[0], (0.7 * m), rtol=1), "Regressor: training data shape incorrect."
        assert np.isclose(est.X_val_.shape[0], (0.3 * m), rtol=1), "Regressor: validation data shape incorrect."
        assert est.X_train_.shape[0] == len(est.y_train_), "Regressor: X_train,y_train shape mismatch "
        assert est.X_val_.shape[0] == len(est.y_val_), "Regressor: X_val,y_val shape mismatch "
        # Test blackbox
        bb = est.blackbox_
        assert bb.total_epochs == 1000, "Regressor: blackbox error, num epochs incorrect"
        assert bb.total_batches == 1000, "Regressor: blackbox error, num batches incorrect"
        assert len(bb.epoch_log.get('train_cost')) == 1000, "Regressor: blackbox error, train_cost length != num epochs"
        assert len(bb.epoch_log.get('val_cost')) == 1000, "Regressor: blackbox error, val_cost length != num epochs"
        assert len(bb.epoch_log.get('train_score')) == 1000, "Regressor: blackbox error, train_score length != num epochs"
        assert len(bb.epoch_log.get('val_score')) == 1000, "Regressor: blackbox error, val_score length != num epochs"        
        assert len(bb.epoch_log.get('theta')) == 1000, "Regressor: blackbox error, theta shape incorrect"
        assert len(bb.epoch_log.get('gradient')) == 1000, "Regressor: blackbox error, gradient shape incorrect"
        assert len(bb.epoch_log.get('gradient_norm')) == 1000, "Regressor: blackbox error, gradient_norm length != num epochs"        
        # Test scorer
        assert isinstance(est.scorer_, regression.R2), "Regressor: scorer incorrect."
        skl_scores = skl.score(X_train, y_train)
        est_scores = est.score(X_train, y_train)
        d = (skl_scores-est_scores)/skl_scores
        print("SKL {k}     MLS {m}     RDIF {d}".format(k=str(skl_scores), m=str(est_scores), d=str(d)))
        assert np.allclose(skl.score(X_train,y_train), est.score(X_train, y_train), rtol=1e-3), "Regressor: inaccurate train scores "
        assert np.allclose(skl.score(X_test,y_test), est.score(X_test, y_test), rtol=1e-3), "Regressor: inaccurate test scores "
        # Test summary
        est.summary()
        # Test results
        assert np.isclose(skl.intercept_, est.intercept_), "Regressor: intercept inaccurate"
        assert np.allclose(skl.coef_, est.coef_), "Regressor: coeficients inaccurate"




        
