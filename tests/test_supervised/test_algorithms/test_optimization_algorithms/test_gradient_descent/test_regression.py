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

from mlstudio.factories import tasks, observers, estimators
from mlstudio.supervised.algorithms.optimization.gradient_descent import GradientDescent
from mlstudio.supervised.algorithms.optimization.observers import base
from mlstudio.supervised.algorithms.optimization.observers import monitor
from mlstudio.supervised.algorithms.optimization.services import loss, tasks
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.supervised.metrics import regression, classification
# --------------------------------------------------------------------------  #
@mark.regressor
class GradientRegressorTests:

    def test_gradient_regressor(self, get_regression_data):
        X, y = None
        est = estimators.GradientDescent.regression_factory()       
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

        with pytest.raises(ValueError):        
            est.fit(X,y)



        
