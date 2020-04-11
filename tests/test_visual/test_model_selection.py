#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : test_cost_plot.py                                                  #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Wednesday, March 18th 2020, 3:03:19 am                       #
# Last Modified : Wednesday, March 18th 2020, 3:03:19 am                       #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
import os
import numpy as np
from pytest import mark
import shutil
from sklearn.model_selection import ShuffleSplit

from mlstudio.supervised.estimator.gradient import GradientDescentRegressor
from mlstudio.supervised.estimator.scorers import R2, RMSE
from mlstudio.supervised.regression import LinearRegression, LassoRegression
from mlstudio.supervised.regression import RidgeRegression, ElasticNetRegression
from mlstudio.visual.model_selection import CostCurve, LearningCurve
from mlstudio.visual.model_selection import ModelScalability
from mlstudio.visual.model_selection import ModelLearningPerformance
from mlstudio.visual.model_selection import LearningCurves, ValidationCurve

@mark.plots
@mark.cost_curve
@mark.skip(reason="visually tested")
class CostCurveTests:

    @mark.cost_curve
    def test_cost_curve_basic(self, get_regression_data):
        path = "./tests/test_visual/figures/basic_cost_curve.png"
        X, y = get_regression_data
        est = GradientDescentRegressor(algorithm=LinearRegression())
        cc = CostCurve(est)
        cc.fit(X, y)
        cc.show()

    @mark.cost_curve
    def test_cost_curve_color(self, get_regression_data):
        path = "./tests/test_visual/figures/learning_rate_cost_curve.png"
        param_grid = {'learning_rate' : np.logspace(-3,-1, num=10).tolist()}
        X, y = get_regression_data
        est = GradientDescentRegressor(algorithm=LinearRegression())
        cc = CostCurve(est, param_grid=param_grid, factor='learning_rate')
        cc.fit(X, y)
        cc.show()

    @mark.cost_curve
    def test_cost_curve_color_facets(self, get_regression_data):
        path = "./tests/test_visual/figures/learning_rate_cost_curve_w_facets.png"
        param_grid = {'learning_rate' : np.logspace(-3,-1, num=10).tolist(),
                      'batch_size': [32,64,128,256,512]}
        X, y = get_regression_data
        est = GradientDescentRegressor(algorithm=LinearRegression())
        cc = CostCurve(est, param_grid=param_grid, factor='learning_rate',
               facet_col='batch_size')
        cc.fit(X, y)
        cc.show()


@mark.plots
@mark.learning_curve
@mark.skip(reason="visually tested")
class LearningCurveTests:

    @mark.learning_curve   
    def test_learning_curve(self, get_regression_data):
        X, y = get_regression_data
        est = est = GradientDescentRegressor(algorithm=LinearRegression(), scorer=RMSE())
        lc = LearningCurve(est)
        lc.fit(X,y)
        lc.show()

    @mark.model_scalability   
    def test_model_scalability_plot(self, get_regression_data):
        X, y = get_regression_data        
        est = GradientDescentRegressor(algorithm=LassoRegression(), scorer=R2())
        ms = ModelScalability(est)
        ms.fit(X,y)
        ms.show()        

    @mark.model_learning_performance   
    def test_model_learning_performance_plot(self, get_regression_data):
        X, y = get_regression_data        
        est = GradientDescentRegressor(algorithm=RidgeRegression())
        ms = ModelLearningPerformance(est)
        ms.fit(X,y)
        ms.show()        

    @mark.learning_curves
    def test_multiple_learning_curve_plots(self, get_regression_data):
        X, y = get_regression_data        
        est1 = est = GradientDescentRegressor(algorithm=LinearRegression())
        est2 = est = GradientDescentRegressor(algorithm=ElasticNetRegression())
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        est = [est1, est2]
        lcp = LearningCurves(est, cv=cv)
        lcp.fit(X,y)
        lcp.show()

@mark.plots
@mark.validation_curve
@mark.skip(reason="visually tested")
class ValidationCurveTests:

    @mark.validation_curve   
    def test_validation_curve(self, get_regression_data):
        X, y = get_regression_data
        est = est = GradientDescentRegressor(algorithm=ElasticNetRegression())
        param_name = 'learning_rate'
        param_range = np.logspace(-3,-1, 10)
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        vc = ValidationCurve(est, param_name=param_name, param_range=param_range, cv=cv)        
        vc.fit(X,y)
        vc.show()