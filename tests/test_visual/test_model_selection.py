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

from mlstudio.supervised.regression import LinearRegression
from mlstudio.visual.model_selection import CostCurve

class CostCurveTests:

    @mark.cost_curve
    def test_cost_curve_basic(self, get_regression_data):
        path = "./tests/test_visual/figures/basic_cost_curve.png"
        X, y = get_regression_data
        est = LinearRegression()
        cc = CostCurve(est, title='Basic Cost Curve')
        cc.fit(X, y)
        cc.show()
        cc.save(filepath=path)
        assert os.path.exists(path), "Figure not saved."

    @mark.cost_curve
    def test_cost_curve_color(self, get_regression_data):
        path = "./tests/test_visual/figures/learning_rate_cost_curve.png"
        param_grid = {'learning_rate' : np.logspace(-3,-1, num=10).tolist()}
        X, y = get_regression_data
        est = LinearRegression()
        cc = CostCurve(est, title='Learning Rate Cost Curve')
        cc.fit(X, y, param_grid=param_grid, color='learning_rate')
        cc.show()
        cc.save(filepath=path)
        assert os.path.exists(path), "Figure not saved."        

    @mark.cost_curve
    def test_cost_curve_color_facets(self, get_regression_data):
        path = "./tests/test_visual/figures/learning_rate_cost_curve.png"
        param_grid = {'learning_rate' : np.logspace(-3,-1, num=10).tolist(),
                      'batch_size': [32,64,128,256,512]}
        X, y = get_regression_data
        est = LinearRegression()
        cc = CostCurve(est, title='Learning Rate Cost Curve by Batch Size')
        cc.fit(X, y, param_grid=param_grid, color='learning_rate',
               facet_col='batch_size')
        cc.show()
        cc.save(filepath=path)
        assert os.path.exists(path), "Figure not saved."        

class LearningCurveTests:

    @mark.learning_curve   
    def test_learning_curve(self, get_regression_data):
        X, y = get_regression_data
        est = LinearRegression()
        title = "Learning Curve: " + est.name
        lc = LearningCurve(est, title=title)
        lc.fit(X,y)
        lc.show()
