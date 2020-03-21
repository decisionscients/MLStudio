#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : test_model_validation.py                                          #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, March 19th 2020, 7:59:39 pm                       #
# Last Modified : Thursday, March 19th 2020, 7:59:59 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
""" Test Model Validation Visuals.  """
import os
import numpy as np
from pytest import mark
import shutil
from sklearn.model_selection import ShuffleSplit

from mlstudio.supervised.regression import LinearRegression, LassoRegression
from mlstudio.supervised.regression import RidgeRegression, ElasticNetRegression
from mlstudio.visual.model_validation import Residuals, StandardizedResiduals
from mlstudio.visual.model_validation import StudentizedResiduals, QQPlot
from mlstudio.visual.model_validation import ResidualsLeverage
from mlstudio.visual.model_validation import ScaleLocation, CooksDistance


class ResidualPlotTests:

    
    def test_residuals(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=500)
        res = Residuals(est)
        res.fit(X,y)
        res.show()

    
    def test_residuals_wo_histogram(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=500)
        res = Residuals(est, hist=False)
        res.fit(X,y)
        res.show()        

    
    def test_residuals_w_density(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=500)
        res = Residuals(est, hist='density')
        res.fit(X,y)
        res.show()        
   
    def test_standardized_residuals(self, get_regression_data):
        X, y = get_regression_data
        est = RidgeRegression(epochs=500)
        res = StandardizedResiduals(est)
        res.fit(X,y)
        res.show()        

    
    def test_studentized_residuals(self, get_regression_data):
        X, y = get_regression_data
        est = LassoRegression(epochs=500)
        res = StudentizedResiduals(est)
        res.fit(X,y)
        res.show()          

    def test_scale_location(self, get_regression_data):
        X, y = get_regression_data
        est = LassoRegression(epochs=500)
        res = ScaleLocation(est)
        res.fit(X,y)
        res.show()            

    def test_normal_qq(self, get_regression_data):
        X, y = get_regression_data
        est = LinearRegression(epochs=500)
        res = QQPlot(est)
        res.fit(X,y)
        res.show()         

    def test_residuals_leverage(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=1000, batch_size=32)
        res = ResidualsLeverage(est, width=1200)
        res.fit(X,y)
        res.show()           

    def test_residuals_leverage_change_threshold(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=1000, batch_size=32)
        res = ResidualsLeverage(est, width=1200, inner_threshold=0.05, outer_threshold=0.1)
        res.fit(X,y)
        res.show()           

    @mark.residuals
    def test_cooks_distance(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=1000, batch_size=32)
        res = CooksDistance(est)
        res.fit(X,y)
        res.show()             

    @mark.residuals
    def test_cooks_distance_threshold(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=1000, batch_size=32)
        res = CooksDistance(est, threshold= 0.01)
        res.fit(X,y)
        res.show()                     