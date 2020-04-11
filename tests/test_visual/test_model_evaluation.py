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
from mlstudio.supervised.regression import LinearRegression, LassoRegression
from mlstudio.supervised.regression import RidgeRegression, ElasticNetRegression
from mlstudio.visual.model_evaluation import PredictionError

@mark.plots
@mark.prediction_error
@mark.skip(reason="visually tested")
class PredictionErrorTests:

    @mark.parametrize("algorithm", [LinearRegression(), LassoRegression(),
                      RidgeRegression(), ElasticNetRegression()])
    def test_prediction_error(self, get_regression_data, algorithm):
        X, y = get_regression_data
        est = GradientDescentRegressor(algorithm=algorithm)        
        pe = PredictionError(est)
        pe.fit(X,y)
        pe.show()