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
from mlstudio.visual.model_validation import Residuals

class ResidualPlotTests:

    @mark.residuals
    def test_residuals(self, get_regression_data):
        X, y = get_regression_data
        est = ElasticNetRegression(epochs=1000)
        res = Residuals(est)
        res.fit(X,y)
        res.show()