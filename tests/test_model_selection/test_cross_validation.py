# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_components.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 11th 2020, 4:54:32 am                        #
# Last Modified : Saturday, July 11th 2020, 4:54:32 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests for data analysis functions."""
import numpy as np
import pytest
from pytest import mark

from mlstudio.supervised.machine_learning.gradient_descent import GDRegressor
from mlstudio.supervised.algorithms.optimization.services.loss import Quadratic
from mlstudio.supervised.algorithms.optimization.services.regularizers import L2
from mlstudio.model.development import ModelBuilder
# --------------------------------------------------------------------------- #
@mark.cross_validation
class CrossValidationTests:

    def test_nested_cv(self, get_regression_data_unscaled):
        X, y = get_regression_data_unscaled
        est = GDRegressor()
        param_set = [
            {"epochs": [100,200,500,1000],
             "batch_size": [32, 64]}
        ]
        ncv = ModelBuilder(estimator=est, parameters=param_set) 
        ncv.fit(X,y)
        print(ncv.results_.get('test_score'))
        print(ncv.results_.get('estimator'))
        


