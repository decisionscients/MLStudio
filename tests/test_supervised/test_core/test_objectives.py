#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_objectives.py                                                #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 3:45:31 pm                          #
# Last Modified : Monday, June 15th 2020, 3:45:31 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
import math
import numpy as np
import pytest
from pytest import mark

from sklearn.metrics import mean_squared_error

from mlstudio.supervised.core.objectives import MSE, CrossEntropy
from mlstudio.supervised.core.objectives import CategoricalCrossEntropy
from mlstudio.supervised.core.regularizers import L1, L2, L1_L2
# --------------------------------------------------------------------------  #
@mark.objectives
class MSETests:

    def test_mse_cost(get_regression_prediction):
        y, y_pred = get_regression_prediction
        theta = {}
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=20)
        theta['bias'] = np.random.default_rng().uniform(low=0, high=1, size=1)
        mse = 0.5 * mean_squared_error(y_true=y, y_pred=y_pred)



