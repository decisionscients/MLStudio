# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_data_management copy.py                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Monday, June 29th 2020, 11:58:17 am                         #
# Last Modified : Monday, June 29th 2020, 11:58:17 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests wrappers."""
#%%
import numpy as np
import pytest
from pytest import mark
from sklearn.datasets import make_regression

from mlstudio.utils.data_manager import StandardScaler, data_split
from mlstudio.utils.wrappers import RegressionWrapperStatsmodel
# --------------------------------------------------------------------------  #
#                     REGRESSION WRAPPER STATSMODELS                          #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.wrappers
def test_regression_wrapper_statsmodels():
    X, y = make_regression(n_features=50)
    X_train, X_test, y_train, y_test = data_split(X, y, random_state=5)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # Instantiate and fit model
    estimator = RegressionWrapperStatsmodel()
    estimator.fit(X_train, y_train)
    print(estimator.results_.summary())
    # Score method
    r2 = estimator.score(X_test, y_test)
    assert r2 > 0.8, "Poor R2 score"

