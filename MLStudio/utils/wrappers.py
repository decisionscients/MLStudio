# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \wrappers.py                                                      #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Monday, June 29th 2020, 9:51:49 am                          #
# Last Modified : Monday, June 29th 2020, 11:18:36 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Scikit-Learn style wrappers for statsmodels and scipy"""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.base import BaseEstimator
import scipy.stats as sp
import statsmodels.api as sm 

from mlstudio.supervised.core.scorers import R2
# --------------------------------------------------------------------------- #
class RegressionWrapper(ABC, BaseEstimator):
    """Abstract base class for regression wrappers."""
    
    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X, y=None):
        pass

    @abstractmethod
    def score(self, X, y):
        pass            

# --------------------------------------------------------------------------- #
class RegressionWrapperStatsmodel(RegressionWrapper):
    """Scikit-Learn style wrapper for statsmodels linear regression function."""

    def __init__(self):
        pass

    def fit(self, X, y):
        X = sm.add_constant(X)
        self.model_ = sm.OLS(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X, y=None):
        X = sm.add_constant(X)
        return self.results_.predict(X)

    def score(self, X, y):
        y_pred = self.predict(X)
        scorer = R2()
        return scorer(y, y_pred)

        