# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \base.py                                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 2:26:47 am                        #
# Last Modified : Thursday, July 16th 2020, 2:26:47 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Base class for all scorer subclasses. """
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
# --------------------------------------------------------------------------- #
class Scorer(ABC, BaseEstimator):
    """Abstract base class for all metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class RegressionScorer(Scorer):
    """Base class for regression metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class ClassificationScorer(Scorer):
    """Base class for classification metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")