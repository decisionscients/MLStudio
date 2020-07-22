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
"""Base classes for performance analytics subclasses. 

The PerformanceAnalytics class occupies the top of the class hierarchy. 

From this class, we derive the Measures and Metrics subclasses.  Measures,
primarily used in classification problems, are derived from taking a 
measurement. Examples include the number of true positives. A metric is
a calculation between two or more measures, e.g. true positive rate.
Measures can be reported; whereas metrics can be used for scoring and
generalized performance estimation.

The RegressionMetrics, ClassificationMeasures and ClassificationMetrics
classes are derived from Measures and Metrics.
"""
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
# --------------------------------------------------------------------------- #
class BasePerformance(ABC, BaseEstimator):
    """Abstract base class for performance analytics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    def _format_results(self, y, y_pred):
        """Formats results into dataframe for evaluation."""
        y = np.array(y)
        y_pred = np.array(y_pred)
        d = {'y': y, 'y_pred': y_pred}
        df = pd.DataFrame(data=d)
        return df                                  

class BaseMeasure(BasePerformance):
    """Base class for performance measures."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class BaseMetric(BasePerformance):
    """Base class for performance metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class BaseRegressionMeasure(BaseMeasure):
    """Base class for regresssion measures."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class BaseRegressionMetric(BaseMetric):
    """Base class for regresssion metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class BaseClassificationMeasure(BaseMeasure):
    """Base class for regresssion measures."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class BaseClassificationMetric(BaseMetric):
    """Base class for regresssion metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class ClassificationMetric(Metric):
    """Base class for classification metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, *args, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")