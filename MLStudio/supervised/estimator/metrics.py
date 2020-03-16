# =========================================================================== #
#                                  METRICS MODULE                             #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \metrics.py                                                           #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday September 24th 2019, 3:16:03 am                        #
# Last Modified: Saturday November 30th 2019, 10:37:08 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Classification and regression metrics classes."""
from abc import ABC, abstractmethod
import math
import numpy as np

from mlstudio.utils.data_manager import decode

class Metric(ABC):
    """Abstract base class for all metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class RegressionMetric(Metric):
    """Base class for regression metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class ClassificationMetric(Metric):
    """Base class for classification metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

# --------------------------------------------------------------------------- #
#                           REGRESSION METRICS                                #
# --------------------------------------------------------------------------- #
class SSR(RegressionMetric):
    """Computes sum squared residuals given"""

    def __init__(self):
        self.mode = 'min'
        self.name = 'residual_sum_squared_error'
        self.label = "Residual Sum Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1

    
    def __call__(self, y, y_pred):
        e = y - y_pred
        return np.sum(e**2)  

class SST(RegressionMetric):
    """Computes total sum of squares"""

    def __init__(self):
        self.mode = 'min'
        self.name = 'total_sum_squared_error'
        self.label = "Total Sum Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1

    
    def __call__(self, y, y_pred):
        y_avg = np.mean(y)
        e = y-y_avg                
        return np.sum(e**2)

class R2(RegressionMetric):
    """Computes coefficient of determination."""

    def __init__(self):
        self.mode = 'max'        
        self.name = 'R2'
        self.label = r"$R^2$"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        self._ssr = SSR()
        self._sst = SST()
        r2 = 1 - (self._ssr(y, y_pred)/self._sst(y, y_pred))        
        return r2

class VarExplained(RegressionMetric):
    """Computes proportion of variance explained."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'percent_variance_explained'
        self.label = "Percent Variance Explained"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        var_explained = 1 - (np.var(y-y_pred) / np.var(y))
        return var_explained                   

class MAE(RegressionMetric):
    """Computes mean absolute error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_absolute_error'
        self.label = "Mean Absolute Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = abs(y-y_pred)
        return np.mean(e)


class MSE(RegressionMetric):
    """Computes mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_squared_error'
        self.label = "Mean Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.mean(e**2)

class NMSE(RegressionMetric):
    """Computes negative mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'neg_mean_squared_error'
        self.label = "Negative Mean Squared Error"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.mean(e**2) * -1

class RMSE(RegressionMetric):
    """Computes root mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'root_mean_squared_error'
        self.label = "Root Mean Squared Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.sqrt(np.mean(e**2)) 

class NRMSE(RegressionMetric):
    """Computes negative root mean squared error given data and parameters."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'neg_root_mean_squared_error'
        self.label = "Negative Root Mean Squared Error"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1

    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return -np.sqrt(np.mean(e**2))

class MSLE(RegressionMetric):
    """Computes mean squared log error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_squared_log_error'
        self.label = "Mean Squared Log Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = np.log(y+1)-np.log(y_pred+1)
        return np.mean(e**2)  

class RMSLE(RegressionMetric):
    """Computes root mean squared log error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'root_mean_squared_log_error'
        self.label = "Root Mean Squared Log Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):
        e = np.log(y+1)-np.log(y_pred+1)
        return np.sqrt(np.mean(e**2))

class MEDAE(RegressionMetric):
    """Computes median absolute error given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'median_absolute_error'
        self.label = "Median Absolute Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):        
        return np.median(np.abs(y_pred-y))

class MAPE(RegressionMetric):
    """Computes mean absolute percentage given data and parameters."""

    def __init__(self):
        self.mode = 'min'
        self.name = 'mean_absolute_percentage_error'
        self.label = "Mean Absolute Percentage Error"
        self.stateful = False
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.precision_factor = -1
    
    def __call__(self, y, y_pred):        
        return 100*np.mean(np.abs((y-y_pred)/y))

class RegressionMetricFactory:
    """Returns the requested score class."""

    def __call__(self, metric='mse'):

        dispatcher = {'r2': R2(),
                      'var_explained': VarExplained(),
                      'mae': MAE(),
                      'mse': MSE(),                      
                      'nmse': NMSE(),
                      'rmse': RMSE(),
                      'nrmse': NRMSE(),
                      'msle': MSLE(),
                      'rmsle': RMSLE(),
                      'medae': MEDAE(),
                      'mape': MAPE()}
        return(dispatcher.get(metric,False))

# --------------------------------------------------------------------------- #
#                       CLASSIFICATION METRICS                                #
# --------------------------------------------------------------------------- #
class Accuracy(ClassificationMetric):
    """Computes accuracy."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'accuracy'
        self.label = "Accuracy"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.precision_factor = 1
    
    def __call__(self, y, y_pred):
        """Computes accuracy as correct over total."""
        # If scoring multinomial logistical regression with one-hot vectors,
        # convert them back to 1d integers.
        if len(y.shape) > 1:
            y = decode(y)
        if len(y_pred.shape) > 1:
            y_pred = decode(y_pred)
        return np.sum(np.equal(y,y_pred)) / y.shape[0]

class ClassificationMetricFactory:
    """Returns the requested score class."""

    def __call__(self, metric='accuracy'):

        dispatcher = {'accuracy': Accuracy()}
        return(dispatcher.get(metric,False))
