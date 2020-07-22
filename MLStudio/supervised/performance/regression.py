# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \regression.py                                                    #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 2:25:57 am                        #
# Last Modified : Thursday, July 16th 2020, 2:25:58 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
import math
import numpy as np

from mlstudio.supervised.metrics.base import RegressionMetric
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
        self.epsilon_factor = -1

    
    def __call__(self, y, y_pred, *args, **kwargs):
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
        self.epsilon_factor = -1

    
    def __call__(self, y, y_pred, *args, **kwargs):
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
        self.epsilon_factor = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):
        self._ssr = SSR()
        self._sst = SST()
        r2 = 1 - (self._ssr(y, y_pred)/self._sst(y, y_pred))     
        return r2


class AdjustedR2(RegressionMetric):
    """Computes adjusted coefficient of determination."""

    def __init__(self):
        self.mode = 'max'        
        self.name = 'R2'
        self.label = r"$\text{Adjusted }R^2$"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1

    
    def __call__(self, y, y_pred, X=None):
        self._ssr = SSR()
        self._sst = SST()        
        n = X.shape[0]
        p = X.shape[1] - 1
        df_e = n-p-1
        df_t = n-1
        ar2 = 1 - ((self._ssr(y, y_pred)/df_e)/(self._sst(y, y_pred)/df_t))        
        return ar2

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
        self.epsilon_factor = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):
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
        self.epsilon_factor = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):
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
        self.epsilon_factor = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):        
        e = y - y_pred
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
        self.epsilon_factor = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):        
        e = y - y_pred
        return -np.mean(e**2)

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
        self.epsilon_factor = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):
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
        self.epsilon_factor = 1

    
    def __call__(self, y, y_pred, *args, **kwargs):
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
        self.epsilon_factor = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):
        e = np.log(y+1)-np.log(y_pred+1)
        y = np.clip(y, 1e-15, 1-1e-15)    
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)    
        e = np.log(y)-np.log(y_pred)
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
        self.epsilon_factor = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):
        y = np.clip(y, 1e-15, 1-1e-15)    
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)    
        e = np.log(y)-np.log(y_pred)
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
        self.epsilon_factor = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):        
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
        self.epsilon_factor = -1
    
    def __call__(self, y, y_pred, *args, **kwargs):        
        return 100*np.mean(np.abs((y-y_pred)/y))
