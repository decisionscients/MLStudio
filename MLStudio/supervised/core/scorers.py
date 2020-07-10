#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : scorers.py                                                         #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                       #
# Last Modified : Wednesday, March 18th 2020, 2:07:20 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Classification and regression scorer classes."""
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

class Scorer(ABC, BaseEstimator):
    """Abstract base class for all metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class RegressionScorer(Scorer):
    """Base class for regression metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

class ClassificationScorer(Scorer):
    """Base class for classification metrics."""

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

    @abstractmethod
    def __call__(self, y, y_pred, **kwargs):
        raise NotImplementedError("This method is not implemented for "
                                  "this Abstract Base Class.")

# --------------------------------------------------------------------------- #
#                           REGRESSION SCORERS                                #
# --------------------------------------------------------------------------- #
class SSR(RegressionScorer):
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

    
    def __call__(self, y, y_pred):
        e = y - y_pred
        return np.sum(e**2)  

class SST(RegressionScorer):
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

    
    def __call__(self, y, y_pred):
        y_avg = np.mean(y)
        e = y-y_avg                
        return np.sum(e**2)

class R2(RegressionScorer):
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

    
    def __call__(self, y, y_pred):
        self._ssr = SSR()
        self._sst = SST()
        r2 = 1 - (self._ssr(y, y_pred)/self._sst(y, y_pred))        
        return r2


class AdjustedR2(RegressionScorer):
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

    
    def __call__(self, y, y_pred):
        self._ssr = SSR()
        self._sst = SST()
        X = kwargs.get('X')
        n = X.shape[0]
        p = X.shape[1] - 1
        df_e = n-p-1
        df_t = n-1
        ar2 = 1 - ((self._ssr(y, y_pred)/df_e)/(self._sst(y, y_pred)/df_t))        
        return ar2

class VarExplained(RegressionScorer):
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

    
    def __call__(self, y, y_pred):
        var_explained = 1 - (np.var(y-y_pred) / np.var(y))
        return var_explained                   

class MAE(RegressionScorer):
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
    
    def __call__(self, y, y_pred):
        e = abs(y-y_pred)
        return np.mean(e)


class MSE(RegressionScorer):
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
    
    def __call__(self, y, y_pred):        
        e = y - y_pred
        return np.mean(e**2)

class NMSE(RegressionScorer):
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

    
    def __call__(self, y, y_pred):        
        e = y - y_pred
        return -np.mean(e**2)

class RMSE(RegressionScorer):
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
    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return np.sqrt(np.mean(e**2)) 

class NRMSE(RegressionScorer):
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

    
    def __call__(self, y, y_pred):
        e = y-y_pred
        return -np.sqrt(np.mean(e**2))

class MSLE(RegressionScorer):
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
    
    def __call__(self, y, y_pred):
        e = np.log(y+1)-np.log(y_pred+1)
        y = np.clip(y, 1e-15, 1-1e-15)    
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)    
        e = np.log(y)-np.log(y_pred)
        return np.mean(e**2)

class RMSLE(RegressionScorer):
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
    
    def __call__(self, y, y_pred):
        y = np.clip(y, 1e-15, 1-1e-15)    
        y_pred = np.clip(y_pred, 1e-15, 1-1e-15)    
        e = np.log(y)-np.log(y_pred)
        return np.sqrt(np.mean(e**2))

class MEDAE(RegressionScorer):
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
    
    def __call__(self, y, y_pred):        
        return np.median(np.abs(y_pred-y))

class MAPE(RegressionScorer):
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
    
    def __call__(self, y, y_pred):        
        return 100*np.mean(np.abs((y-y_pred)/y))


# --------------------------------------------------------------------------- #
#                       CLASSIFICATION SCORERS                                #
# --------------------------------------------------------------------------- #
class Accuracy(ClassificationScorer):
    """Computes accuracy."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'accuracy'
        self.label = "Accuracy"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
    
    def __call__(self, y, y_pred):
        """Computes accuracy as correct over total."""        
        return np.sum(np.equal(y,y_pred)) / y.shape[0]

class BalancedAccuracy(ClassificationScorer):
    """Computes balanced accuracy."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'balanced_accuracy'
        self.label = "Balanced Accuracy"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
    
    def __call__(self, y, y_pred, positive=1, negative=0):
        """Computes balanced accuracy for unbalanced data sets."""        
        tpr = Recall()(y, y_pred, positive, negative) 
        tnr = Specificity()(y, y_pred, positive, negative)
        return (tpr + tnr) / 2

class F1(ClassificationScorer):
    """Computes F1 Score as 2TP / (2TP+FP+FN)"""

    def __init__(self):
        self.mode = 'max'
        self.name = 'F1'
        self.label = "F1 Score"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
    
    def __call__(self, y, y_pred, positive=1, negative=0):
        df = format_results(y, y_pred)
        tp = true_positives(df, positive, negative)
        fp = false_positives(df, positive, negative)
        fn = false_negatives(df, positive, negative)
        return 2 * tp / (2 * tp + fp + fn)
            

class Precision(ClassificationScorer):
    """Computes precision as tp / (tp + fp)."""       

    def __init__(self):
        self.mode = 'max'
        self.name = 'precision'
        self.label = "Precision"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
    
    def __call__(self, y, y_pred, positive=1, negative=0):         
        df = format_results(y, y_pred)
        tp = true_positives(df, positive=positive, negative=negative)
        fp = false_positives(df, positive=positive, negative=negative)
        return tp / (tp + fp)

class Recall(ClassificationScorer):
    """Computes Recall as tp / (tp + fn)."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'recall'
        self.label = "Recall"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
    
    def __call__(self, y, y_pred, positive=1, negative=0):        
        df = format_results(y, y_pred)
        tp = true_positives(df, positive=positive, negative=negative)
        fn = false_negatives(df, positive=positive, negative=negative)
        return tp / (tp + fn)

class Specificity(ClassificationScorer):
    """Computes specificity as tn / (tn + fp)."""

    def __init__(self):
        self.mode = 'max'
        self.name = 'specificity'
        self.label = "Specificity"
        self.stateful = False
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
    
    def __call__(self, y, y_pred, positive=1, negative=0):
        """Computes precision as tn / (tn + fp)."""        
        df = format_results(y, y_pred)
        tn = true_negatives(df, positive=positive, negative=negative)
        fp = false_positives(df, positive=positive, negative=negative)
        return tn / (tn + fp)                
# --------------------------------------------------------------------------- #
#                    CLASSIFICATION HELPER FUNCTIONS                          #
# --------------------------------------------------------------------------- #        
def format_results(y, y_pred):
    """Formats results into dataframe for evaluation."""
    y = np.array(y)
    y_pred = np.array(y_pred)
    d = {'y': y, 'y_pred': y_pred}
    df = pd.DataFrame(data=d)
    return df

def true_positives(df, positive=1, negative=0):
    """Computes true positives in binary classification."""
    result = df[(df['y'] == positive) & (df['y_pred'] == positive)]
    return len(result.index)

def true_negatives(df, positive=1, negative=0):
    """Computes true negatives in binary classification."""
    result = df[(df['y'] == negative) & (df['y_pred'] == negative)]
    return len(result.index)    

def false_positives(df, positive=1, negative=0):
    """Computes false positives in binary classification."""
    result = df[(df['y'] == negative) & (df['y_pred'] == positive)]
    return len(result.index)

def false_negatives(df, positive=1, negative=0):
    """Computes false negatives in binary classification."""
    result = df[(df['y'] == positive) & (df['y_pred'] == negative)]
    return len(result.index)    