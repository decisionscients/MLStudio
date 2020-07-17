# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \classification.py                                                #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Thursday, July 16th 2020, 2:26:07 am                        #
# Last Modified : Thursday, July 16th 2020, 2:26:07 am                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
import math
import numpy as np

from mlstudio.supervised.metrics.base import ClassificationScorer
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
        self.probability_metric = False
    
    def __call__(self, y, y_pred, *args, **kwargs):
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
        self.probability_metric = False
    
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
        self.probability_metric = False
    
    def __call__(self, y, y_pred, positive=1, negative=0):
        df = format_results(y, y_pred)
        tp = true_positives(df, positive, negative)
        fp = false_positives(df, positive, negative)
        fn = false_negatives(df, positive, negative)
        num = 2.0 * tp 
        den = (2.0 * tp + fp + fn)
        try:
            f1 = num / den
        except Exception as e:
            if den == 0:
                f1 = 0
            else:
                print(e)
        return f1
            

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
        self.probability_metric = False
    
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
        self.probability_metric = False
    
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
        self.probability_metric = False
    
    def __call__(self, y, y_pred, positive=1, negative=0):
        """Computes precision as tn / (tn + fp)."""        
        df = format_results(y, y_pred)
        tn = true_negatives(df, positive=positive, negative=negative)
        fp = false_positives(df, positive=positive, negative=negative)
        return tn / (tn + fp)                
# --------------------------------------------------------------------------- #
#                    CLASSIFICATION HELPER FUNCTIONS                          #
# --------------------------------------------------------------------------- #        
def format_results(y, y_pred, *args, **kwargs):
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