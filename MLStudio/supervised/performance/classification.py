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
"""Performance analytics classes for classification problems. 

This module contains classification measures and metrics. Measures are derived
from taking a measurement, e.g. true positives. Metrics are computed based
upon two or more measures, e.g. true positive rate. Measures and metrics
are used to report performance; however, only metrics can be used for 
scoring and performance generalization estimation.

"""
import math
import numpy as np

from mlstudio.supervised.performance.base import BaseClassificationMeasure
from mlstudio.supervised.performance.base import BaseClassificationMetric
# --------------------------------------------------------------------------- #
#                        CLASSIFICATION MEASURES                              #
# --------------------------------------------------------------------------- #

# -------------------------- BASE MEASURES ---------------------------------- #
class TruePositives(BaseClassificationMeasure):
    """Computes the number true positives."""
    
    def __init__(self):
        self.code = "TP"        
        self.name = "True Positives (Power)"
        self.category = "Base Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == positive) & (df['y_pred'] == positive)]
        return len(result.index)    

class TrueNegatives(BaseClassificationMeasure):
    """Computes the number true negatives."""
    
    def __init__(self):
        self.code = "TN"        
        self.name = "True Negatives"
        self.category = "Base Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == negative) & (df['y_pred'] == negative)]
        return len(result.index)    

class FalsePositives(BaseClassificationMeasure):
    """Computes the number false positives."""
    
    def __init__(self):
        self.code = "FP"        
        self.name = "False Positives (Type I Error)"
        self.category = "Base Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == negative) & (df['y_pred'] == positive)]
        return len(result.index)    

class FalseNegatives(BaseClassificationMeasure):
    """Computes the number false negatives."""
    
    def __init__(self):
        self.code = "FN"        
        self.name = "False Negatives (Type II Error)"
        self.category = "Base Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        df = self._format_results(y=y, y_pred=y_pred)
        result = df[(df['y'] == positive) & (df['y_pred'] == negative)]
        return len(result.index)            

# -------------------------- 1st LEVEL MEASURES ----------------------------- #
class PositiveCondition(BaseClassificationMeasure):
    """Computes positive condition as true positives + false negatives."""
    
    def __init__(self):
        self.code = "P"        
        self.name = "Positive Condition"
        self.category = "1st Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        tp = TruePositives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)
        fn = FalseNegatives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)                              
        return tp + fn

class NegativeCondition(BaseClassificationMeasure):
    """Computes negative condition as false positives + true negatives."""    
    
    def __init__(self):
        self.code = "N"        
        self.name = "Negative Condition"
        self.category = "1st Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        fp = FalsePositives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)
        tn = TrueNegatives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)                              
        return fp + tn        

# --------------------------------------------------------------------------- #
class OutcomePositive(BaseClassificationMeasure):
    """Computes outcome positive as true positives plus false positives."""
    
    def __init__(self):
        self.code = "OP"        
        self.name = "Outcome Positive"
        self.category = "1st Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        tp = TruePositives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)
        fp = FalsePositives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)                              
        return tp + fp

class OutcomeNegative(BaseClassificationMeasure):
    """Computes outcome positive as false negatives plus true negatives."""    
    
    def __init__(self):
        self.code = "ON"        
        self.name = "Outcome Negative"
        self.category = "1st Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        fn = FalseNegatives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)
        tn = TrueNegatives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)                              
        return fn + tn        

# --------------------------------------------------------------------------- #
class TrueClassification(BaseClassificationMeasure):
    """True classification is the sum of true positives and true negatives.""" 
    
    def __init__(self):
        self.code = "TC"        
        self.name = "True Classification"
        self.category = "1st Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        tp = TruePositives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)
        tn = TrueNegatives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)                              
        return tp + tn        

class FalseClassification(BaseClassificationMeasure):
    """False classification is the sum of false positives and false negatives."""
    
    def __init__(self):
        self.code = "FC"        
        self.name = "False Classification"
        self.category = "1st Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        fp = FalsePositives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)
        fn = FalseNegatives()(y=y, y_pred=y_pred, negative=negative, 
                              positive=positive)                              
        return fp + fn        

# ------------------------ 2ND LEVEL MEASURES ------------------------------- #
class PositiveLikelihoodRatio(BaseClassificationMeasure):
    """Positive likelihood ratio is true positive rate / false positive rate."""

    def __init__(self):
        self.code = 'LRP'
        self.name = "Positive Likelihood Ratio"
        self.category = "2nd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)
        fpr = FalsePositiveRate()(y, y_pred, positive=positive, negative=negative)
        return tpr / fpr    

class NegativeLikelihoodRatio(BaseClassificationMeasure):
    """Negative likelihood ratio is false negative rate / true negative rate."""

    def __init__(self):
        self.code = 'LRP'
        self.name = "Positive Likelihood Ratio"
        self.category = "2nd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        fnr = FalseNegativeRate()(y, y_pred, positive=positive, negative=negative)
        tnr = TrueNegativeRate()(y, y_pred, positive=positive, negative=negative)
        return fnr / tnr

# --------------------------------------------------------------------------- #
class Bias(BaseClassificationMeasure):
    """Computes bias as the ratio of outcome positive and the sample size."""

    def __init__(self):
        self.code = 'BIAS'
        self.name = "Bias"
        self.category = "2nd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        op = OutcomePositive()(y, y_pred, positive=positive, negative=negative)        
        return op / y.shape[0]

# --------------------------------------------------------------------------- #
class Prevalence(BaseClassificationMeasure):
    """Computes prevalence as positive condition / sample size."""

    def __init__(self):
        self.code = 'PREV'
        self.name = "Prevalence"
        self.category = "2nd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        p = PositiveCondition()(y, y_pred, positive=positive, negative=negative)        
        return p / y.shape[0]

class Skew(BaseClassificationMeasure):
    """Computes skew as ratio of negative and positive condition."""

    def __init__(self):
        self.code = 'SKEW'
        self.name = "Skew"
        self.category = "2nd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        n = NegativeCondition()(y, y_pred, positive=positive, negative=negative)        
        p = PositiveCondition()(y, y_pred, positive=positive, negative=negative)        
        return n / p
# --------------------------------------------------------------------------- #
class CohensKappaChance(BaseClassificationMeasure):
    """Computes Cohens Kappa Chance as (P*OP + N*ON) / sample size squared."""

    def __init__(self):
        self.code = 'CKc'
        self.name = "Cohen's Kappa Chance"
        self.category = "2nd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        n = NegativeCondition()(y, y_pred, positive=positive, negative=negative)        
        p = PositiveCondition()(y, y_pred, positive=positive, negative=negative)        
        op = OutcomePositive()(y, y_pred, positive=positive, negative=negative)        
        on = OutcomeNegative()(y, y_pred, positive=positive, negative=negative)                
        return ((p * op) + (n * on)) / y.shape[0]**2

# ------------------------ 3RD LEVEL MEASURES ------------------------------- #
class OddsRatio(BaseClassificationMeasure):
    """Computes odds ratio as (tp-tn) / (fp-fn)."""

    def __init__(self):
        self.code = 'OR'
        self.name = "Odds Ratio"
        self.category = "3rd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        tp = TruePositives()(y, y_pred, positive=positive, negative=negative)
        tn = TrueNegatives()(y, y_pred, positive=positive, negative=negative)        
        fp = FalsePositives()(y, y_pred, positive=positive, negative=negative)        
        fn = FalseNegatives()(y, y_pred, positive=positive, negative=negative)        
        return (tp-tn) / (fp-fn)        

class DiscrimitivePower(BaseClassificationMeasure):
    """Computes descrimitive power as:
    
    .. math:: \frac{\sqrt{3}}{\pi}\text{log(OR)}
    
    """

    def __init__(self):
        self.code = 'DP'
        self.name = "Discrimitive Power"
        self.category = "3rd Level Measure"

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):
        return (np.sqrt(3) / np.pi) * \
            np.log(OddsRatio()(y, y_pred, positive=positive, negative=negative))        
        
# --------------------------------------------------------------------------- #
#                        CLASSIFICATION METRICS                               #
# --------------------------------------------------------------------------- #

# --------------------------- BASE METRICS ---------------------------------- #
class Accuracy(BaseClassificationMetric):
    """Computes accuracy ratio of true classification and sample size."""

    def __init__(self):
        self.code = "ACC"
        self.name = "Accuracy"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):                
        tc = TrueClassification()(y, y_pred, positive=positive, negative=negative)   
        return tc / y.shape[0]

class DetectionRate(BaseClassificationMetric):
    """Computes detection rate as tp/sn."""

    def __init__(self):
        self.code = "DR"
        self.name = "Detection Rate"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):   
        tp = TruePositives()(y, y_pred, positive=positive, negative=negative)
        return tp / y.shape[0]        

class RejectionRate(BaseClassificationMetric):
    """Computes rejection rate as tn/sn."""

    def __init__(self):
        self.code = "CRR"
        self.name = "Rejection Rate (Corrected)"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        tn = TrueNegatives()(y, y_pred, positive=positive, negative=negative)
        return tn / y.shape[0]

# --------------------------------------------------------------------------- #
class PositivePredictiveValue(BaseClassificationMetric):
    """Positive predictive value as ratio of true positives and outcome positive."""       

    def __init__(self):
        self.code = "PPV"
        self.name = "Positive Predictive Value (Precision)"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        tp = TruePositives()(y, y_pred, positive=positive, negative=negative)
        op = OutcomePositive()(y, y_pred, positive=positive, negative=negative)
        return tp / op

class Precision(PositivePredictiveValue):
    """Alias class for PositivePredictiveValue."""

    def __init__(self):
        self.code = "PPV"
        self.name = "Precision"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False    

class NegativePredictiveValue(BaseClassificationMetric):
    """Negative predictive value as ratio of true negatives and outcome negative."""       

    def __init__(self):
        self.code = "NPV"
        self.name = "Negative Predictive Value"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        tn = TrueNegatives()(y, y_pred, positive=positive, negative=negative)
        on = OutcomeNegative()(y, y_pred, positive=positive, negative=negative)
        return tn / on

class FalseDiscoveryRate(BaseClassificationMetric):
    """False discovery rate as ratio of false positives and outcome positive."""       

    def __init__(self):
        self.code = "FDR"
        self.name = "False Discovery Rate"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        fp = FalsePositives()(y, y_pred, positive=positive, negative=negative)
        op = OutcomePositive()(y, y_pred, positive=positive, negative=negative)
        return fp / op

class FalseOmissionRate(BaseClassificationMetric):
    """False omission rate as ratio of false negatives and outcome negative."""       

    def __init__(self):
        self.code = "FOR"
        self.name = "False Omission Rate"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        fn = FalseNegatives()(y, y_pred, positive=positive, negative=negative)
        on = OutcomeNegative()(y, y_pred, positive=positive, negative=negative)
        return fn / on

# --------------------------------------------------------------------------- #
class AUC:
    """Area under Receiver Operating Characteristic (ROC) Curve """
    @property
    def auc(self):
        try:
            return self._roc_data['auc'].sum()
        except:
            raise Exception("AUC has not been calculated.")            
    @property
    def roc_data(self):
        try:
            return self._roc_data
        except:
            raise Exception("ROC data has not been calculated.")

    def _compute_roc_data(self, df):
        # Compute predictions and sort by probabilities (desc)
        df['y_prob'] = df['y_pred']
        df['y_pred'] = int(df['y_prob'].round({'y_prob':0}))
        df = df.sort_values(by='y_prob', axis=0, ascending=False)
        # Add true positives and false positives
        df['tp'] = np.where(df['y_pred']==1 & df['y']==1, 1, 0)
        df['fp'] = np.where(df['y_pred']==1 & df['y']==0, 1, 0)
        df['tn'] = np.where(df['y_pred']==0 & df['y']==0, 1, 0)
        df['fn'] = np.where(df['y_pred']==0 & df['y']==1, 1, 0)
        # Count them up
        df['n_tp'] = df.groupby('tp').cumcount()
        df['n_fp'] = df.groupby('fp').cumcount()
        df['n_tn'] = df.groupby('tn').cumcount()
        df['n_fn'] = df.groupby('fn').cumcount()
        # Compute statistics
        df['tpr'] = df['n_tp'] / (df['n_tp']+df['n_fn'])
        df['fpr'] = df['n_fp'] / (df['n_fp']+df['n_tn'])
        width = df['fpr'].diff()
        height = (df['tpr'].shift(1) + df['tpr'])/2
        df['auc'] = width * height
        self._roc_data = df        

    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):                       
        df = self._format_results(y, y_pred)
        self._compute_roc_data(df)
        return self._roc_data['auc'].sum()

# --------------------------------------------------------------------------- #
class TruePositiveRate(BaseClassificationMetric):
    """True positive rate computed as true positives / positive condition."""       

    def __init__(self):
        self.code = "TPR"
        self.name = "True Positive Rate"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        tp = TruePositives()(y, y_pred, positive=positive, negative=negative)
        p = PositiveCondition()(y, y_pred, positive=positive, negative=negative)
        return tp / p

class Sensitivity(TruePositiveRate):
    """Alias for True Positive Rate."""       

    def __init__(self):
        self.code = "TPR"
        self.name = "Sensitivity"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    

class Recall(TruePositiveRate):
    """Alias for True Positive Rate."""       

    def __init__(self):
        self.code = "TPR"
        self.name = "Recall"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    

class FalseNegativeRate(BaseClassificationMetric):
    """False negative rate computed as false negatives / positive condition."""       

    def __init__(self):
        self.code = "FNR"
        self.name = "False Negative Rate"
        self.category = "Base Metric"
        self.mode = 'min'
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.epsilon_factor = -1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        fn = FalseNegatives()(y, y_pred, positive=positive, negative=negative)
        p = PositiveCondition()(y, y_pred, positive=positive, negative=negative)
        return fn / p

class TrueNegativeRate(BaseClassificationMetric):
    """True negative rate computed as true negatives / negative condition."""       

    def __init__(self):
        self.code = "TNR"
        self.name = "True Negative Rate (Specificity)"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        tn = TrueNegatives()(y, y_pred, positive=positive, negative=negative)
        n = NegativeCondition()(y, y_pred, positive=positive, negative=negative)
        return tn / n

class Specificity(TrueNegativeRate):
    """Alias for TrueNegativeRate."""       

    def __init__(self):
        self.code = "TNR"
        self.name = "Specificity (True Negative Rate)"
        self.category = "Base Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
        
class FalsePositiveRate(BaseClassificationMetric):
    """False positive rate computed as false positives / negative condition."""       

    def __init__(self):
        self.code = "FPR"
        self.name = "False Positive Rate"
        self.category = "Base Metric"
        self.mode = 'min'
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.epsilon_factor = -1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        fp = FalsePositives()(y, y_pred, positive=positive, negative=negative)
        n = NegativeCondition()(y, y_pred, positive=positive, negative=negative)
        return fp / n

# --------------------------------------------------------------------------- #
class MissclassificationRate(BaseClassificationMetric):
    """Missclassification rate computed as false classification / negative condition."""       

    def __init__(self):
        self.code = "MCR"
        self.name = "Missclassification Rate"
        self.category = "Base Metric"
        self.mode = 'min'
        self.best = np.min
        self.better = np.less
        self.worst = np.Inf
        self.epsilon_factor = -1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        fc = FalseClassification()(y, y_pred, positive=positive, negative=negative)        
        return fc / y.shape[0]

# -------------------------- 1ST LEVEL METRICS ------------------------------ #
class F1(BaseClassificationMetric):
    """F1 score computed as 2 * (PPV * TPR) / (PPV + TPR)."""       

    def __init__(self):
        self.code = "F1"
        self.name = r"$F_1$ Score"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, positive=positive, negative=negative)        
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        return 2 * (ppv * tpr) / (ppv + tpr)

class F05(BaseClassificationMetric):
    """F0.5 score computed as (1.25 * PPV * TPR) / (0.25 * PPV + TPR)."""       

    def __init__(self):
        self.code = "F0.5"
        self.name = r"$F_{0.5}$ Score"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, positive=positive, negative=negative)        
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        return (1.25 * ppv * tpr) / (0.25 * ppv + tpr)

class F2(BaseClassificationMetric):
    """F2 score computed as (5 * PPV * TPR) / (4 * PPV + TPR)."""       

    def __init__(self):
        self.code = "F2"
        self.name = r"$F_{2}$ Score"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, positive=positive, negative=negative)        
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        return (5 * ppv * tpr) / (4 * ppv + tpr)

class FBeta(BaseClassificationMetric):
    """FBeta score computed as ((1+beta^2) * PPV * TPR) / ((1+beta^2) * PPV + TPR)."""       

    def __init__(self):
        self.code = "FBeta"
        self.name = r"$F_{\beta}$ Score"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, beta, negative=0, positive=1, *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, positive=positive, negative=negative)        
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        return ((1+beta**2) * ppv * tpr) / ((1+beta**2) * ppv + tpr)

# --------------------------------------------------------------------------- #
class Informedness(BaseClassificationMetric):
    """Informedness is computed as TPR + TNR - 1."""       

    def __init__(self):
        self.code = "INFORM"
        self.name = "Informedness"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        tnr = TrueNegativeRate()(y, y_pred, positive=positive, negative=negative)        
        return tpr + tnr - 1

class Markedness(BaseClassificationMetric):
    """Markedness is computed as PPV + NPV - 1."""       

    def __init__(self):
        self.code = "MARK"
        self.name = "Markedness"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred, negative=0, positive=1, *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, positive=positive, negative=negative)        
        npv = NegativePredictiveValue()(y, y_pred, positive=positive, negative=negative)        
        return ppv + npv - 1

class BalancedAccuracy(BaseClassificationMetric):
    """Balanced accuracy is computed as (TPR + TNR) / 2."""       

    def __init__(self):
        self.code = "BACC"
        self.name = "Balanced Accuracy"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred,negative=0, positive=1, *args, **kwargs):           
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        tnr = TrueNegativeRate()(y, y_pred, positive=positive, negative=negative)        
        return (tpr + tnr) / 2

class GMeasure(BaseClassificationMetric):
    """GMeasure is computed as sqrt(TPR - TNR)."""       

    def __init__(self):
        self.code = "Gm"
        self.name = "G Measure"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred,negative=0, positive=1, *args, **kwargs):           
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        tnr = TrueNegativeRate()(y, y_pred, positive=positive, negative=negative)        
        return np.sqrt(tpr * tnr)

# --------------------------------------------------------------------------- #
class CohensKappa(BaseClassificationMetric):
    """Computes Cohen's Kappa as (ACC-CKc) / (1-CKc)."""

    def __init__(self):
        self.code = "CK"
        self.name = "Cohen's Kappa"
        self.category = "1st Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred,negative=0, positive=1, *args, **kwargs):           
        acc = Accuracy()(y, y_pred, positive=positive, negative=negative)        
        ckc = CohensKappaChance()(y, y_pred, positive=positive, negative=negative)        
        return (acc-ckc) / (1-ckc)

# -------------------------- 2nd LEVEL METRIC ------------------------------- #
class MatthewsCorrelationCoefficient(BaseClassificationMetric):
    """Computes Mathew's Correlation Coefficient as sqrt(INFORM * MARK)."""

    def __init__(self):
        self.code = "MCC"
        self.name = "Matthew's Correlation Coefficient"
        self.category = "2nd Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred,negative=0, positive=1, *args, **kwargs):           
        inform = Informedness()(y, y_pred, positive=positive, negative=negative)        
        mark = Markedness()(y, y_pred, positive=positive, negative=negative)        
        return np.sqrt(inform * mark)

class FowlkesMallowsIndex(BaseClassificationMetric):
    """Computes Fowlkes-Mallows index as sqrt(PPV * TPR)."""

    def __init__(self):
        self.code = "FM"
        self.name = "Fowlkes-Mallows Index"
        self.category = "2nd Level Metric"
        self.mode = 'max'
        self.best = np.max
        self.better = np.greater
        self.worst = -np.Inf
        self.epsilon_factor = 1
        self.probability_metric = False
    
    def __call__(self, y, y_pred,negative=0, positive=1, *args, **kwargs):           
        ppv = PositivePredictiveValue()(y, y_pred, positive=positive, negative=negative)        
        tpr = TruePositiveRate()(y, y_pred, positive=positive, negative=negative)        
        return np.sqrt(ppv * tpr)
