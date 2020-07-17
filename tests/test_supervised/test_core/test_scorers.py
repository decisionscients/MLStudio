#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : test_scorers.py                                                   #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, March 16th 2020, 12:13:19 am                        #
# Last Modified : Wednesday, March 18th 2020, 2:22:39 pm                      #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests Scorer classes."""
#%%
import math
import numpy as np
import pytest
from pytest import mark
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score
from sklearn.metrics import accuracy_score, auc, roc_auc_score, roc_curve

from mlstudio.supervised.algorithms.optimization.services import scorers 

@mark.scorers
@mark.classification_scorers
class ClassificationMetricsTests:

    def test_precision(self, get_classification_metric_test_package):
        d = get_classification_metric_test_package        
        scorer = scorers.Precision()
        score = scorer(d['y'], d['y_pred'])
        assert np.isclose([score], d['precision']), "Precision score not correct." 

    def test_specificity(self, get_classification_metric_test_package):
        d = get_classification_metric_test_package        
        scorer = scorers.Recall()
        score = scorer(d['y'], d['y_pred'])
        assert np.isclose([score], d['specificity']), "Recall score not correct."         

    def test_specificity(self, get_classification_metric_test_package):
        d = get_classification_metric_test_package        
        scorer = scorers.Specificity()
        score = scorer(d['y'], d['y_pred'])
        assert np.isclose([score], d['specificity']), "Specificity score not correct."                 

    def test_f1(self, get_classification_metric_test_package):
        d = get_classification_metric_test_package        
        scorer = scorers.F1()
        score = scorer(d['y'], d['y_pred'])
        assert np.isclose([score], d['f1']), "F1 score not correct."                 

    def test_balanced_accuracy(self, get_classification_metric_test_package):
        d = get_classification_metric_test_package        
        scorer = scorers.BalancedAccuracy()
        score = scorer(d['y'], d['y_pred'])
        assert np.isclose([score], d['balanced_accuracy']), "Balanced accuracy score not correct."                 

class RegressionMetricsTests:

    @mark.scorers
    def test_r2(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.R2()(y, y_pred)         
        skl = r2_score(y, y_pred)   
        assert x<=1, "R2 is not less than 1"
        assert np.isclose(x,skl,rtol=1e-2), "R2 not close to sklearn value"

    @mark.scorers
    def test_var_explained(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.VarExplained()(y, y_pred)        
        skl = explained_variance_score(y, y_pred)
        assert x<=1, "Variance explained not between 0 and 1"        
        assert np.isclose(x,skl,rtol=1e-2), "Variance explained not close to sklearn value"

    @mark.scorers
    def test_mae(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.MAE()(y, y_pred)        
        skl = mean_absolute_error(y, y_pred)
        assert x>0, "MAE is not positive"       
        assert np.isclose(x,skl,rtol=1e-2), "Mean absolute error not close to sklearn value" 

    @mark.scorers
    def test_mse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.MSE()(y, y_pred)        
        skl = mean_squared_error(y, y_pred)
        assert isinstance(x, float), "MSE is not a float"        
        assert x > 0, "MSE is not positive"
        assert np.isclose(x,skl,rtol=1e-2), "Mean squared error not close to sklearn value"

    @mark.scorers
    def test_nmse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.NMSE()(y, y_pred)      
        skl = -1*mean_squared_error(y, y_pred)  
        assert isinstance(x, float), "NMSE is not a float"                
        assert x < 0, "NMSE is not negative"
        assert np.isclose(x,skl,rtol=1e-2), "Negative mean squared error not close to sklearn value"

    @mark.scorers
    def test_rmse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.RMSE()(y, y_pred)      
        skl = mean_squared_error(y, y_pred)  
        assert isinstance(x, float), "RMSE is not a float"                
        assert x > 0, "RMSE is not positive"        
        assert np.isclose(x,np.sqrt(skl),rtol=1e-2), "root mean squared error not close to sklearn value"

    @mark.scorers
    def test_nrmse(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.NRMSE()(y, y_pred)       
        skl = mean_squared_error(y, y_pred)   
        assert isinstance(x, float), "NRMSE is not a float"                
        assert x < 0, "NRMSE is not negative"         
        assert np.isclose(x,-np.sqrt(skl),rtol=1e-2), "negative root mean squared error not close to sklearn value"

    @mark.scorers
    def test_msle(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.MSLE()(y, y_pred) 
        if all(y_pred > 0) and (y > 0):
            skl = mean_squared_log_error(y, y_pred)  
            assert x > 0, "MSLE is not  positive"
            assert np.isclose(x,skl,rtol=1e-2), "Mean squared log error not close to sklearn value" 
        else:
            print("\nUnable to compute MSLE with negative targets.")                               


    @mark.scorers
    def test_rmsle(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.RMSLE()(y, y_pred)        
        if all(y_pred > 0) and (y > 0):
            skl = np.sqrt(mean_squared_log_error(y, y_pred))
            assert x > 0, "RMSLE is not  positive"
            assert np.isclose(x,skl,rtol=1e-2), "Root mean squared log error not close to sklearn value" 
        else:
            print("\nUnable to compute RMSLE with negative targets.")                               

    @mark.scorers
    def test_medae(self, get_regression_prediction):
        X, y, y_pred = get_regression_prediction
        x = scorers.MEDAE()(y, y_pred)        
        skl = median_absolute_error(y, y_pred)
        assert isinstance(x, float), "MEDAE is not a float"                
        assert x > 0, "MEDAE is not  positive"          
        assert np.isclose(x,skl,rtol=1e-2), "Median absolute error not close to sklearn value"

class ClassificationMetricsTests:

    @mark.scorers
    def test_accuracy(self, get_logistic_regression_prediction):
        X, y, y_pred = get_logistic_regression_prediction
        x = scorers.Accuracy()(y, y_pred)         
        skl = accuracy_score(y, y_pred)   
        assert x<=1, "R2 is not less than 1"
        assert np.isclose(x,skl,rtol=1e-2), "R2 not close to sklearn value"        
    