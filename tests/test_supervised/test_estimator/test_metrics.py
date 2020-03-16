#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : test_metrics.py                                                    #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Monday, March 16th 2020, 12:13:19 am                         #
# Last Modified : Monday, March 16th 2020, 12:13:19 am                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
# --------------------------------------------------------------------------- #
#                              TEST METRICS                                   #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pytest
from pytest import mark
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, mean_squared_log_error
from sklearn.metrics import median_absolute_error, r2_score

from mlstudio.supervised.estimator import metrics 

class MetricsTests:

    @mark.metrics
    def test_r2(self, predict_y):
        y, y_pred = predict_y
        x = metrics.R2()(y, y_pred)         
        skl = r2_score(y, y_pred)   
        assert x<=1, "R2 is not less than 1"
        assert np.isclose(x,skl,rtol=1e-1), "R2 not close to sklearn value"

    @mark.metrics
    def test_var_explained(self, predict_y):
        y, y_pred = predict_y
        x = metrics.VarExplained()(y, y_pred)        
        skl = explained_variance_score(y, y_pred)
        assert x<=1, "Variance explained not between 0 and 1"        
        assert np.isclose(x,skl,rtol=1e-1), "Variance explained not close to sklearn value"

    @mark.metrics
    def test_mae(self, predict_y):
        y, y_pred = predict_y
        x = metrics.MAE()(y, y_pred)        
        skl = mean_absolute_error(y, y_pred)
        assert x>0, "MAE is not positive"       
        assert np.isclose(x,skl,rtol=1e-1), "Mean absolute error not close to sklearn value" 

    @mark.metrics
    def test_mse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.MSE()(y, y_pred)        
        skl = mean_squared_error(y, y_pred)
        assert isinstance(x, float), "MSE is not a float"        
        assert x > 0, "MSE is not positive"
        assert np.isclose(x,skl,rtol=1e-1), "Mean squared error not close to sklearn value"

    @mark.metrics
    def test_nmse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.NMSE()(y, y_pred)      
        skl = -1*mean_squared_error(y, y_pred)  
        assert isinstance(x, float), "NMSE is not a float"                
        assert x < 0, "NMSE is not negative"
        assert np.isclose(x,skl,rtol=1e-1), "Negative mean squared error not close to sklearn value"

    @mark.metrics
    def test_rmse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.RMSE()(y, y_pred)      
        skl = mean_squared_error(y, y_pred)  
        assert isinstance(x, float), "RMSE is not a float"                
        assert x > 0, "RMSE is not positive"        
        assert np.isclose(x,np.sqrt(skl),rtol=1e-1), "root mean squared error not close to sklearn value"

    @mark.metrics
    def test_nrmse(self, predict_y):
        y, y_pred = predict_y
        x = metrics.NRMSE()(y, y_pred)       
        skl = mean_squared_error(y, y_pred)   
        assert isinstance(x, float), "NRMSE is not a float"                
        assert x < 0, "NRMSE is not negative"         
        assert np.isclose(x,-np.sqrt(skl),rtol=1e-1), "negative root mean squared error not close to sklearn value"

    # Not testing. Negative predictions.
    # @mark.metrics
    # def test_msle(self, predict_y):
    #     y, y_pred = predict_y
    #     x = metrics.MSLE()(y, y_pred)    
    #     if any(y < 0): print ("Y IS NEGATIVE")    
    #     if any(y_pred < 0): print ("Y_PRED IS NEGATIVE")    
    #     assert isinstance(x, float), "MSLE is not a float"                
    #     assert x > 0, "MSLE is not  positive"                        

    # @mark.metrics
    # def test_rmsle(self, predict_y):
    #     y, y_pred = predict_y
    #     x = metrics.RMSLE()(y, y_pred)        
    #     assert isinstance(x, float), "RMSLE is not a float"                
    #     assert x > 0, "RMSLE is not  positive"           

    @mark.metrics
    def test_medae(self, predict_y):
        y, y_pred = predict_y
        x = metrics.MEDAE()(y, y_pred)        
        skl = median_absolute_error(y, y_pred)
        assert isinstance(x, float), "MEDAE is not a float"                
        assert x > 0, "MEDAE is not  positive"          
        assert np.isclose(x,skl,rtol=1e-1), "Median absolute error not close to sklearn value"
    