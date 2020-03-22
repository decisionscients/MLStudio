#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : test_logistic_regression.py                                       #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, March 21st 2020, 4:32:52 pm                       #
# Last Modified : Saturday, March 21st 2020, 4:33:59 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
""" Tests Logistic Regression """ 
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark
import sklearn.linear_model as lm
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator

from mlstudio.supervised.classification import LogisticRegression


from mlstudio.supervised.estimator.callbacks import Callback
from mlstudio.supervised.estimator.cost import Cost, BinaryClassification
from mlstudio.supervised.estimator.cost import MultiClassification
from mlstudio.supervised.estimator.scorers import Metric
from mlstudio.supervised.estimator.early_stop import EarlyStop

@mark.logistic
@mark.logistic_sklearn
@parametrize_with_checks([LogisticRegression(metric='r2'),
                          LogisticRegression(batch_size=1),
                          LogisticRegression(early_stop=True, val_size=0.3)])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


class LogisticRegressionTests:

    @mark.logistic
    @mark.logistic_results
    def test_logistic_regression_results(self, split_regression_data):
        X, X_test, y, y_test = split_regression_data
        est = LogisticRegression(gradient_descent=False)            
        est.fit(X,y)
        score1 = est.score(X_test, y_test)
        print("----------------------------")
        print(score1)
        est = LogisticRegression(epochs=5000)            
        est.fit(X,y)        
        score2 = est.score(X_test, y_test)        
        print("----------------------------")
        print(score2)    
        est = lm.LogisticRegression()            
        est.fit(X,y)
        y_pred = est.predict(X_test)
        score3 = np.mean(y_test - y_pred)**2
        print("----------------------------")
        print(score3)        
        assert abs(score1-score2)/score2 < 0.05, "Scores 1 and 2 are not close"  
        assert abs(score2-score3)/score3 < 0.05, "Scores 1 and 2 are not close"  

    
    @mark.logistic
    @mark.logistic_early_stop
    @mark.logistic_early_stop_from_estimator
    def test_logistic_regression_fit_early_stop_from_estimator_val_score(self, regression, get_logistic_regression_data):        
        X, y = get_logistic_regression_data                
        est = regression(learning_rate=0.5, epochs=5000, early_stop=True, val_size=0.3)
        est.fit(X,y)
        assert est.history_.total_epochs < 5000, "didn't stop early"
        assert len(est.history_.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"        

    @mark.logistic
    @mark.logistic_early_stop
    @mark.logistic_early_stop_from_estimator
    def test_logistic_regression_fit_early_stop_from_estimator_val_cost(self, regression, get_logistic_regression_data):        
        X, y = get_logistic_regression_data                
        est = regression(learning_rate=0.5, epochs=5000, early_stop=True, val_size=0.3, metric=None)
        est.fit(X,y)
        assert est.history_.total_epochs < 5000, "didn't stop early"
        assert len(est.history_.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"        


    @mark.logistic
    @mark.logistic_history
    def test_logistic_regression_history_no_val_data_no_metric(self, regression, get_logistic_regression_data):        
        X, y = get_logistic_regression_data        
        est = regression(epochs=10, metric=None, val_size=0)
        est.fit(X, y)        
        # Test epoch history
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert est.history_.epoch_log.get('train_cost')[0] > est.history_.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history_.epoch_log.get("train_score", None) is None, "train score without metric is not None"
        assert est.history_.epoch_log.get("val_cost", None) is None, "val cost without early stopping is not None"
        assert est.history_.epoch_log.get("val_score", None) is None, "val score without early stopping is not None"
        # Test batch history
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.logistic
    @mark.logistic_history
    def test_logistic_regression_history_w_val_data_and_metric(self, regression, get_logistic_regression_data):        
        X, y = get_logistic_regression_data  
        stop = EarlyStop()
        est = regression(epochs=10, learning_rate=0.001, val_size=0.3, 
                         early_stop=stop, metric='nrmse')
        est.fit(X, y)        
        # Test epoch history
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('val_cost')), "number of val costs in log doesn't match epochs"        
        assert est.history_.epoch_log.get('train_cost')[0] > est.history_.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history_.epoch_log.get('train_score')[0] < est.history_.epoch_log.get('train_score')[-1], "train_score did not improve"
        assert est.history_.epoch_log.get('val_score')[0] < est.history_.epoch_log.get('val_score')[-1], "val_score did not improve"
        # Test batch history
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.logistic
    @mark.logistic_history
    def test_logistic_regression_history_no_val_data_w_metric(self, regression, get_logistic_regression_data):        
        X, y = get_logistic_regression_data        
        est = regression(epochs=10, metric='mse', val_size=0)
        est.fit(X, y)        
        # Test epoch history
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('train_score')), "number of train scores in log doesn't match epochs"        
        assert est.history_.epoch_log.get('train_cost')[0] > est.history_.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history_.epoch_log.get('train_score')[0] > est.history_.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert est.history_.epoch_log.get("val_cost", None) is None, "val cost without early stopping is not None"
        assert est.history_.epoch_log.get("val_score", None) is None, "val score without early stopping is not None"
        # Test batch history
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.logistic
    @mark.logistic_history
    def test_logistic_regression_history_w_val_data_w_metric(self, regression, get_logistic_regression_data):        
        X, y = get_logistic_regression_data     
        stop = EarlyStop()   
        est = regression(epochs=10, learning_rate=0.001, val_size=0.3, metric='mse', early_stop=stop)
        est.fit(X, y)        
        # Test epoch history
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('epoch')), "number of epochs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('learning_rate')), "number of learning rates in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('theta')), "number of thetas in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('train_cost')), "number of train costs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('val_cost')), "number of val costs in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('train_score')), "number of train score in log doesn't match epochs"        
        assert est.history_.total_epochs == len(est.history_.epoch_log.get('val_score')), "number of val score in log doesn't match epochs"        
        assert est.history_.epoch_log.get('train_cost')[0] > est.history_.epoch_log.get('train_cost')[-1], "train_cost does not decrease"
        assert est.history_.epoch_log.get('train_score')[0] > est.history_.epoch_log.get('train_score')[-1], "train_score does not decrease"
        assert est.history_.epoch_log.get('val_cost')[0] > est.history_.epoch_log.get('val_cost')[-1], "val_cost does not decrease"
        assert est.history_.epoch_log.get('val_score')[0] > est.history_.epoch_log.get('val_score')[-1], "val_score does not decrease"        
        # Test batch history
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch')), "number of batches in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('batch_size')), "number of batch sizes in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('theta')), "number of thetas in log doesn't match total batches"        
        assert est.history_.total_batches == len(est.history_.batch_log.get('train_cost')), "number of train_costs in log doesn't match total batches"        

    @mark.logistic
    @mark.linear_regression
    def test_linear_regression_name(self, get_logistic_regression_data):        
        X, y = get_logistic_regression_data
        est = LogisticRegression()
        est.fit(X,y)
        assert est.description == "Logistic Regression with Batch Gradient Descent", "incorrect name"

    @mark.logistic
    @mark.linear_regression
    @mark.linear_regression_validation
    def test_linear_regression_validation(self, get_logistic_regression_data):        
        X, y = get_logistic_regression_data        
        with pytest.raises(ValueError):
            est = LogisticRegression(metric='mse')
            est.fit(X, y)  
        with pytest.raises(ValueError):
            est = LogisticRegression(cost='quadratic')
            est.fit(X, y)              

    @mark.logistic
    @mark.linear_regression
    @mark.linear_regression_predict
    def test_logistic_regression_predict(self, regression, get_logistic_regression_data):
        X, y = get_logistic_regression_data                
        est = regression(learning_rate = 0.1, epochs=1000)
        with pytest.raises(Exception): # Tests predict w/o first fitting model
            y_pred = est.predict(X)        
        with pytest.raises(ValueError):
            est.fit(X,y)
            y_pred = est.predict([1,2,3])
        with pytest.raises(Exception):            
            y_pred = est.predict(np.reshape(X, (-1,1)))        
        est.fit(X,y)
        y_pred = est.predict(X)
        assert all(np.equal(y.shape, y_pred.shape)), "y and y_pred have different shapes"  
        assert np.array_equal(y_pred, y_pred.astype(bool)), "Predictions not equal to zero or one."

    @mark.logistic
    @mark.logistic_score
    def test_logistic_regression_score(self, regression, get_logistic_regression_data_w_validation, 
                                    regression_metric):
        X, X_test, y, y_test = get_logistic_regression_data_w_validation                
        est = regression(learning_rate = 0.1, epochs=1000, metric=regression_metric)
        with pytest.raises(Exception):
            score = est.score(X, y)
        est.fit(X, y)
        with pytest.raises(ValueError):
            score = est.score(X, [1,2,3])        
        with pytest.raises(ValueError):
            score = est.score(X, np.array([1,2,3]))        
        with pytest.raises(ValueError):
            score = est.score(np.reshape(X, (-1,1)), y)    
        # Model evaluation 
        score = est.score(X_test, y_test)
        assert isinstance(score, (int,float)), "score is not an int nor a float"  

   