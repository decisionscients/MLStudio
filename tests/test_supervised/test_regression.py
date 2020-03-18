#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : test_regression.py                                                 #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Monday, March 16th 2020, 12:31:37 am                         #
# Last Modified : Monday, March 16th 2020, 12:31:37 am                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
# --------------------------------------------------------------------------- #
#                          TEST GRADIENT DESCENT                              #
# --------------------------------------------------------------------------- #
#%%
import math
import numpy as np
import pandas as pd
import pytest
from pytest import mark
import sklearn.linear_model as lm
from sklearn.utils.estimator_checks import parametrize_with_checks
from sklearn.utils.estimator_checks import check_estimator

from mlstudio.supervised.regression import LinearRegression
from mlstudio.supervised.regression import LassoRegression
from mlstudio.supervised.regression import RidgeRegression
from mlstudio.supervised.regression import ElasticNetRegression


from mlstudio.supervised.estimator.callbacks import Callback
from mlstudio.supervised.estimator.cost import Cost, Quadratic, BinaryCrossEntropy
from mlstudio.supervised.estimator.cost import CategoricalCrossEntropy
from mlstudio.supervised.estimator.scorers import Metric
from mlstudio.supervised.estimator.early_stop import EarlyStop

@mark.regression
@mark.regression_sklearn
@parametrize_with_checks([LinearRegression(metric='r2'),
                          LassoRegression(metric='r2'),
                          RidgeRegression(metric='r2'),
                          ElasticNetRegression(metric='r2')])
def test_sklearn_compatible_estimator(estimator, check):
    check(estimator)


class RegressionTests:

    @mark.regression
    @mark.regression_val
    def test_regression_validation(self, regression, get_regression_data):

        X, y = get_regression_data
        with pytest.raises(TypeError):
            est = regression(learning_rate="x")
            est.fit(X, y)        
        with pytest.raises(TypeError):
            est = regression(batch_size='k')            
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = regression(theta_init='k')            
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = regression(epochs='k')           
            est.fit(X, y)
        with pytest.raises(ValueError):
            est = regression(cost='x')                                
            est.fit(X, y) 
        with pytest.raises(ValueError):
            est = regression(cost=None)                                
            est.fit(X, y)                  
        with pytest.raises(TypeError):
            est = regression(early_stop='x')                                
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = regression(metric=0)                                
            est.fit(X, y)             
        with pytest.raises(ValueError):
            est = regression(metric='x')                                
            est.fit(X, y) 
        with pytest.raises(TypeError):
            est = regression(epochs=10,verbose=None)                                                                                      
            est.fit(X, y)
        with pytest.raises(ValueError):
            est = regression(epochs=10,checkpoint=-1)
            est.fit(X, y)
        with pytest.raises(TypeError):
            est = regression(epochs=10,checkpoint='x')
            est.fit(X, y)  
        with pytest.raises(TypeError):
            est = regression(epochs=10,random_state='k')                                
            est.fit(X, y)
                       

    @mark.regression
    @mark.regression_get_params
    def test_regression_get_params(self, regression):
        est = regression(learning_rate=0.01, theta_init=np.array([2,2,2]),
                             epochs=10, cost='quadratic', 
                             verbose=False, checkpoint=100, 
                             name=None, random_state=50)
        params = est.get_params()
        assert params['learning_rate'] == 0.01, "learning rate is invalid" 
        assert all(np.equal(params['theta_init'], np.array([2,2,2]))) , "theta_init is invalid"
        assert params['epochs'] == 10, "epochs is invalid"        
        assert params['verbose'] == False, "verbose is invalid"

    @mark.regression
    @mark.regression_validate_data
    def test_regression_validate_data(self, regression, get_regression_data):
        est = regression(epochs=10)
        X, y = get_regression_data        
        with pytest.raises(ValueError):
            est.fit([1,2,3], y)
        with pytest.raises(ValueError):
            est.fit(X, [1,2,3])            
        with pytest.raises(ValueError):
            est.fit(X, y[0:5])            

    @mark.regression
    @mark.regression_init_weights
    def test_regression_init_weights_shape_mismatch(self, regression, get_regression_data):        
        X, y = get_regression_data        
        theta_init = np.ones(X.shape[1])        
        est = regression(epochs=10, theta_init=theta_init)
        with pytest.raises(ValueError):
            est.fit(X,y)

    @mark.regression
    @mark.regression_results
    def test_regression_results(self, split_regression_data):
        X, X_test, y, y_test = split_regression_data
        est = LinearRegression(gradient_descent=False)            
        est.fit(X,y)
        score1 = est.score(X_test, y_test)
        print("----------------------------")
        print(score1)
        est = LinearRegression(epochs=5000)            
        est.fit(X,y)        
        score2 = est.score(X_test, y_test)        
        print("----------------------------")
        print(score2)    
        est = lm.LinearRegression()            
        est.fit(X,y)
        y_pred = est.predict(X_test)
        score3 = np.mean(y_test - y_pred)**2
        print("----------------------------")
        print(score3)        
        assert abs(score1-score2)/score2 < 0.05, "Scores 1 and 2 are not close"  
        assert abs(score2-score3)/score3 < 0.05, "Scores 1 and 2 are not close"  


    @mark.regression
    @mark.regression_rate
    def test_regression_fit_rate_constant(self, regression, get_regression_data):
        X, y = get_regression_data        
        est = regression(learning_rate = 0.1, epochs=10)
        est.fit(X,y)
        assert est.learning_rate == 0.1, "learning rate not initialized correctly"
        assert est.history_.epoch_log['learning_rate'][0]==\
            est.history_.epoch_log['learning_rate'][-1], "learning rate not constant in history"

    @mark.regression
    @mark.regression_batch_size
    def test_regression_fit_batch_size(self, regression, get_regression_data):
        X, y = get_regression_data         
        X = X[0:33]
        y = y[0:33]       
        est = regression(batch_size=32, epochs=10, val_size=0)
        est.fit(X,y)                
        assert est.history_.total_epochs == 10, "total epochs in history not correct"
        assert est.history_.total_batches == 20, "total batches in history not correct"
        assert est.history_.total_epochs != est.history_.total_batches, "batches and epochs are equal"
        assert est.history_.batch_log['batch_size'][0]==32, "batch size not correct in history"
        assert est.history_.batch_log['batch_size'][1]!=32, "batch size not correct in history"
        assert len(est.history_.batch_log['batch_size']) ==20, "length of batch log incorrect"
        assert len(est.history_.epoch_log['learning_rate'])==10, "length of epoch log incorrect"


    @mark.regression
    @mark.regression_epochs
    def test_regression_fit_epochs(self, regression, get_regression_data):
        X, y = get_regression_data                
        est = regression(epochs=10)
        est.fit(X,y)
        assert est.epochs == 10, "regression epochs invalid"
        assert est.history_.total_epochs == 10, "total epochs in history not valid"
        assert len(est.history_.epoch_log['learning_rate']) == 10, "epoch log not equal to epochs"

    
    @mark.regression
    @mark.regression_early_stop
    @mark.regression_early_stop_from_estimator
    def test_regression_fit_early_stop_from_estimator_val_score(self, regression, get_regression_data):        
        X, y = get_regression_data                
        est = regression(learning_rate=0.5, epochs=5000, early_stop=True, val_size=0.3)
        est.fit(X,y)
        assert est.history_.total_epochs < 5000, "didn't stop early"
        assert len(est.history_.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"        

    @mark.regression
    @mark.regression_early_stop
    @mark.regression_early_stop_from_estimator
    def test_regression_fit_early_stop_from_estimator_val_cost(self, regression, get_regression_data):        
        X, y = get_regression_data                
        est = regression(learning_rate=0.5, epochs=5000, early_stop=True, val_size=0.3, metric=None)
        est.fit(X,y)
        assert est.history_.total_epochs < 5000, "didn't stop early"
        assert len(est.history_.epoch_log['learning_rate']) < 5000, "epoch log too long for early stop"        


    @mark.regression
    @mark.regression_history
    def test_regression_history_no_val_data_no_metric(self, regression, get_regression_data):        
        X, y = get_regression_data        
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

    @mark.regression
    @mark.regression_history
    def test_regression_history_w_val_data_and_metric(self, regression, get_regression_data):        
        X, y = get_regression_data  
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

    @mark.regression
    @mark.regression_history
    def test_regression_history_no_val_data_w_metric(self, regression, get_regression_data):        
        X, y = get_regression_data        
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

    @mark.regression
    @mark.regression_history
    def test_regression_history_w_val_data_w_metric(self, regression, get_regression_data):        
        X, y = get_regression_data     
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

    @mark.regression
    @mark.linear_regression
    def test_linear_regression_name(self, get_regression_data):        
        X, y = get_regression_data
        est = LinearRegression()
        est.fit(X,y)
        assert est.description == "Linear Regression with Batch Gradient Descent", "incorrect name"

    @mark.regression
    @mark.linear_regression
    @mark.linear_regression_validation
    def test_linear_regression_validation(self, get_regression_data):        
        X, y = get_regression_data        
        with pytest.raises(ValueError):
            est = LinearRegression(metric='accuracy')
            est.fit(X, y)  
        with pytest.raises(ValueError):
            est = LinearRegression(cost='binary_cross_entropy')
            est.fit(X, y)              

    @mark.regression
    @mark.linear_regression
    @mark.linear_regression_predict
    def test_regression_predict(self, regression, get_regression_data):
        X, y = get_regression_data                
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

    @mark.regression
    @mark.regression_score
    def test_regression_score(self, regression, get_regression_data_w_validation, 
                                    regression_metric):
        X, X_test, y, y_test = get_regression_data_w_validation                
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

    @mark.regression
    @mark.lasso_regression
    def test_lasso_regression_name(self, get_regression_data):   
        X, y = get_regression_data     
        est = LassoRegression()
        est.fit(X,y)
        assert est.description == "Lasso Regression with Batch Gradient Descent", "incorrect name"

    @mark.regression
    @mark.ridge_regression
    def test_ridge_regression_name(self, get_regression_data): 
        X, y = get_regression_data       
        est = RidgeRegression()
        est.fit(X,y)
        assert est.description == "Ridge Regression with Batch Gradient Descent", "incorrect name"        

    @mark.regression
    @mark.elasticnet_regression
    def test_elasticnet_regression_name(self, get_regression_data):  
        X, y = get_regression_data      
        est = ElasticNetRegression()
        est.fit(X,y)
        assert est.description == "Elastic Net Regression with Batch Gradient Descent", "incorrect name"        