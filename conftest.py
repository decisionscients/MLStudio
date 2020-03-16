#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : conftest.py                                                        #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 10:02:50 pm                         #
# Last Modified : Sunday, March 15th 2020, 10:02:50 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
# %%
import numpy as np
import pandas as pd
from pytest import fixture

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from mlstudio.supervised.regression import LinearRegression
from mlstudio.supervised.regression import LassoRegression
from mlstudio.supervised.regression import RidgeRegression
from mlstudio.supervised.regression import ElasticNetRegression

from mlstudio.supervised.estimator.cost import RegressionCostFactory
from mlstudio.supervised.estimator.metrics import RegressionMetricFactory

# ============================================================================ #

@fixture(scope="session")
def get_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                            random_state=50)
    return X_train, X_test, y_train, y_test

# ============================================================================ #
#                                 COST                                         #
# ============================================================================ #
@fixture(scope='class')
def get_quadratic_y():
    return np.array([2257, 4744, 7040, 5488, 9755, 7435, 3812, 5296, 7300, 7041])


@fixture(scope='class')
def get_quadratic_y_pred():
    return np.array([8306, 6811, 1125, 4265, 1618, 3128, 2614, 2767, 3941, 4499])


@fixture(scope='class')
def get_quadratic_X():
    filename = "tests/test_data/test_quadratic_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',
                       usecols=[2, 3, 4], skipfooter=2)
    X = df.values
    return X


@fixture(scope='class')
def get_quadratic_cost():
    return 9384127.6


@fixture(scope='class')
def get_quadratic_gradient():
    return np.array([-2109.4, -113092063.3, -34441317.9])

@fixture(scope='class')
def get_binary_cost_X():
    filename = "tests/test_data/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[2, 3, 4])
    X = df.values
    return X


@fixture(scope='class')
def get_binary_cost_y():
    filename = "tests/test_data/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[0])
    y = df.values
    return y


@fixture(scope='class')
def get_binary_cost_y_pred():
    filename = "tests/test_data/test_binary_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[1])
    y_pred = df.values
    return y_pred


@fixture(scope='class')
def get_binary_cost():
    return 0.345424815


@fixture(scope='class')
def get_binary_cost_gradient():
    return np.array([-1.556243917, -960.1098781, -1342.758965])


@fixture(scope='class')
def get_categorical_cost_X():
    filename = "tests/test_data/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[12, 13, 14])
    X = df.values
    return X


@fixture(scope='class')
def get_categorical_cost_y():
    filename = "tests/test_data/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[0, 1, 2])
    y = df.values
    return y


@fixture(scope='class')
def get_categorical_cost_y_pred():
    filename = "tests/test_data/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1', usecols=[3, 4, 5])
    y_pred = df.values
    return y_pred


@fixture(scope='class')
def get_categorical_cost():
    return 0.367654163


@fixture(scope='class')
def get_categorical_cost_gradient():
    filename = "tests/test_data/test_categorical_cost.xlsx"
    df = pd.read_excel(io=filename, sheet_name='Sheet1',
                       usecols=[26, 27, 28], skipfooter=7)
    y_grad = df.values
    return y_grad

# ============================================================================ #
#                            EARLY STOP                                        #
# ============================================================================ #
@fixture(scope='session', params=['mae',
                                  'mse',
                                  'rmse',
                                  'mae',
                                  'r2',
                                  'var_explained',
                                  'nmse',
                                  'nrmse'])                                  
def models_by_metric(request):
    model = LinearRegression(metric=request.param)
    model.cost_function = RegressionCostFactory()(cost='quadratic')
    model.scorer = RegressionMetricFactory()(metric=request.param)    
    return model 

@fixture(scope='session', params=['train_cost',
                                  'train_score',
                                  'val_cost',
                                  'val_score'])
def early_stop_monitor(request):
    return request.param    

@fixture(scope='session', params=['mae',
                                  'mse',
                                  'rmse',
                                  'medae'])                                  
def model_lower_is_better(request):
    model = LinearRegression(metric=request.param, early_stop=True,
                            val_size=0.3, precision=0.1,
                            patience=2)
    model.cost_function = RegressionCostFactory()(cost='quadratic')
    model.scorer = RegressionMetricFactory()(metric=request.param)                            
    return model

@fixture(scope='session', params=['r2',
                                  'var_explained',
                                  'nmse',
                                  'nrmse'])                                  
def model_higher_is_better(request):
    model = LinearRegression(metric=request.param, early_stop=True,
                            val_size=0.3, precision=0.1,
                            patience=2)
    model.cost_function = RegressionCostFactory()(cost='quadratic')
    model.scorer = RegressionMetricFactory()(metric=request.param)                            
    return model    

# ============================================================================ #
#                              METRICS                                         #
# ============================================================================ #  

@fixture(scope='session')
def predict_y():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    gd = LinearRegression(epochs=5)
    gd.fit(X, y)
    y_pred = gd.predict(X)
    return y, y_pred    

# ============================================================================ #
#                               REGRESSION                                     #
# ============================================================================ #  
@fixture(scope='session', params=[LinearRegression,
                                   LassoRegression,
                                   RidgeRegression,
                                   ElasticNetRegression])
def regression(request):
    return request.param    

@fixture(scope="session")
def get_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y    

@fixture(scope="session")
def get_regression_data_w_validation(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test    

@fixture(scope='session', params=['r2',
                                  'var_explained',
                                  'mae',
                                  'mse',
                                  'nmse',
                                  'rmse',
                                  'nrmse',
                                  'medae'])
def regression_metric(request):
    return request.param    