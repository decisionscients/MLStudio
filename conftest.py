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

# ============================================================================ #
#                                  DATA                                        #
# ============================================================================ #  
@fixture(scope="session")
def get_regression_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y

@fixture(scope="session")
def get_regression_data_features():
    data = datasets.load_boston()
    return data['feature_names']

@fixture(scope="session")
def get_regression_data_split(get_regression_data):
    X, y = get_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test        


@fixture(scope="session")
def get_logistic_regression_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y   

@fixture(scope="session")
def get_logistic_regression_split_data():
    X, y = datasets.load_breast_cancer(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test                

@fixture(scope="session")
def get_logistic_regression_data_features():
    data = datasets.load_breast_cancer()
    return data['feature_names']

@fixture(scope="session")
def get_softmax_regression_split_data():
    X, y = datasets.load_iris(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test                

@fixture(scope="session")
def get_softmax_regression_data_features():
    data = datasets.load_iris()
    return data['feature_names']    