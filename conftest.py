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
import pytest
from pytest import fixture

from sklearn import datasets
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from mlstudio.supervised.core.optimizers import Adagrad
from mlstudio.supervised.core.objectives import StyblinskiTank
from mlstudio.supervised.callbacks.learning_rate import TimeDecay
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescent
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentRegressor
from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentClassifier

# ============================================================================ #
#                               FILES TO SKIP                                  #
# ============================================================================ #
collect_ignore_glob = ["/test_visual/test_model*.py"]

# ============================================================================ #
#                                  DATA                                        #
# ============================================================================ #  
@fixture(scope="session")
def get_target_2d_vector():
    y = np.array([0,1,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,0,0], ndmin=2).reshape(-1,1)
    return y

@fixture(scope="session")
def get_binary_target_numeric():
    y = np.array([0,1,1,0,1,0,1,0,0,0,1,0,1,1,0,1,0,1,0,0])
    return y

@fixture(scope="session")
def get_binary_target_char():
    y = np.array(["blue","green","green","blue","green","blue","green","blue","blue","blue","green","blue","green","green","blue","green","blue","green","blue","blue"])
    return y

@fixture(scope="session")
def get_multiclass_target_num():
    y = np.array([0,1,1,0,1,0,1,2,0,0,1,0,9,1,17,1,0,1,0,0])
    return y    

@fixture(scope="session")
def get_multiclass_target_char():
    y = np.array(["blue", "red", "green","green","blue","green","blue","green","blue","blue","blue","green","blue","green","green","blue","green","blue","green","blue","blue","blue","green"])
    return y    

@fixture(scope="session")
def get_one_hot():
    y = np.array([[0,0,0,1],
             [0,0,1,0],
             [0,1,0,0],
             [1,0,0,0]])
    return y

@fixture(scope="session")
def get_multilabel_numeric():
    y = np.arange(0,100)
    y = y.reshape(20,-1)
    return y

@fixture(scope="session")
def get_multilabel_char():
    labels = np.array(["red", "green", "blue", "orange"])
    y = np.random.choice(a=labels, size=(20,5))
    return y

@fixture(scope="session")
def get_data_management_data(get_target_2d_vector,
                             get_binary_target_numeric,
                             get_binary_target_char,
                             get_multiclass_target_num,
                             get_multiclass_target_char,
                             get_one_hot,
                             get_multilabel_numeric,
                             get_multilabel_char):
    d = {}
    d['target_2d_vector'] = get_target_2d_vector
    d['binary_target_numeric'] = get_binary_target_numeric
    d['binary_target_char'] = get_binary_target_char
    d['multiclass_target_num'] = get_multiclass_target_num
    d['multiclass_target_char'] = get_multiclass_target_char
    d['one_hot'] = get_one_hot
    return d

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
def get_softmax_regression_data():
    X, y = datasets.load_iris(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
    return X, y

@fixture(scope="session")
def get_softmax_regression_split_data(get_softmax_regression_data):
    X, y = get_softmax_regression_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    return X_train, X_test, y_train, y_test                

@fixture(scope="session")
def get_softmax_regression_data_features():
    data = datasets.load_iris()
    return data['feature_names']    

@fixture(scope="session")
def get_regression_prediction(get_regression_data):
    X, y = get_regression_data
    gd = GradientDescentRegressor(algorithm=LinearRegression(),
                                  epochs=4000)
    gd.fit(X,y)
    y_pred = gd.predict(X)    
    return y, y_pred

@fixture(scope="session")
def get_logistic_regression_prediction(get_logistic_regression_data):
    X, y = get_logistic_regression_data
    gd = GradientDescentClassifier(algorithm=LogisticRegression())
    gd.fit(X,y)
    y_pred = gd.predict(X)    
    return y, y_pred    

@fixture(scope="session")
def get_softmax_regression_prediction(get_softmax_regression_data):
    X, y = get_softmax_regression_data
    gd = GradientDescentClassifier(algorithm=SoftmaxRegression())
    gd.fit(X,y)
    y_pred = gd.predict(X)    
    return y, y_pred       

@fixture(scope="session")
def get_estimator():
    objective = StyblinskiTank()
    optimizer = Adagrad()
    est = GradientDescent(learning_rate=0.1,
                          theta_init=objective.start, 
                          epochs=500, objective=objective,
                          schedule=TimeDecay(decay_factor=0.9),
                          optimizer=optimizer)
    est.fit()    
    return est

@fixture(scope='session')
def get_log():
    filepath = "tests/test_data/test_monitor.csv"
    df = pd.read_csv(filepath)
    log = []
    for idx, row in df.iterrows():
        epoch_log = {'epoch': row['epoch'],
                    'train_cost': row['train_cost'],
                    'train_score': row['train_score'],
                    'val_cost': row['val_cost'],
                    'val_score': row['val_score'],
                    'theta_norm': row['theta_norm'],
                    'gradient_norm': row['gradient_norm']                    
                    }
        log.append(epoch_log)
    log = np.array(log)
    return log




