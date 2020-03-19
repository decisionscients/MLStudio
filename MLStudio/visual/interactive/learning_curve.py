#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : learning_curve.py                                                 #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, March 19th 2020, 4:05:30 am                       #
# Last Modified : Thursday, March 19th 2020, 4:41:16 am                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
""" Interactive visualizations of regression based optimizations."""
#%%
import os
from pathlib import Path
import platform
import psutil
import site
PROJECT_DIR = Path(__file__).resolve().parents[3]
site.addsitedir(PROJECT_DIR)

import numpy as np
import pandas as pd

from ipywidgets import widgets
import plotly.graph_objects as go
from sklearn import datasets
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler

from mlstudio.supervised.regression import LinearRegression
from mlstudio.supervised.regression import LassoRegression
from mlstudio.supervised.regression import RidgeRegression
from mlstudio.supervised.regression import ElasticNetRegression

# =========================================================================== #
#                           LEARNING CURVE                                    #
# =========================================================================== #
# --------------------------------------------------------------------------- #
#                               Estimators                                    #
# --------------------------------------------------------------------------- #
estimators = {'linear_regression': LinearRegression(),
              'lasso_regression': LassoRegression(),
              'ridge_regression': RidgeRegression(),
              'elasticnet_regression': ElasticNetRegression()}
#%%              
# --------------------------------------------------------------------------- #
#                               Data                                          #
# --------------------------------------------------------------------------- #
def get_data():
    X, y = datasets.load_boston(return_X_y=True)
    scaler = StandardScaler()    
    X = scaler.fit_transform(X)
return X, y

# --------------------------------------------------------------------------- #
#                               Widgets                                       #
# --------------------------------------------------------------------------- #
estimator=widgets.Dropdown(
    options=[('Linear Regression', LinearRegression),
             ('Lasso Regression', LassoRegression),
             ('Ridge Regression', RidgeRegression), 
             ('Elastic Net Regression', ElasticNetRegression)],
    value=LinearRegression,
    description='Algorithm:'
)
training_size_min=widgets.IntText(
    value=0.1,
    description="Training Set Size Minimimum (Log):",
    disabled=False
)
training_size_steps=widgets.IntText(
    value=10,
    description="Training Set Size Steps:",
    disabled=False
)
training_size_max=widgets.IntText(
    value=1,
    description="Training Set Size Maximum (Log):",
    disabled=False
)
learning_rate = widgets.FloatLogSlider(
    value=-2,
    base=10,
    min=-3,
    max=-1,
    step=0.2,
    description="Learning Rate"
)
epochs = widgets.FloatLogSlider(
    value=2,
    min=2,
    max=4,
    step=0.1,
    description="Epochs"
)
batch_size = widgets.FloatLogSlider(
    value=1,
    min=1,
    max=3,
    step=0.1,
    description="Batch Size"
)
early_stop = widgets.Checkbox(
    value=False,
    description='Early Stop',
    disabled=False,
    indent=False
)
metric=widgets.Dropdown(
    options=[('R2','r2'),('Mean Absolute Error','mae'),
             ('Variance Explained', 'var_explained'), 
             ('Mean Squared Error', 'mse'),
             ('Negative Mean Squared Error', 'nmse'), 
             ('Root Mean Squared Error', 'rmse'), 
             ('Negative Root Mean Squared Error','nrmse'), 
             ('Mean Squared Log Error','msle'), 
             ('Root Mean Squared Log Error','rmsle'),
             ('Median Absolute Error', 'medae'), 
             ('Mean Absolute Percentage Error','mape')],
    value='mse',
    description='Metric:'
)
val_size=widgets.FloatText(
    value=0.0,
    description='Validation File Size:',
    disabled=False
)
fit = widgets.Button(
    description='Fit Model',
    disabled=False,
    button_style='info',
    tooltip='Fit Model',
    icon='burn'
)


train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

# --------------------------------------------------------------------------- #
#                               Traces                                        #
# --------------------------------------------------------------------------- #
# Plot training scores line and ribbon
train_upper = go.Scatter(
    x=train_sizes, y=train_scores_mean + train_scores_std,
    mode='lines',
    marker=dict(color="#b3cde0"),            
    fillcolor="#b3cde0",
    fill='tonexty',
    showlegend=False
)
train = go.Scatter(
    name='Training Scores',
    x=train_sizes, y=train_scores_mean, 
    mode='lines+markers',  
    line=dict(color='#005b96'),            
    marker=dict(color='#005b96'),
    fillcolor="#b3cde0",
    fill='tonexty',            
    showlegend=False
)
train_lower = go.Scatter(
    x=train_sizes, y=train_scores_mean - train_scores_std,
    mode='lines',
    line=dict(color='#b3cde0'),            
    showlegend=False
)        

# Plot validation scores line and ribbon
val_upper = go.Scatter(
    x=train_sizes, y=test_scores_mean + test_scores_std,
    mode='lines',
    marker=dict(color='rgba(179,226,205, 0.5)'),
    line=dict(width=0),
    fillcolor="rgba(179,226,205, 0.5)",
    fill='tonexty',
    showlegend=False
)
val = go.Scatter(
    name='Cross-Validation Scores',
    x=train_sizes, y=test_scores_mean,
    mode='lines+markers',             
    line=dict(color='rgb(27,158,119)'), 
    marker=dict(color='rgb(27,158,119)'),           
    fillcolor="rgba(179,226,205, 0.5)",
    fill='tonexty',            
    showlegend=False
)
val_lower = go.Scatter(
    x=train_sizes, y=test_scores_mean - test_scores_std,
    mode='lines',
    line=dict(color="rgba(179,226,205, 0.5)"),
    showlegend=False
)                
# Create empty FigureWidget 
data = [val_lower, val, val_upper, train_lower, train, train_upper]
# Update layout with designated template
lc = go.FigureWidget(data=data, layout=go.Layout(
    title=dict(
        text='Learning Curve'
    ),
    template='plotly_white',
    barmode='overlay'
))

# --------------------------------------------------------------------------- #
#                               Functions                                     #
# --------------------------------------------------------------------------- #
def get_estimator():
    if algorithm.value in estimators.values():
        return algorithm.value()
    else:
        raise ValueError("Invalid algorithm")

def fit(est, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(0.1,1,10), 
                    return_times=False):
    train_sizes, train_scores, test_scores = \
        learning_curve(est, X, y, cv=None, n_jobs=None,
                        train_sizes=np.linspace(0.1,1,10), 
                        return_times=False)
    return train_sizes, train_scores, test_scores

def response():
    est = get_estimator()
    
    if training_size_min.value or training_size_max.value
            or training_size_steps:
        
        
            algorithm.value 

# --------------------------------------------------------------------------- #
#                               Observers                                     #
# --------------------------------------------------------------------------- #
estimator.observe(response, names="value")
training_size_min.observe(response, names="value")
training_size_steps.observe(response, names="value")
training_size_max.observe(response, names="value")
learning_rate.observe(response, names="value")
epochs.observe(response, names="value")
batch_size.observe(response, names="value")
early_stop.observe(response, names="value")
metric.observe(response, names="value")
val_size.observe(response, names="value")
fit.observe(response, names="value")

# %%
