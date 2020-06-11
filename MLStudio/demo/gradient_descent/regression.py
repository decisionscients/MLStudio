#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : regression.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 3:27:23 pm                         #
# Last Modified : Wednesday, June 10th 2020, 9:11:49 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
from collections import OrderedDict
import os
from pathlib import Path
import sys
homedir = str(Path(__file__).parents[3])
demodir = str(Path(__file__).parents[1])
sys.path.append(homedir)

import pandas as pd
import numpy as np
from sklearn.linear_model import SGDRegressor

from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentRegressor
from mlstudio.utils.data_manager import StandardScaler, data_split
from mlstudio.visual.animations import animate_optimization_regression
from mlstudio.visual.static import plot_cost_scores

def get_data():
    """Obtains the Ames housing price data for modeling."""
    # ----------------------------------------------------------------------  #
    # Designate file locations
    datadir = os.path.join(homedir,"mlstudio/demo/data/Ames/")
    filepath = os.path.join(datadir, "train.csv")
    # ----------------------------------------------------------------------  #
    # Obtain and scale data
    cols = ["GrLivArea", "SalePrice"]
    df = pd.read_csv(filepath, nrows=500, usecols=cols)
    df_samples = df.head()
    X = np.array(df['GrLivArea']).reshape(-1,1)
    y = df['SalePrice']
    X_train, X_test, y_train, y_test = data_split(X,y, random_state=5)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def train_models(X, y):
    """Trains batch, stochastic and minibatch gradient descent."""    
    bgd = GradientDescentRegressor(theta_init=np.array([0,0]), epochs=500, random_state=50)
    sgd = GradientDescentRegressor(theta_init=np.array([0,0]), epochs=500, batch_size=1, random_state=50)
    mbgd = GradientDescentRegressor(theta_init=np.array([0,0]), epochs=500, batch_size=64, random_state=50)
    bgd.fit(X,y)
    sgd.fit(X,y)
    mbgd.fit(X,y)
    estimators = {'Batch Gradient Descent': bgd, 'Stochastic Gradient Descent': sgd,
            'Minibatch Gradient Descent': mbgd}
    return estimators

def append_filepath(filepath=None, appendage=None):
    """Appends the appendage to the filepath."""
    if filepath:
        base = os.path.basename(filepath)
        ext = os.path.splitext(base)[1]
        base = base + "_" + appendage + ext
        filepath = os.path.join(os.path.dirname(filepath), base)
    return filepath

def plot_optimization(estimators, filepath=None, show=True):
    """Renders surface, scatterplot, and line plots."""        
    filepath = append_filepath(filepath, "optimization")
    # Render plot
    animate_optimization_regression(estimators=estimators, filepath=filepath, show=show)

def plot_results(estimators, X_train, X_test, y_train, y_test, filepath=None, show=None):
    """Renders training error and validation scores for models vis-a-vis sklearn."""
    # Get Sklearn's SGDRegressor estimation
    skl = SGDRegressor(penalty=None, max_iter=500, random_state=50)
    skl.fit(X_train, y_train)    

    # Add SGDRegressor to dict of estimators
    estimators['Scikit-learn SGD Regressor'] = skl
    
    # Extract cost data from the model blackboxes            
    cost_plot_data = OrderedDict()
    for name, estimator in estimators.items():
        if hasattr(estimator, 'blackbox_'):
            cost_plot_data[name] = estimator.blackbox_.epoch_log.get('train_cost')[-1]
    
    # Obtain score data by scoring each estimator
    score_plot_data = OrderedDict()
    for name, estimator in estimators.items():
        score_plot_data[name] = estimator.score(X_test, y_test)

    # Format plot data
    plot_data = OrderedDict()
    plot_data['cost'] = cost_plot_data
    plot_data['score'] = score_plot_data

    # Add plot type to plot filepath
    filepath = append_filepath(filepath, "test_results")

    # Render plot
    plot_cost_scores(data=plot_data, filepath=filepath, show=show)

def regression_demo(filepath=None, show=True):
    """Regression demo main processing function.
    
    Parameters
    ----------
    filepath:  str
        A relative or absolute filepath. 

    show: bool (default=True)
        Indicates whether to render the plot  
    
    """
     
    X_train, X_test, y_train, y_test = get_data()
    estimators = train_models(X_train, y_train)
    plot_optimization(estimators=estimators, filepath=filepath, show=show)
    plot_results(estimators=estimators, X_train=X_train, X_test=X_test, 
                 y_train=y_train, y_test=y_test, filepath=filepath, 
                 show=show)

filepath = os.path.join(demodir, 'figures/regression_demo.html')
regression_demo(filepath)
#%%

