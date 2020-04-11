#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : variants.py                                                       #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 1:52:41 pm                         #
# Last Modified : Friday, April 10th 2020, 1:52:41 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[3]
site.addsitedir(PROJECT_DIR)

import ipywidgets as widgets
import numpy as np
import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, plot
from sklearn.datasets import make_regression

from mlstudio.supervised.estimator.gradient import GradientDescentRegressor
from mlstudio.utils.data_manager import StandardScaler, todf

# Obtain and standardize data
X, y, coef = make_regression(n_samples=1000, n_features=1, effective_rank=5, 
                             noise=50, random_state=5, coef=True)
scaler = StandardScaler()
X = scaler.transform(X)

# Run Models
theta_init = np.array([0.5, 1.0])
bgd = GradientDescentRegressor(theta_init=theta_init)
mbgd = GradientDescentRegressor(theta_init=theta_init, batch_size=32)
sgd = GradientDescentRegressor(theta_init=theta_init, batch_size=1)
bgd.fit(X,y)
mbgd.fit(X,y)
sgd.fit(X,y)
def _cost_mesh(X, y, THETA):
    return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))  

# Create the x=theta0, y=theta1 mesh grid
weights = todf(model.history_.epoch_log['theta'], stub='theta_')        
theta_0 = weights['theta_0']
theta_1 = weights['theta_1']
theta_0_range = np.linspace(min(theta_0), max(theta_0), 100)    
theta_1_range = np.linspace(min(theta_1), max(theta_1), 100) 
theta0_mesh, theta1_mesh = np.meshgrid(theta_0_range, theta_1_range)        

# Create cost grid based upon X_design matrix, y and the grid of thetas
Js = np.array([_cost_mesh(model.X_design, model.y, THETA)
            for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
Js = Js.reshape(theta0_mesh.shape)

fig = go.Figure(
    data=[go.Surface(z=Js, x=theta0_mesh, y=theta1_mesh, 
                colorscale='Viridis', opacity=0.75),
          go.Surface(z=Js, x=theta0_mesh, y=theta1_mesh, 
                colorscale='Viridis', opacity=0.75)],
    layout=go.Layout(
        template='plotly_white',
        title='Gradient Descent Linear Regression',
        width=1200, height=600, 
        scene=dict(xaxis=dict(title=dict(text='Theta 0')),
                   yaxis=dict(title=dict(text='Theta 1')),
                   zaxis=dict(title=dict(text='Cost'))),
        scene_camera=dict(eye=dict(x=-2, y=2, z=0.)),
        updatemenus = [
             {
                'buttons': [
                    {
                        'args': [[None], {'frame': {'duration': 100, 'redraw': False},
                                'fromcurrent': True, 'transition': {'duration': 100}}],
                        'label': 'Play',
                        'method': 'animate'
                    },
                    {
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate',
                        'transition': {'duration': 0}}],
                        'label': 'Pause',
                        'method': 'animate'
                    }
                ],
                'type': 'buttons'                
            }
        ]
    ),
    frames = [go.Frame(
        data=[go.Scatter3d(
            x=[theta_0[i]],
            y=[theta_1[i]],
            z=[model.history_.epoch_log['train_cost'][i]],
            mode='markers',
            marker=dict(color='red', size=2))])

            for i in range(model.history_.total_epochs)]
)        

plot(fig)
#%%



