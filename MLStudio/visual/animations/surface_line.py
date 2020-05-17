#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : gradient_descent_.py                                              #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 14th 2020, 3:02:34 am                         #
# Last Modified : Thursday, May 14th 2020, 3:02:35 am                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Animates gradient descent with surface and line plots."""
import os

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio 
import plotly.offline as py
from plotly.subplots import make_subplots

from mlstudio.utils.data_manager import todf
# --------------------------------------------------------------------------  #
class SurfaceLine:
    """Animates gradient descent with a surface and line plot."""
    def __init__(self):
        pass

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))        

    def animate(self, model, directory=None, filename=None):

        # Extract model blackbox
        blackbox = model.blackbox_

        # Extract the theta, x=theta0, y=theta1 and cost history
        theta = blackbox.epoch_log.get('theta')
        weights = todf(blackbox.epoch_log['theta'], stub='theta_')        
        theta0 = np.array(weights['theta_0'])
        theta1 = np.array(weights['theta_1'])
        cost = blackbox.epoch_log.get('train_cost')

        # Establish boundaries of plot
        theta0_min = min(theta0)
        theta1_min = min(theta1)
        theta0_max = max(theta0) 
        theta1_max = max(theta1) 
        theta0_mesh = np.linspace(theta0_min, theta0_max, 50)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 50)
        theta0_mesh, theta1_mesh = np.meshgrid(theta0_mesh, theta1_mesh)

        # Create z axis grid based upon X,y and the grid of thetas
        Js = np.array([self._cost_mesh(model.X_train_, model.y_train_, THETA)
                    for THETA in zip(np.ravel(theta0_mesh), np.ravel(theta1_mesh))])
        Js = Js.reshape(theta0_mesh.shape)  

        # Create regression line data
        n_frames = len(theta0) 
        def f(x, theta):
            return theta[0] + x * theta[1]

        X = model.X_train_
        x = X[:,1]        
        xm = np.min(x)
        xM = np.max(x)
        xx = np.linspace(xm, xM)
        yy = []
        for i in range(n_frames):
            ym = f(xm, theta[i])
            yM = f(xM, theta[i])
            yy.append(np.linspace(ym, yM))      
    
        # Initialize figure with 2 subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Gradient Descent", "Linear Regression"),
                            specs=[[{'type': "surface"}, {"type": "scatter"}]])      

        # Add surface and scatter plot
        fig.add_trace(
            go.Surface(x=theta0, y=theta1, z=Js, colorscale="YlGnBu", showscale=False),
            row=1, col=1)

        fig.add_trace(
            go.Scatter(x=model.X_train_[:,1], y=model.y_train_,
                       name="ames",
                       mode="markers",
                       marker=dict(color="#1560bd")), row=1, col=2)

        # Create frames definition                       
        frames = [go.Frame(
            dict(
                name = k,
                data = [
                    go.Scatter3d(x=[theta0[k]], y=[theta1[k]], z=[cost[k]], mode='markers', marker=dict(color="red", size=16)),
                    go.Scatter(x=xx, y=yy[k], mode="lines")
                ],
                traces=[0,1])
            ) for k in range(n_frames)]

        # Update the menus
        updatemenus = [dict(type='buttons',
                            buttons=[dict(label="Play",
                                          method="animate",
                                          args=[[f'{k}' for k in range(n_frames)],
                                            dict(frame=dict(duration=100, redraw=False),
                                                 transition=dict(duration=0),
                                                 easing="linear",
                                                 fromcurrent=True,
                                                 mode="immediate")])],
                            direction="left",
                            pad=dict(r=10, t=85),
                            showactive=True, x=0.1, y=0, xanchor="right", yanchor="top")]

        sliders = [{"yanchor": "top",
                   "xanchor": "left",
                   "currentvalue": {"font": {"size": 16}, "prefix": "Iteration: ", "visible":True, "xanchor": "right"},
                   'transition': {'duration': 100.0, 'easing': 'linear'},
                   'pad': {'b': 10, 't': 50}, 
                   'len': 0.9, 'x': 0.1, 'y': 0, 
                   'steps': [{'args': [[k], {'frame': {'duration': 100.0, 'easing': 'linear', 'redraw': False},
                                      'transition': {'duration': 0, 'easing': 'linear'}}], 
                       'label': k, 'method': 'animate'} for k in range(n_frames)       
                    ]}]

        fig.update(frames=frames)
        fig.update_layout(
            xaxis=dict(range=[theta0_min, theta0_max], autorange=False, zeroline=False),
            yaxis=dict(range=[theta1_min, theta1_max], autorange=False, zeroline=False),            
            title=dict(xanchor='center', yanchor='top', x=0.5, y=0.9),        
            font=dict(family="Open Sans"),    
            updatemenus=updatemenus, 
            sliders=sliders, 
            template='plotly_white')
        if directory and filename:
            filepath = os.path.join(directory, filename)
            fig.write_html(filepath, include_mathjax='cdn')
        pio.renderers.default = "browser"
        fig.show()
