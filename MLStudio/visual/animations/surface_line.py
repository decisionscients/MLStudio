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
from collections import OrderedDict
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.io as pio 
import plotly.offline as py
from plotly.subplots import make_subplots

from mlstudio.utils.data_manager import todf
from mlstudio.utils.file_manager import check_directory
# --------------------------------------------------------------------------  #
class SurfaceLine:
    """Animates gradient descent with a surface and line plot."""
    def __init__(self):
        pass

    def _cost_mesh(self,X, y, THETA):
        return(np.sum((X.dot(THETA) - y)**2)/(2*len(y)))        

    def animate(self, estimators, filepath=None, show=True):
        # ------------------------------------------------------------------  #
        # Extract parameter and cost data from the model blackboxes
        theta0 = []
        theta1 = []
        models = OrderedDict()
        names = [] 
        for name, estimator in estimators.items():
            theta = estimator.blackbox_.epoch_log.get('theta')
            # Thetas converted to individual columns in dataframe and extacted  
            theta = todf(theta, stub='theta_')
            theta0.extend(theta['theta_0'])
            theta1.extend(theta['theta_1'])
            d = OrderedDict()
            d['theta_0'] = theta['theta_0']
            d['theta_1'] = theta['theta_1']
            d['cost'] = estimator.blackbox_.epoch_log.get('train_cost')
            models[name] = d
            names.append(name)

        # ------------------------------------------------------------------  #
        # Obtain training data and set range along x axis
        X_train_ = estimators[names[0]].X_train_
        y_train_ = estimators[names[0]].y_train_
        xm, xM = np.min(X_train_[:,1]), np.max(X_train_[:,1])        
        xx = np.linspace(xm, xM)            

        # ------------------------------------------------------------------  #
        # Create data for surface plot
        theta0_min, theta0_max = min(theta0), max(theta0)
        theta1_min, theta1_max = min(theta1), max(theta1)
        theta0_mesh = np.linspace(theta0_min, theta0_max, 50)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 50)
        theta0_mesh_grid, theta1_mesh_grid = np.meshgrid(theta0_mesh, theta1_mesh)        
        # Create z axis grid based upon X,y and the grid of thetas
        Js = np.array([self._cost_mesh(X_train_, y_train_, THETA)
                    for THETA in zip(np.ravel(theta0_mesh_grid), np.ravel(theta1_mesh_grid))])
        Js = Js.reshape(theta0_mesh_grid.shape)          
        

        # ------------------------------------------------------------------  #
        # Create regression line data
        n_frames = len(models[names[0]]['theta_0'])
        def f(x, theta0, theta1):
            return theta0 + x * theta1
                
        for name, model in models.items():
            yy = []
            for j in range(n_frames):
                ym = f(xm, model['theta_0'][j], model['theta_1'][j])
                yM = f(xM, model['theta_0'][j], model['theta_1'][j])
                yy.append(np.linspace(ym, yM))
            models[name]['yy'] = yy
  
        # ------------------------------------------------------------------  #
        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Gradient Descent", "Linear Regression"),
                            specs=[[{'type': "surface"}, {"type": "xy"}]])      

        # ------------------------------------------------------------------  #
        # Add colors to model
        colors = ["red", "green", "black"]
        for i, model in enumerate(models.values()):
            model['color'] = colors[i]

        # ------------------------------------------------------------------  #
        # Add Surface Plot        
        # Subplot 1, Trace 0: Surface plot
        fig.add_trace(
            go.Surface(x=theta0_mesh, y=theta1_mesh, z=Js, colorscale="YlGnBu", 
                       showscale=False, showlegend=False), row=1, col=1)

        # ------------------------------------------------------------------  #
        # Subplots 1, Traces 1, 2 & 3: Gradient Descent Trajectories
        for name, model in models.items():
            fig.add_trace(
                go.Scatter3d(x=model['theta_0'][:1], y=model['theta_1'][:1], z=model['cost'][:1],
                            name=name, 
                            showlegend=False, 
                            mode='lines', line=dict(color=model['color'], width=5)),
                            row=1, col=1)            

        # ------------------------------------------------------------------  #
        # Subplot 2, Trace 4 Ames Data
        fig.add_trace(
            go.Scatter(x=X_train_[:,1], y=y_train_,
                       name="Ames Data",
                       mode="markers",
                       showlegend=True,
                       marker=dict(color="#1560bd")), row=1, col=2)                    

        # ------------------------------------------------------------------  #
        # Subplot 2, Traces 5, 6 & 7: Regression Lines
        for name, model in models.items():
            fig.add_trace(
                go.Scatter(x=xx, y=model['yy'][0], 
                        name=name,
                        mode="lines", marker=dict(color=model['color'], size=0.5)),
                        row=1, col=2)

        # ------------------------------------------------------------------  #
        # Set layout title, font, template, etc...
        fig.update_layout(
            height=600,
            #scene_xaxis=dict(range=[theta0_min, theta0_max], autorange=False),
            #scene_yaxis=dict(range=[theta1_min, theta1_max], autorange=False),            
            #scene_zaxis=dict(range=[zm, zM], autorange=False),
            title=dict(xanchor='center', yanchor='top', x=0.5, y=0.9),        
            font=dict(family="Open Sans"),                
            showlegend=True,            
            template='plotly_white');                       

        # ------------------------------------------------------------------  #
        # Create frames                       
        frames = [go.Frame(
            dict(
                name = f'{k+1}',
                data = [                    
                    go.Scatter3d(x=models[names[0]]['theta_0'][:k+1], 
                                 y=models[names[0]]['theta_1'][:k+1], 
                                 z=models[names[0]]['cost'][:k+1]),
                    go.Scatter3d(x=models[names[1]]['theta_0'][:k+1], 
                                 y=models[names[1]]['theta_1'][:k+1], 
                                 z=models[names[1]]['cost'][:k+1]),                                 
                    go.Scatter3d(x=models[names[2]]['theta_0'][:k+1], 
                                 y=models[names[2]]['theta_1'][:k+1], 
                                 z=models[names[2]]['cost'][:k+1]),                                                                  
                    go.Scatter(x=xx, y=models[names[0]]['yy'][k]),
                    go.Scatter(x=xx, y=models[names[1]]['yy'][k]),
                    go.Scatter(x=xx, y=models[names[2]]['yy'][k])
                ],
                traces=[1,2,3,5,6,7])
            ) for k in range(n_frames-1)]

        # Update the menus
        updatemenus = [dict(type='buttons',
                            buttons=[dict(label="Play",
                                          method="animate",
                                          args=[[f'{k+1}' for k in range(n_frames-1)],
                                            dict(frame=dict(duration=1, redraw=True),
                                                 transition=dict(duration=1),
                                                 easing="linear",
                                                 fromcurrent=True,
                                                 mode="immediate")])],
                            direction="left",
                            pad=dict(r=10, t=85),
                            showactive=True, x=0.1, y=0, xanchor="right", yanchor="top")]

        sliders = [{"yanchor": "top",
                   "xanchor": "left",
                   "currentvalue": {"font": {"size": 16}, "prefix": "Iteration: ", "visible":True, "xanchor": "right"},
                   'transition': {'duration': 1, 'easing': 'linear'},
                   'pad': {'b': 10, 't': 50}, 
                   'len': 0.9, 'x': 0.1, 'y': 0, 
                   'steps': [{'args': [[f'{k+1}'], {'frame': {'duration': 1, 'easing': 'linear', 'redraw': False},
                                      'transition': {'duration': 1, 'easing': 'linear'}}], 
                       'label': k, 'method': 'animate'} for k in range(n_frames-1)       
                    ]}]

        fig.update(frames=frames)

        fig.update_layout(
            updatemenus=updatemenus,
            sliders=sliders
        )

        if filepath:
            fig.write_html(filepath, include_plotlyjs='cdn', include_mathjax='cdn')
        if show:
            pio.renderers.default = "browser"
            fig.show()

