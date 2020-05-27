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
from copy import copy, deepcopy    
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
class Surface:
    """Animates gradient descent on several benchmark functions."""
    def __init__(self):
        pass    

    def animate(self, estimators, directory=None, filename=None, show=True):
        # ------------------------------------------------------------------  #
        # Extract parameter and cost data from the model blackboxes
        # Data for meshgrid
        xm, xM = 0, 0 # x range from objective function object
        ym, yM = 0, 0 # y range from objective function object
        # Models containing plot data (parameters theta and cost)
        models = OrderedDict()
        # Index for model dictionary  
        names = [] 
        # Data used to obtain best model for the objective function
        true = None
        estimates = []
        similarities = [] 
        # The number of frames = model.epochs
        n_frames = 0
        # The objective function
        objective = None
        
        for name, estimator in estimators.items():
            model = estimator['model']
            results = estimator['results']      
            # Get range from objective object for meshgrid
            x, y = model.objective.range
            xm, xM = x['min'], x['max']
            ym, yM = y['min'], y['max']
            # Extract model data for plotting gradient descent models 
            theta = model.blackbox_.epoch_log.get('theta')
            theta = todf(theta, stub='theta_')
            # Clip thetas to plotting range 
            theta_0 = np.clip(np.array(theta['theta_0']), xm, xM)
            theta_1 = np.clip(np.array(theta['theta_1']), ym, yM)
            d = OrderedDict()
            d['theta_0'] = theta_0[::2]
            d['theta_1'] = theta_1[::2]
            d['cost'] = model.blackbox_.epoch_log.get('train_cost')[::2]
            models[name] = d
            # Place index for models dictionary in a list for easy access
            names.append(name)
            # Update best model data
            true = model.objective.minimum if true is None else true
            estimates.append(model.theta_)
            similarities.append(results['sim'])
            # Get objective function name for plot header 
            objective = model.objective if objective is None else objective
            # Designate the number of frames
            n_frames = len(d['theta_0']) if n_frames == 0 else n_frames

        # ------------------------------------------------------------------  #
        # Find best performing estimator and data
        best_idx = np.argmax(np.array(similarities))
        best_model = names[best_idx]
        best_true = str(true)
        best_est = str(np.around(estimates[best_idx],2))
        best_sim = str(np.around(similarities[best_idx],2))
        best_msg = "<b>Best Model: " + best_model + "</b><br>" + \
            "<b>True Minimum:</b> " + best_true +\
            "<b> Estimated Minimum:</b> " + best_est + \
                "<b> Cosine Similarity:</b> " + best_sim

        # ------------------------------------------------------------------  #
        # Create data for surface plot
        theta0_min, theta0_max = xm, xM
        theta1_min, theta1_max = ym, yM
        theta0_mesh = np.linspace(theta0_min, theta0_max, 50)
        theta1_mesh = np.linspace(theta1_min, theta1_max, 50)
        theta0_mesh_grid, theta1_mesh_grid = np.meshgrid(theta0_mesh, theta1_mesh)        
        # Create z axis grid based upon X,y and the grid of thetas
        Js = np.array([objective(THETA) for THETA in zip(np.ravel(theta0_mesh_grid), np.ravel(theta1_mesh_grid))])
        Js = Js.reshape(theta0_mesh_grid.shape)          
  
        # ------------------------------------------------------------------  #
        # Add colors to model
        colors = [ "red", "blue", "green", "purple", "orange", "black", "darkcyan", "maroon",
                   "darkgoldenrod", "sienna", "darkslategrey", "lightcoral", "lime"]
        for i, model in enumerate(models.values()):
            model['color'] = colors[i]


        fig = go.Figure()
        # ------------------------------------------------------------------  #
        # Add Surface Plot        
        # Trace 0: Surface plot                
        fig.add_trace(
            go.Surface(x=theta0_mesh, y=theta1_mesh, z=Js, colorscale="YlGnBu", 
                       showscale=False, showlegend=False))

        # ------------------------------------------------------------------  #
        # Traces 1-12: Add Gradient Descent Trajectories
        for name, model in models.items():
            fig.add_trace(
                go.Scatter3d(x=model['theta_0'][:1], y=model['theta_1'][:1], z=model['cost'][:1],
                            name=name, 
                            showlegend=True, 
                            mode='lines', line=dict(color=model['color'], width=5)))                        

        # ------------------------------------------------------------------  #
        # Set layout title, font, template, etc...
        fig.update_layout(
            height=600, width=1200,
            scene=dict(
                xaxis=dict(nticks=20),
                zaxis=dict(nticks=4)
            ),
            #scene_xaxis=dict(range=[theta0_min, theta0_max], autorange=False),
            #scene_yaxis=dict(range=[theta1_min, theta1_max], autorange=False),            
            #scene_zaxis=dict(range=[zm, zM], autorange=False),            
            title=dict(text=objective.name, xanchor='center', yanchor='top', x=0.5, y=0.9),        
            font=dict(family="Open Sans"),                
            showlegend=True,      
            annotations=[
                dict(
                    x=0.5,
                    y=1.0,
                    showarrow=False,
                    text=best_msg,
                    xref="paper",
                    yref="paper"
                )               
            ],      
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
                    go.Scatter3d(x=models[names[3]]['theta_0'][:k+1], 
                                 y=models[names[3]]['theta_1'][:k+1], 
                                 z=models[names[3]]['cost'][:k+1]),
                    go.Scatter3d(x=models[names[4]]['theta_0'][:k+1], 
                                 y=models[names[4]]['theta_1'][:k+1], 
                                 z=models[names[4]]['cost'][:k+1]),                                 
                    go.Scatter3d(x=models[names[5]]['theta_0'][:k+1], 
                                 y=models[names[5]]['theta_1'][:k+1], 
                                 z=models[names[5]]['cost'][:k+1]),
                    go.Scatter3d(x=models[names[6]]['theta_0'][:k+1], 
                                 y=models[names[6]]['theta_1'][:k+1], 
                                 z=models[names[6]]['cost'][:k+1]),
                    go.Scatter3d(x=models[names[7]]['theta_0'][:k+1], 
                                 y=models[names[7]]['theta_1'][:k+1], 
                                 z=models[names[7]]['cost'][:k+1]),   
                    go.Scatter3d(x=models[names[8]]['theta_0'][:k+1], 
                                 y=models[names[8]]['theta_1'][:k+1], 
                                 z=models[names[8]]['cost'][:k+1]),   
                    go.Scatter3d(x=models[names[9]]['theta_0'][:k+1], 
                                 y=models[names[9]]['theta_1'][:k+1], 
                                 z=models[names[9]]['cost'][:k+1]),   
                    go.Scatter3d(x=models[names[10]]['theta_0'][:k+1], 
                                 y=models[names[10]]['theta_1'][:k+1], 
                                 z=models[names[10]]['cost'][:k+1]),   
                    go.Scatter3d(x=models[names[11]]['theta_0'][:k+1], 
                                 y=models[names[11]]['theta_1'][:k+1], 
                                 z=models[names[11]]['cost'][:k+1]),                                                                                                                                                                                                                                                                                                                                         

                ],
                traces=[1,2,3,4,5,6,7,8,9,10,11,12])
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

        if directory and filename:
            filepath = os.path.join(directory, filename)
            fig.write_html(filepath, include_plotlyjs='cdn', include_mathjax='cdn')
        if show:
            pio.renderers.default = "browser"
            fig.show()

