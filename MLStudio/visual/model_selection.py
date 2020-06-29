#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : model_selection.py                                                 #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Tuesday, March 17th 2020, 7:25:56 pm                         #
# Last Modified : Tuesday, March 17th 2020, 7:25:56 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
""" Model selection plots.

Model selection visualizations include: 

    * Cost Curves: Training costs by epoch.
    * Learning Curves: Training and validation scores by training set sizes.
    * Scalability Curves: Fit times by training examples
    * Performance Curve: Scores by fit times.
    * Validation Curves: Training and validation scores by parameter.
    * Validation Surfaces: Validation scores by two parameters.


"""
from abc import ABC, abstractmethod
import math
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve
from sklearn.utils.validation import check_X_y, check_array

from mlstudio.utils.format import proper
from mlstudio.visual.base import BaseVisualizer, BaseMultiVisualizer
from mlstudio.visual.base import BaseAnimator, BaseMultiAnimator
# ---------------------------------------------------------------------------- #
#                       TRAINING OPTIMIZATION CURVE                            #
# ---------------------------------------------------------------------------- #
class OptimizationCurve(BaseVisualizer):
    """ Plots training and validation error by epoch

    Parameters
    ----------
    estimator : An unfitted MLStudio estimator object
        The estimator object being visualized

    title : str (default=None)
        The title of the visualization. Each subclass will have a default
        title.

    height : int (default=450)
        The height of the visualization in pixels

    width : int (default=900)
        The width of the visualization in pixels

    template : str (default='plotly_white')
        The plotly template to be used. Valid values include:
        'ggplot2', 'seaborn', 'simple_white', 'plotly',
        'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
        'ygridoff', 'gridon', 'none'    

    metric : str 'cost' or 'score' (default='cost')
        Determines whether to plot cost or scores

    """
    _default_title = {'cost': 'Training Optimization Curve',
                      'score': 'Performance Optimization Curve'}

    def __init__(self, estimator, title=None, height=450, width=900, 
                 template='plotly_white', metric='cost'): 
        super(OptimizationCurve, self).__init__(
            estimator=estimator,
            title=title or self._default_title[metric],
            height=height,
            width=width,
            template=template
        )    
        self.metric = metric

    def _get_data(self):
        """Extracts data from blackbox and returns a dictionary for plotting."""
        d = {}
        d['Epoch'] = np.arange(1,self.estimator.blackbox_.total_epochs+1)
        if self.metric == 'cost':
            d['yaxis_title'] = self.estimator.objective.name
            d['Training Loss'] = self.estimator.blackbox_.epoch_log.get('train_cost')
            if self.estimator.blackbox_.epoch_log.get('val_cost'):
                d['Validation Loss'] = self.estimator.blackbox_.epoch_log.get('val_cost')
        else:
            d['yaxis_title'] = self.estimator.scorer.label
            d['Training Score'] = self.estimator.blackbox_.epoch_log.get('train_score')
            if self.estimator.blackbox_.epoch_log.get('val_score'):
                d['Validation Score'] = self.estimator.blackbox_.epoch_log.get('val_score')            
        return d

    def fit(self, X, y):
        """Fits the model and creates the figure object.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        
        """
        super(OptimizationCurve, self).fit(X, y)        
        # Extract data from the estimator's blackbox for plotting
        data = self._get_data()

        self.fig = go.Figure()

        # Training metric trace
        [training_metric] = [metric for metric in data.keys() if "Training" in metric]        
        self.fig.add_trace(go.Scatter(
            name=training_metric,
            mode='lines',
            x=data['Epoch'], y=data[training_metric],
            line=dict(color='#005b96'),            
            showlegend=True
        ))

        # Validation metric if available
        if self.estimator.val_size:
            [validation_metric] = [metric for metric in data.keys() if "Validation" in metric]        
            self.fig.add_trace(go.Scatter(
                name=validation_metric,
                mode='lines',
                x=data['Epoch'], y=data[validation_metric],
                line=dict(color='rgb(27,158,119)'),             
                showlegend=True
            ))

        self.fig.update_layout(
            title=self.title,
            xaxis=dict(title='Epoch'),
            yaxis=dict(title=data['yaxis_title']),
            title_x=0.5,
            template=self.template)

        return self
