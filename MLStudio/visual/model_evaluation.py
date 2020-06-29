#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : model_evaluation.py                                               #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, March 19th 2020, 9:14:48 am                       #
# Last Modified : Thursday, March 19th 2020, 9:14:48 am                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Model Evaluation Plots. """
import math
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve

from .base import ModelVisualizer
from mlstudio.supervised.machine_learning.linear_regression import LinearRegression
from mlstudio.supervised.machine_learning.ols_regression import OLSRegression
from mlstudio.utils.format import proper

# ---------------------------------------------------------------------------- #
#                             PREDICTION ERROR                                 #
# ---------------------------------------------------------------------------- #        
class PredictionError(ModelVisualizer):        
    """Plots actual target values against predicted values.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 
        
    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.
    
    kwargs : dict
        Keyword arguments that are passed to the base class and influence
        the visualization. Optional keyword arguments include:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        height      specify the height of the figure
        width       specify the width of the figure
        title       specify the title of the figure
        template    specify the template for the figure.
        =========   ==========================================    
    
    """

    def __init__(self, estimator, fig=None, **kwargs):
        super(PredictionError, self).__init__(estimator=estimator, 
                                              fig=fig, **kwargs)
        self.title = self.title or str(estimator.description + "<br>Prediction Error")

    def fit(self, X, y):
        """Generates the prediction error plot.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Compute predicted vs actual.
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)

        # Compute best fit line predicted vs actual
        y = y.reshape(-1,1)
        est = OLSRegression()
        est.fit(y, y_pred) 

        # Format data for identity and best fit lines
        y = y.ravel()
        best_fit_x = np.arange(min(y), max(y))
        best_fit_y = est.intercept_ + est.coef_ * best_fit_x
        identity_x = best_fit_x
        identity_y = best_fit_x 

        # Scatterplot of predicted vs actual
        scatter = go.Scatter(
            x=y, y=y_pred,
            mode='markers',
            marker=dict(color='#005b96'),
            line_color='rgba(255,255,255,0.5)',
            opacity=0.75,
            showlegend=False
        )

        # Plot best fit line
        best_fit = go.Scatter(
            name='Best Fit',
            x=best_fit_x, y=best_fit_y, 
            mode='lines',  
            line=dict(color='#005b96'),            
            showlegend=True
        )
        identity = go.Scatter(
            name='Identity',
            x=identity_x, y=identity_y,
            mode='lines',
            line=dict(color='#b3cde0'),            
            showlegend=True
        )        

        # Load from bottom up
        data = [scatter, best_fit, identity]
        # Update layout with designated template
        layout = go.Layout(
            xaxis=dict(title='y'),
            yaxis=dict(title=r'$\hat{y}$'),
            title=self.title,title_x=0.5,
            template=self.template
        )
        self.fig = go.Figure(data=data, layout=layout)

