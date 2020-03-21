#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : model_validation.py                                               #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 5:21:32 am                      #
# Last Modified : Thursday, March 19th 2020, 7:18:57 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Model Validation Plots."""
import math
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve

from .base import ModelVisualatrix
from mlstudio.visual import COLORS
from mlstudio.supervised.regression import LinearRegression
from mlstudio.utils.format import proper        
# --------------------------------------------------------------------------  #
#                              RESIDUALS                                      #
# --------------------------------------------------------------------------  #
class Residuals(ModelVisualatrix):        
    """Plots residuals versus predicted values.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget
        The plotting object. 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    hist : Boolean
        If True, a histogram is adjoined to the plot     
    
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

    def __init__(self, estimator, fig=None, hist=True, **kwargs):
        super(Residuals, self).__init__(estimator=estimator,
                                        fig=fig, **kwargs)

        self.hist = hist
        self.title = self.title or str(estimator.description + "<br>Residuals vs. Predicted")

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
        self.estimator.fit(X,y)
        y_pred = self.estimator.predict(X)
        res = y - y_pred

        x=y_pred.ravel()
        y=res.ravel()
        d = {'Residuals': y, 'Predicted': x}
        df = pd.DataFrame(d)

        self.fig = px.scatter(df, x='Predicted', y='Residuals', marginal_y="histogram",
                              width=self.width, height=self.height)

        self.fig.update_traces(
            marker=dict(
                color=COLORS['blue'],
                line=dict(
                    width=1,
                    color='white'
                )))

        self.fig.update_layout(
            title=self.title,
            title_x=0.5,
            height=self.height,
            width=self.width,
            template=self.template)