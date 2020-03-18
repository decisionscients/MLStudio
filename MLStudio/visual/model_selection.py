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
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go 
import plotly.express as px
from sklearn.model_selection import ParameterGrid 

from .base import Visualatrix
from mlstudio.utils.format import proper
# ---------------------------------------------------------------------------- #
#                              COST CURVES                                     #
# ---------------------------------------------------------------------------- #
class CostCurve(Visualatrix):
    """ Plots training costs by epoch.

    This visualization illuminates the path towards gradient descent convergence 
    over a designated number of epochs or until a stop condition is met. The
    basic line plot presents the number of epochs on the x-axis and costs on the
    y-axis. A factor variable may be provided to show multiple lines, one
    for each value of the factor variable. Subplots can also be created
    for each value of a subplot variable.

    Parameters
    ----------
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
    def __init__(self, estimator, **kwargs):
        super(CostCurve, self).__init__(**kwargs)
        self._estimator = estimator

    def fit(self, X, y=None, param_grid=None, color=None, facet_col=None, 
            facet_col_wrap=2):
        """Fits the model and creates the figure object.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        
        param_grid : dict (default=None)
            A dictionary in which the keys are estimator hyperparameter names, 
            and the values are hyperparameter values to be estimated. 
        
        color : str or None (Default=None)
            The param_grid key that is used to assign colors to marks.

        facet_col : str or None (Default=None)
            The param_grid key that is used to assign marks ot facetted subplots
            in the horizontal direction.  

        facet_col_wrap : int (Default=2)
            The maximum number of facet columns.  Wraps the column variable 
            at this width, so that the column facets span multiple rows.

        """
        self._validate(X, y, param_grid, color, facet_col, facet_col_wrap)
        data = self._fit_plot_data(X, y, param_grid)
        self._fig = self._fit_plot(data, color, facet_col, facet_col_wrap)

        return self


    def _validate(self, X, y, param_grid, color, facet_col, facet_col_wrap):
        if param_grid:
            if not color and not facet_col:
                raise ValueError("If using a param_grid, the color or facet_col\
                    parameters must be set.")
            if color and color not in param_grid:
                raise ValueError("The 'color' parameter must be the 'param_grid'.")
            if facet_col and facet_col not in param_grid:
                raise ValueError("The 'facet_col' parameter must be the 'param_grid'.")
            if not isinstance(facet_col_wrap, int):
                raise ValueError("The 'facet_col_wrap' parameter must be an integer.")


    def _fit_plot_data(self, X, y, param_grid):
        """Fits models and creates data in tidy for plotting."""
        data = pd.DataFrame()
        if param_grid:
            grid = ParameterGrid(param_grid)
            for params in grid:
                # Set params attributes on estimator
                d = {}
                for param, value in params.items():                    
                    setattr(self._estimator, param, value)
                    d[proper(param)] = round(value,4)

                self._estimator.fit(X,y)
                
                d['Epoch'] = np.arange(start=1, stop=self._estimator.history.total_epochs+1)
                d['Cost'] = self._estimator.history.epoch_log['train_cost']
                df = pd.DataFrame(d)
                data = pd.concat([data, df], axis=0)  
        else:      
            self._estimator.fit(X,y)
            d = {}
            d['Epoch'] = np.arange(start=1, stop=self._estimator.history.total_epochs+1)
            d['Cost'] = self._estimator.history.epoch_log['train_cost']
            data = pd.DataFrame(d)

        return data
    
    def _fit_plot(self, data, color, facet_col, facet_col_wrap):
        """Creates the plot express object."""
        if color and facet_col:
            fig = px.line(data, x='Epoch', y='Cost', color=proper(color), 
                        facet_col=proper(facet_col),
                        facet_col_wrap=facet_col_wrap,
                        template=self._template,
                        title=self._title,
                        height=self._height,
                        width=self._width)        
        elif color:
            fig = px.line(data, x='Epoch', y='Cost', color=proper(color),
                        template=self._template,
                        title=self._title,
                        height=self._height,
                        width=self._width)

        elif facet_col:             
            fig = px.line(data, x='Epoch', y='Cost', 
                        facet_col=proper(facet_col),
                        facet_col_wrap=facet_col_wrap,
                        template=self._template,
                        title=self._title,
                        height=self._height,
                        width=self._width)                    
        else:
            fig = px.line(data, x='Epoch', y='Cost',
                        template=self._template,
                        title=self._title,
                        height=self._height,
                        width=self._width)                    
        return fig
        
    


                    

                


