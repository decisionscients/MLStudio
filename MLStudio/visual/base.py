#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : base.py                                                            #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Tuesday, March 17th 2020, 7:15:23 pm                         #
# Last Modified : Tuesday, March 17th 2020, 7:15:23 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Base class for visualizations."""
import os
from abc import ABC, abstractmethod

import plotly.express as px
import plotly.graph_objects as go
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array
# --------------------------------------------------------------------------- #
#                            VISUALATRIX                                      #
# --------------------------------------------------------------------------- #

class Visualatrix(ABC, BaseEstimator):
    """Abstact base class at the top of the visualator object hierarchy
    
    This base class defines how the interface and common behaviors for and
    among the visualizations classes.  

    Parameters
    ----------
    kwargs : dict
        Keyword arguments including:

        =========   ==========================================
        Property    Description
        --------    ------------------------------------------
        fig         Plotly Figure or FigureWidget object
        height      specify the height of the figure
        width       specify the width of the figure
        template    specify the template for the figure.
        title       specify the title for the figure.
        =========   ==========================================

    """
    _PLOT_DEFAULT_HEIGHT = 450
    _PLOT_DEFAULT_WIDTH  = 900   
    _PLOT_DEFAULT_TEMPLATE = "plotly_white"    
    _PLOT_AVAILABLE_TEMPLATES = ['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none']    
    
    def __init__(self, fig=None, **kwargs):        
        self.fig = fig
        self.height = kwargs.pop('height', self._PLOT_DEFAULT_HEIGHT)
        self.width = kwargs.pop('width', self._PLOT_DEFAULT_WIDTH)
        self.template = kwargs.pop('template', self._PLOT_DEFAULT_TEMPLATE)
        self.title = kwargs.pop('title', None)

    @property
    def height(self):
        """Returns the height of the figure."""
        return self._height

    @height.setter
    def height(self, value):
        """Sets the height of the figure."""
        self._height = value

    @property
    def width(self):
        """Returns the width of the figure."""
        return self._width

    @height.setter
    def width(self, value):
        """Sets the width of the figure."""
        self._width = value        

    @property
    def template(self):
        """Returns the template of the figure."""
        return self._template

    @template.setter
    def template(self, value):
        """Sets the template of the figure."""
        self._template = value

    @abstractmethod
    def fit(self, X, y=None, **kwargs):
        """Fits the visualator to the data."""
        pass    

    def _validate(self, X, y=None, **kwargs):
        """Validates arguments passed to the fit method.
        
        Parameters
        ----------
        X : array-like of shape (m samples, n features)
            Object containing typically feature data.

        y : array-like of shape (m samples) (Optional)
            The target variable.
        """
        if y is not None:
            check_X_y(X, y)
        else:
            check_array(X)
        

    
    def show(self, **kwargs):
        """Renders the visualization"""
        self.fig.show()

    def save(self, filepath):
        """Saves image to filepath

        Parameters
        ----------
        filepath : str
            Relative filepath including file name and extension
        """
        directory = os.path.basename(filepath)
        if not os.path.exists(directory):
            os.mkdir(directory)
        self.fig.write_image(filepath)


# --------------------------------------------------------------------------- #
#                            ModelVisualatrix                                 #
# --------------------------------------------------------------------------- #

class ModelVisualatrix(Visualatrix):
    """Base class for model visualization classes."""

    def __init__(self, estimator, fig=None, **kwargs):
        super(ModelVisualatrix, self).__init__(fig=fig, **kwargs)
        self.estimator = estimator

    def _validate(self, X, y=None, **kwargs):
        """Validates parameters and arguments sent to the fit method.
        
        Parameters
        ----------
        X : array-like of shape (m samples, n features)
            Object containing typically feature data.

        y : array-like of shape (m samples) (Optional)
            The target variable.

        Raises
        ------
        ValueError : estimator has no 'fit' or 'predict' method.

        """
        super(ModelVisualatrix, self)._validate(X, y)

        # Confirm the estimator has fit and predict methods.
        if getattr(self.estimator, 'fit', None) and \
        getattr(self.estimator, 'predict', None):
            pass
        else:
            raise ValueError("The estimator object must have a 'fit' and \
                a 'predict' method.")

        
               