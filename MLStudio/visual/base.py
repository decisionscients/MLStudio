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

import plotly.graph_objects as go
from sklearn.base import BaseEstimator
# --------------------------------------------------------------------------- #
#                            VISUALATRIX                                      #
# --------------------------------------------------------------------------- #

class Visualatrix(ABC, BaseEstimator):
    """Abstact base class at the top of the visualator object hierarchy
    
    This base class defines how MLStudio creates, stores, and renders 
    visualizations using Plotly.

    Parameters
    ----------
    name : str
        A name for the plot object.

    """
    PLOT_DEFAULT_HEIGHT = 450
    PLOT_DEFAULT_WIDTH  = 700   
    PLOT_DEFAULT_TEMPLATE = "plotly_white"    
    PLOT_AVAILABLE_TEMPLATES = ['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none']
    
    def __init__(self, **kwargs):        
        self._fig = None
        self._height = kwargs.pop('height', self.PLOT_DEFAULT_HEIGHT)
        self._width = kwargs.pop('width', self.PLOT_DEFAULT_WIDTH)
        self._title = kwargs.pop('title', None)
        self._template = kwargs.pop('template', self.PLOT_DEFAULT_TEMPLATE)

    @property
    def fig(self):
        """The plotly figure object."""
        return self._fig

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
    def title(self):
        """Returns the title of the figure."""
        return self._title

    @title.setter
    def title(self, value):
        """Sets the title of the figure."""
        self._title = value

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
    
    def show(self, **kwargs):
        """Renders the visualization"""
        self._fig.show()

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
        self._fig.write_image(filepath)



