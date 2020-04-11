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
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from sklearn.model_selection import ParameterGrid, learning_curve 
from sklearn.model_selection import validation_curve
from sklearn.utils.validation import check_X_y, check_array

from .base import Visualatrix
from mlstudio.utils.format import proper
from mlstudio.visual.base import ModelVisualatrix
# ---------------------------------------------------------------------------- #
#                              COST CURVES                                     #
# ---------------------------------------------------------------------------- #
class CostCurve(ModelVisualatrix):
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

    fig : Plotly Figure or FigureWidget object.
        The object being evaluated.

    param_grid : dict (default=None)
        A dictionary in which the keys are estimator hyperparameter names, 
        and the values are hyperparameter values to be estimated. 
    
    factor : str or None (Default=None)
        The column to be used as a factor variable which groups data for 
        plotting.

    facet_col : str or None (Default=None)
        The param_grid key that is used to assign marks ot facetted subplots
        in the horizontal direction.  

    facet_col_wrap : int (Default=2)
        The maximum number of facet columns.  Wraps the column variable 
        at this width, so that the column facets span multiple rows.        

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
    def __init__(self, estimator, fig=None, param_grid=None, 
                 factor=None, facet_col=None, facet_col_wrap=2, **kwargs):
        super(CostCurve, self).__init__(fig=fig, estimator=estimator, **kwargs)    

        self.param_grid = param_grid
        self.factor = factor
        self.facet_col = facet_col
        self.facet_col_wrap = facet_col_wrap
        self.title = self.title or str(estimator.description + \
            "<br>Cost Curve Plot")

    def fit(self, X, y=None):
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
        self._validate(X, y)
        data = self._fit_plot_data(X, y)
        self.fig = self._fit_plot(data)

        return self


    def _validate(self, X, y):
        super(CostCurve, self)._validate(X,y)

        if self.param_grid:
            if not self.factor and not self.facet_col:
                raise ValueError("If using a param_grid, the color or facet_col\
                    parameters must be set.")
            if self.factor and self.factor not in self.param_grid:
                raise ValueError("The 'factor' parameter must be the 'param_grid'.")
            if self.facet_col and self.facet_col not in self.param_grid:
                raise ValueError("The 'facet_col' parameter must be the 'param_grid'.")
            if not isinstance(self.facet_col_wrap, int):
                raise ValueError("The 'facet_col_wrap' parameter must be an integer.")


    def _fit_plot_data(self, X, y):
        """Fits models and creates data in tidy format for plotting."""
        data = pd.DataFrame()

        # If a parameter grid was provided, set the parameters on the estimator.
        if self.param_grid:
            grid = ParameterGrid(self.param_grid)
            for params in grid:
                # Set params attributes on estimator
                d = {}
                for param, value in params.items():                    
                    setattr(self.estimator, param, value)
                    d[proper(param)] = round(value,4)

                self.estimator.fit(X,y)
                
                d['Epoch'] = np.arange(start=1, stop=self.estimator.history_.total_epochs+1)
                d['Cost'] = self.estimator.history_.epoch_log['train_cost']
                df = pd.DataFrame(d)
                data = pd.concat([data, df], axis=0)  

        # Otherwise, fit the model with current parameters as presented
        else:      
            self.estimator.fit(X,y)
            d = {}
            d['Epoch'] = np.arange(start=1, stop=self.estimator.history_.total_epochs+1)
            d['Cost'] = self.estimator.history_.epoch_log['train_cost']
            data = pd.DataFrame(d)

        return data
    
    def _fit_plot(self, data):
        """Creates the plot express object."""

        if self.factor and self.facet_col:
            fig = px.line(data, x='Epoch', y='Cost', color=proper(self.factor), 
                        facet_col=proper(self.facet_col),
                        facet_col_wrap=self.facet_col_wrap,
                        template=self.template,
                        title=self.title,
                        height=self.height,
                        width=self.width)        
        elif self.factor:
            fig = px.line(data, x='Epoch', y='Cost', color=proper(self.factor),
                        template=self.template,
                        title=self.title,
                        height=self.height,
                        width=self.width)

        elif self.facet_col:             
            fig = px.line(data, x='Epoch', y='Cost', 
                        facet_col=proper(self.facet_col),
                        facet_col_wrap=self.facet_col_wrap,
                        template=self.template,
                        title=self.title,
                        height=self.height,
                        width=self.width)                    
        else:
            fig = px.line(data, x='Epoch', y='Cost',
                        template=self.template,
                        title=self.title,
                        height=self.height,
                        width=self.width)            

        fig.update_layout(title_x=0.5)        
        return fig

# ---------------------------------------------------------------------------- #
#                              LEARNING CURVE                                  #
# ---------------------------------------------------------------------------- #        
class LearningCurve(ModelVisualatrix):        
    """Plots training and cross-validation scores by training sample size.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget object.
        The object being analyzed 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    
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

    def __init__(self, estimator, fig=None, cv=None, n_jobs=None, 
                 train_sizes=np.linspace(.1, 1.0, 5), **kwargs):
        super(LearningCurve, self).__init__(estimator=estimator, fig=fig, **kwargs)        

        self.cv = cv
        self.n_jobs = n_jobs
        self.train_sizes = train_sizes
        self.title = self.title or str(estimator.description + "<br>Learning Curve Plot")

    def fit(self, X, y):
        """Generates the learning curve plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        train_sizes, train_scores, test_scores = \
            learning_curve(self.estimator, X, y, cv=self.cv, n_jobs=self.n_jobs,
                           train_sizes=self.train_sizes, return_times=False)

        # Extract statistics
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Create confidence band by wrapping and reversing the data 
        # X line
        x = train_sizes
        x_rev = x[::-1]

        # Training Data
        train_upper = train_scores_mean + train_scores_std
        train_lower = train_scores_mean - train_scores_std
        train_lower = train_lower[::-1]

        # Validation Data
        test_upper = test_scores_mean + test_scores_std
        test_lower = test_scores_mean - test_scores_std
        test_lower = test_lower[::-1]        

        self.fig = go.Figure()

        # Plot training scores confidence bank
        self.fig.add_trace(go.Scatter(
            name='train_band',     
            mode='lines',       
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((train_upper,train_lower), axis=0),
            fillcolor='#005b96',
            line_color='rgba(255,255,255,0)',
            fill='toself',
            opacity=0.15,
            showlegend=False
        ))

        self.fig.add_trace(go.Scatter(
            name='test_band',
            mode='lines',
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((test_upper,test_lower), axis=0),
            fillcolor='rgb(27,158,119)',
            line_color='rgba(255,255,255,0)',
            fill='toself',
            opacity=0.15,
            showlegend=False
        ))

        # Plot training and validation lines
        self.fig.add_trace(go.Scatter(
            name='Training Scores',
            mode='lines+markers',
            x=x, y=train_scores_mean,
            line=dict(color='#005b96'),            
            showlegend=True
        ))

        self.fig.add_trace(go.Scatter(
            name='Cross-Validation Scores',
            mode='lines+markers',
            x=x, y=test_scores_mean,
            line=dict(color='rgb(27,158,119)'),             
            showlegend=True
        ))

        self.fig.update_layout(
            title=self.title,
            xaxis=dict(title='Training Samples'),
            yaxis=dict(title=self.estimator.scorer.label),
            title_x=0.5,
            template=self.template)

        
# ---------------------------------------------------------------------------- #
#                           MODEL SCALABILITY PLOT                             #
# ---------------------------------------------------------------------------- #        
class ModelScalability(ModelVisualatrix):        
    """Plots fit times against training samples.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget object.
        The object being analyzed 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.


    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))        
    
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

    def __init__(self, estimator, fig=None, cv=None, n_jobs=None, 
                 train_sizes=np.linspace(.1, 1.0, 5), **kwargs):
        super(ModelScalability, self).__init__(estimator=estimator, fig=fig, **kwargs)
        
        self.cv = cv
        self.n_jobs = n_jobs
        self.train_sizes = train_sizes
        self.title = self.title or str(estimator.description + "<br>Model Scalability Plot")

    def fit(self, X, y):
        """Generates the model scalability plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self.estimator, X, y, cv=self.cv, n_jobs=self.n_jobs,
                           train_sizes=self.train_sizes, return_times=True)

        fit_times_mean = np.mean(fit_times, axis=1)
        fit_times_std = np.std(fit_times, axis=1)

        # Plot fit times by training samples
        fit_times_upper = go.Scatter(
            x=train_sizes, y=fit_times_mean + fit_times_std,
            mode='lines',
            marker=dict(color="#b3cde0"),            
            fillcolor="#b3cde0",
            fill='tonexty',
            showlegend=False
        )
        fit_times = go.Scatter(
            name='Fit Times',
            x=train_sizes, y=fit_times_mean, 
            mode='lines+markers',             
            line=dict(color='#005b96'),            
            marker=dict(color='#005b96'),
            fillcolor="#b3cde0",
            fill='tonexty',            
            showlegend=False
        )
        fit_times_lower = go.Scatter(
            x=train_sizes, y=fit_times_mean - fit_times_std,
            mode='lines',
            line=dict(color='#b3cde0'),            
            showlegend=False
        )        

        # Load from bottom up
        data = [fit_times_lower, fit_times, fit_times_upper]
        # Update layout with designated template
        layout = go.Layout(
            xaxis=dict(title='Training Samples'),
            yaxis=dict(title='Fit Times'),
            title=self.title,title_x=0.5,
            template=self.template
        )
        self.fig = go.Figure(data=data, layout=layout)
                    
# ---------------------------------------------------------------------------- #
#                        MODEL LEARNING PERFORMANCE                            #
# ---------------------------------------------------------------------------- #        
class ModelLearningPerformance(ModelVisualatrix):        
    """Plots scores against fit times.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget object.
        The object being analyzed 

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.


    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))        
    
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

    def __init__(self, estimator, fig=None, cv=None, n_jobs=None, 
                 train_sizes=np.linspace(.1, 1.0, 5), **kwargs):
        super(ModelLearningPerformance, self).__init__(estimator=estimator,
                                                       fig=fig, **kwargs)

        self.cv = cv
        self.n_jobs = n_jobs
        self.train_sizes = train_sizes                                                       
        self.title = self.title or str(estimator.description + "<br>Model Learning Performance Plot")

    def fit(self, X, y):
        """Generates the model scalability plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self.estimator, X, y, cv=self.cv, n_jobs=self.n_jobs,
                           train_sizes=self.train_sizes, return_times=True)

        fit_times_mean = np.mean(fit_times, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)        

        # Plot fit times by training samples
        fit_times_upper = go.Scatter(
            x=fit_times_mean, y=test_scores_mean + test_scores_std,
            mode='lines',
            marker=dict(color="#b3cde0"),            
            fillcolor="#b3cde0",
            fill='tonexty',
            showlegend=False
        )
        fit_times = go.Scatter(
            name='Test Scores',
            x=fit_times_mean, y=test_scores_mean, 
            mode='lines+markers',             
            line=dict(color='#005b96'),            
            marker=dict(color='#005b96'),
            fillcolor="#b3cde0",
            fill='tonexty',            
            showlegend=False
        )
        fit_times_lower = go.Scatter(
            x=fit_times_mean, y=test_scores_mean - test_scores_std,
            mode='lines',
            line=dict(color='#b3cde0'),            
            showlegend=False
        )        

        # Load from bottom up
        data = [fit_times_lower, fit_times, fit_times_upper]
        # Update layout with designated template
        layout = go.Layout(
            xaxis=dict(title='Fit Times'),
            yaxis=dict(title=self.estimator.scorer.label),
            title=self.title,title_x=0.5,
            template=self.template
        )
        self.fig = go.Figure(data=data, layout=layout)
                
# ---------------------------------------------------------------------------- #
#                             LEARNING CURVES                                  #
# ---------------------------------------------------------------------------- #        
class LearningCurves(ModelVisualatrix):        
    """Plots all learning curve plots for a series of models.
    
    This class renders the set of learning curve plots for a designated
    set of models. For each model, the following plots are rendered:

        * Learning Curve
        * Model Scalability Plot
        * Model Learning Performance Plot.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget object
        The object being analyzed 

    estimators : a list of MLStudio estimator objects.
        The objects that implement the 'fit' and 'predict' methods.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 5-fold cross-validation,
        - integer, to specify the number of folds.
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.


    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))        
    
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

    def __init__(self, estimators, fig=None, cv=None, n_jobs=None, 
                 train_sizes=np.linspace(.1, 1.0, 5), **kwargs):
        super(LearningCurves, self).__init__(estimator=None, fig=fig, **kwargs)
        
        self.estimators = estimators
        self.cv = cv
        self.n_jobs = n_jobs
        self.train_sizes = train_sizes     
        self.title = self.title or "Learning Curve Analysis"

    def fit(self, X, y):
        """Generates the learning curve analysis plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        # Designate shape of subplots
        rows = len(self.estimators)
        cols = 3

        # Obtain subplot titles
        subtitles = []
        for estimator in self.estimators:
            subtitles.append(estimator.description + "<br>Learning Curve")
            subtitles.append(estimator.description + "<br>Model Scalability Plot")
            subtitles.append(estimator.description + "<br>Model Learning Performance")

        # Create subplots
        self.fig = plotly.tools.make_subplots(rows=rows, cols=cols,
                                               subplot_titles=subtitles,
                                               horizontal_spacing=0.3/cols,
                                               vertical_spacing = 0.3/rows)

        # Update font size for subplot titles.
        for i in self.fig['layout']['annotations']:
            i['font']['size'] = 12
        
        # Populate cells of subplot
        for i, estimator in enumerate(self.estimators):
            lc = LearningCurve(estimator, cv=self.cv, n_jobs=self.n_jobs, \
                train_sizes=self.train_sizes)
            ms = ModelScalability(estimator, cv=self.cv, n_jobs=self.n_jobs, \
                train_sizes=self.train_sizes)
            mp = ModelLearningPerformance(estimator, cv=self.cv, \
                n_jobs=self.n_jobs, train_sizes=self.train_sizes)
        
            # Fit the models
            lc.fit(X,y)
            ms.fit(X,y)
            mp.fit(X,y)
        
            # Extract traces from figures and append to subplots
            traces = lc.fig.data
            for j, trace in enumerate(traces):
                if i == 1 and j in [2,3]:
                    trace.showlegend = True 
                else:
                    trace.showlegend = False               
                self.fig.append_trace(trace, i+1, 1)
            traces = ms.fig.data
            for trace in traces:
                self.fig.append_trace(trace, i+1, 2)
            traces = mp.fig.data
            for trace in traces:
                self.fig.append_trace(trace, i+1, 3)                
        
            # Update subplot xaxis titles
            self.fig.update_xaxes(title_text="Training Samples", row=i+1, col=1)
            self.fig.update_xaxes(title_text="Training Samples", row=i+1, col=2)
            self.fig.update_xaxes(title_text="Fit Times", row=i+1, col=3)

            # Update subplot yaxis titles
            self.fig.update_yaxes(title_text=estimator.scorer.label, row=i+1, col=1)
            self.fig.update_yaxes(title_text="Fit Times", row=i+1, col=2)
            self.fig.update_yaxes(title_text=estimator.scorer.label, row=i+1, col=3)

        #  Add title and template to layout.
        self.fig.update_layout(title_text="Learning Curve Analysis",
                                template=self.template,
                                height=rows*400, width=1200,
                                title_x=0.5)

        # Update subplot title font size
        for i in self.fig['layout']['annotations']:
            i['font']['size'] = 12
# ---------------------------------------------------------------------------- #
#                            VALIDATION CURVE                                  #
# ---------------------------------------------------------------------------- #        
class ValidationCurve(ModelVisualatrix):        
    """Plots training and cross-validation scores by hyperparameter.

    Parameters
    ----------
    fig : Plotly Figure or FigureWidget object
        The object being analyzed

    estimator : MLStudio estimator object.
        The object that implements the 'fit' and 'predict' methods.

    param_name : str
        The parameter being evaluated

    param_range : array-like
        The range of the parameter being evaluated.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    scoring : str
        The metric used to evaluate performance.         
    
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

    def __init__(self, estimator, param_name, param_range, fig=None, 
                 cv=None, n_jobs=None, scoring="r2", **kwargs):
        super(ValidationCurve, self).__init__(estimator=estimator, 
                                              fig=fig, **kwargs)
        
        self.param_name = param_name
        self.param_range = param_range
        self.cv = cv
        self.n_jobs = n_jobs
        self.scoring = scoring                
        self.title = self.title or str(estimator.description + "<br>Validation Curve")

    def fit(self, X, y):
        """Generates the validation curve plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        """
        train_scores, test_scores = \
            validation_curve(self.estimator, X, y, 
            param_name=self.param_name,
            param_range=self.param_range, scoring=self.scoring, 
            n_jobs=self.n_jobs)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Training Data
        train_upper = train_scores_mean + train_scores_std
        train_lower = train_scores_mean - train_scores_std
        train_lower = train_lower[::-1]

        # Validation Data
        test_upper = test_scores_mean + test_scores_std
        test_lower = test_scores_mean - test_scores_std
        test_lower = test_lower[::-1]        

        self.fig = go.Figure()

        x = self.param_range
        x_rev = x[::-1]

        # Plot training scores confidence bank
        self.fig.add_trace(go.Scatter(
            name='train_band',     
            mode='lines',       
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((train_upper,train_lower), axis=0),
            fillcolor='#005b96',
            line_color='rgba(255,255,255,0)',
            fill='toself',
            opacity=0.15,
            showlegend=False
        ))

        self.fig.add_trace(go.Scatter(
            name='test_band',
            mode='lines',
            x=np.concatenate((x, x_rev), axis=0),
            y=np.concatenate((test_upper,test_lower), axis=0),
            fillcolor='rgb(27,158,119)',
            line_color='rgba(255,255,255,0)',
            fill='toself',
            opacity=0.15,
            showlegend=False
        ))

        # Plot training and validation lines
        self.fig.add_trace(go.Scatter(
            name='Training Scores',
            mode='lines+markers',
            x=x, y=train_scores_mean,
            line=dict(color='#005b96'),            
            showlegend=True
        ))

        self.fig.add_trace(go.Scatter(
            name='Cross-Validation Scores',
            mode='lines+markers',
            x=x, y=test_scores_mean,
            line=dict(color='rgb(27,158,119)'),             
            showlegend=True
        ))

        self.fig.update_layout(
            title=self.title,
            xaxis=dict(title=proper(self.param_name)),
            yaxis=dict(title=proper(self.scoring)),
            title_x=0.5,
            template=self.template)
