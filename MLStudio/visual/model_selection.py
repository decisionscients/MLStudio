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
from ipywidgets import interactive, HBox, VBox
import math
import numpy as np
import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go 
import plotly.offline as py
from sklearn.model_selection import ParameterGrid, learning_curve 

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
        self._title = self._title or str(estimator.description + "<br>Cost Curve Plot")

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
                
                d['Epoch'] = np.arange(start=1, stop=self._estimator.history_.total_epochs+1)
                d['Cost'] = self._estimator.history_.epoch_log['train_cost']
                df = pd.DataFrame(d)
                data = pd.concat([data, df], axis=0)  
        else:      
            self._estimator.fit(X,y)
            d = {}
            d['Epoch'] = np.arange(start=1, stop=self._estimator.history_.total_epochs+1)
            d['Cost'] = self._estimator.history_.epoch_log['train_cost']
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

        fig.update_layout(title_x=0.5)        
        return fig

# ---------------------------------------------------------------------------- #
#                              LEARNING CURVE                                  #
# ---------------------------------------------------------------------------- #        
class LearningCurve(Visualatrix):        
    """Plots training and cross-validation scores by training sample size.

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
        super(LearningCurve, self).__init__(**kwargs)
        self._estimator = estimator     
        self._title = self._title or str(estimator.description + "<br>Learning Curve Plot")

    def fit(self, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """Generates the learning curve plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

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

        """
        train_sizes, train_scores, test_scores = \
            learning_curve(self._estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, return_times=False)

        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plot training scores line and ribbon
        train_upper = go.Scatter(
            x=train_sizes, y=train_scores_mean + train_scores_std,
            mode='lines',
            marker=dict(color="#b3cde0"),            
            fillcolor="#b3cde0",
            fill='tonexty',
            showlegend=False
        )
        train = go.Scatter(
            name='Training Scores',
            x=train_sizes, y=train_scores_mean, 
            mode='lines+markers',  
            line=dict(color='#005b96'),            
            marker=dict(color='#005b96'),
            fillcolor="#b3cde0",
            fill='tonexty',            
            showlegend=False
        )
        train_lower = go.Scatter(
            x=train_sizes, y=train_scores_mean - train_scores_std,
            mode='lines',
            line=dict(color='#b3cde0'),            
            showlegend=False
        )        

        # Plot validation scores line and ribbon
        val_upper = go.Scatter(
            x=train_sizes, y=test_scores_mean + test_scores_std,
            mode='lines',
            marker=dict(color='rgba(179,226,205, 0.5)'),
            line=dict(width=0),
            fillcolor="rgba(179,226,205, 0.5)",
            fill='tonexty',
            showlegend=False
        )
        val = go.Scatter(
            name='Cross-Validation Scores',
            x=train_sizes, y=test_scores_mean,
            mode='lines+markers',             
            line=dict(color='rgb(27,158,119)'), 
            marker=dict(color='rgb(27,158,119)'),           
            fillcolor="rgba(179,226,205, 0.5)",
            fill='tonexty',            
            showlegend=False
        )
        val_lower = go.Scatter(
            x=train_sizes, y=test_scores_mean - test_scores_std,
            mode='lines',
            line=dict(color="rgba(179,226,205, 0.5)"),
            showlegend=False
        )                
        # Load from bottom up
        data = [val_lower, val, val_upper, train_lower, train, train_upper]
        # Update layout with designated template
        layout = go.Layout(
            xaxis=dict(title='Training Samples'),
            yaxis=dict(title=self._estimator.get_scorer().label),
            title=self._title,title_x=0.5,
            template=self._template
        )
        self._fig = go.Figure(data=data, layout=layout)

        
# ---------------------------------------------------------------------------- #
#                           MODEL SCALABILITY PLOT                             #
# ---------------------------------------------------------------------------- #        
class ModelScalability(Visualatrix):        
    """Plots fit times against training samples.

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
        super(ModelScalability, self).__init__(**kwargs)
        self._estimator = estimator     
        self._title = self._title or str(estimator.description + "<br>Model Scalability Plot")

    def fit(self, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """Generates the model scalability plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

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

        """
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self._estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, return_times=True)

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
            title=self._title,title_x=0.5,
            template=self._template
        )
        self._fig = go.Figure(data=data, layout=layout)
                    
# ---------------------------------------------------------------------------- #
#                           MODEL SCALABILITY PLOT                             #
# ---------------------------------------------------------------------------- #        
class ModelLearningPerformance(Visualatrix):        
    """Plots scores against fit times.

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
        super(ModelLearningPerformance, self).__init__(**kwargs)
        self._estimator = estimator     
        self._title = self._title or str(estimator.description + "<br>Model Learning Performance Plot")

    def fit(self, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """Generates the model scalability plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

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

        """
        train_sizes, train_scores, test_scores, fit_times, _ = \
            learning_curve(self._estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, return_times=True)

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
            yaxis=dict(title=self._estimator.get_scorer().label),
            title=self._title,title_x=0.5,
            template=self._template
        )
        self._fig = go.Figure(data=data, layout=layout)
                
# ---------------------------------------------------------------------------- #
#                             LEARNING CURVES                                  #
# ---------------------------------------------------------------------------- #        
class LearningCurves(Visualatrix):        
    """Plots all learning curve plots for a series of models.
    
    This class renders the set of learning curve plots for a designated
    set of models. For each model, the following plots are rendered:

        * Learning Curve
        * Model Scalability Plot
        * Model Learning Performance Plot.

    Parameters
    ----------
    estimators : a list of MLStudio estimator objects.
        The objects that implement the 'fit' and 'predict' methods.
    
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

    def __init__(self, estimators, **kwargs):
        super(LearningCurves, self).__init__(**kwargs)
        self._estimators = estimators     
        self._title = self._title or "Learning Curve Analysis"

    def fit(self, X, y, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
        """Generates the learning curve analysis plot

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples) or (n_samples, n_features), optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

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

        """
        # Designate shape of subplots
        rows = len(self._estimators)
        cols = 3

        # Obtain subplot titles
        subtitles = []
        for estimator in self._estimators:
            subtitles.append(estimator.description + "<br>Learning Curve")
            subtitles.append(estimator.description + "<br>Model Scalability Plot")
            subtitles.append(estimator.description + "<br>Model Learning Performance")

        # Create subplots
        self._fig = plotly.tools.make_subplots(rows=rows, cols=cols,
                                               subplot_titles=subtitles,
                                               horizontal_spacing=0.25/cols,
                                               vertical_spacing = 0.25/rows)

        # Update font size for subplot titles.
        for i in self._fig['layout']['annotations']:
            i['font']['size'] = 12
        
        # Populate cells of subplot
        for i, estimator in enumerate(self._estimators):
            lc = LearningCurve(estimator)
            ms = ModelScalability(estimator)
            mp = ModelLearningPerformance(estimator)
        
            # Fit the models
            lc.fit(X,y, cv, n_jobs, train_sizes)
            ms.fit(X,y, cv, n_jobs, train_sizes)
            mp.fit(X,y, cv, n_jobs, train_sizes)
        
            # Extract traces from figures and append to subplots
            traces = lc.fig.data
            for j, trace in enumerate(traces):
                if i == 1 and j in [1,4]:
                    trace.showlegend = True                
                self._fig.append_trace(trace, i+1, 1)
            traces = ms.fig.data
            for trace in traces:
                self._fig.append_trace(trace, i+1, 2)
            traces = mp.fig.data
            for trace in traces:
                self._fig.append_trace(trace, i+1, 3)                
        
            # Update subplot xaxis titles
            self._fig.update_xaxes(title_text="Training Samples", row=i+1, col=1)
            self._fig.update_xaxes(title_text="Training Samples", row=i+1, col=2)
            self._fig.update_xaxes(title_text="Fit Times", row=i+1, col=3)

            # Update subplot yaxis titles
            self._fig.update_yaxes(title_text=estimator.get_scorer().label, row=i+1, col=1)
            self._fig.update_yaxes(title_text="Fit Times", row=i+1, col=2)
            self._fig.update_yaxes(title_text=estimator.get_scorer().label, row=i+1, col=3)


        #  Add title to layout.
        self._fig.update_layout(title_text="Learning Curve Analysis",
        title_x=0.5)

        plotly.offline.plot(self._fig)    
                


            
