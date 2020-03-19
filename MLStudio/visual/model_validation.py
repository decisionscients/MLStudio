#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : model_diagnostics.py                                               #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Tuesday, March 17th 2020, 7:25:45 pm                         #
# Last Modified : Tuesday, March 17th 2020, 7:25:46 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
# ---------------------------------------------------------------------------- #
#                               RESIDUALS                                      #
# ---------------------------------------------------------------------------- #        
class Residuals(Visualatrix):        
    """Plots residuals versus predicted values.

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
        super(Residuals, self).__init__(**kwargs)
        self._estimator = estimator     
        self._title = self._title or str(estimator.description + "<br>Residuals vs. Predicted")

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
        self._estimator.fit(X,y)
        y_pred = self._estimator.predict(X)

        # Compute best fit line predicted vs actual
        y = y.reshape(-1,1)
        est = LinearRegression(gradient_descent=False)
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
            title=self._title,title_x=0.5,
            template=self._template
        )
        self.fig = go.Figure(data=data, layout=layout)
