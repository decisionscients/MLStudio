# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \model_selection.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, June 30th 2020, 7:20:33 pm                         #
# Last Modified : Wednesday, July 1st 2020, 12:38:36 am                       #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Model selection classes"""
from abc import ABC, abstractmethod
from collections import OrderedDict


import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate

from mlstudio.utils.format import proper
from mlstudio.utils.data_analyzer import one_sample_ttest, critical_values
from mlstudio.models.base import BaseModelEvaluator
# --------------------------------------------------------------------------- #
class NestedCV(BaseModelEvaluator):
    """Model valuation via cross-validation
    
    Parameters
    ----------
    estimator : A scikit-learn compatible estimator object.
        The estimator to be evaluated
    
    cv : int (default=5)
        The number of cross-validation folds

    scoring : str, callable, list/tuple, or dict, (default=None)
        A single str (see `The scoring parameter: defining model 
        evaluation rules <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_) 
        or a callable (see `Defining your scoring strategy from metric 
        functions <https://scikit-learn.org/stable/modules/model_evaluation.html#scoring>`_) 
        to evaluate the predictions on the test set.

        For evaluating multiple metrics, either give a list of (unique) 
        strings or a dict with names as keys and callables as values.
        
        If None, the estimator’s score method is used.        

    alpha : float (default=0.05)
        The level of significance to use when computing parameter
        confidence intervals. 
    
    return_train_score : bool (default=True)
        Whether to include train_scores in the cross-validation results

    return_estimator : bool (default=True)
        Whether to return the estimators fitted on each split.

    feature_names : array-like of strings (default=None)
        An array containing feature names including the intercept and all
        coeficients.

    top_n_features : int in range [1,20] (default=20)
        Absolute values of feature parameters serve as a proxy for 
        feature importance. This parameter places an upper limit on the
        number of features (sorted by the absolute value of the 
        parameter value descending) to analyze.

    Attributes
    ----------
    evaluation_results : dict of float arrays of shape (n_splits,) or (n_splits,n_features)
        The results of the cross-validation. The keys of this dict include:

        test_score : array_like of shape (n_splits,)
            The score array for test scores on each cv split. For multiple
            scoring metrics, the suffix _score in test_score changes
            to the specific metric i.e. test_r2 or test_accuracy.

        test_score_stats : DataFrame of shape(n_splits, 7)
            Pandas DataFrame object containing test_score statistics
            over cv splits. Columns include 'Test Score', mean
            standard deviation, standard error, t_statistic, p_value
            and confidence interval for the test_scores.

        train_score : array_like of shape (n_splits,)
            The score array for train scores on each cv split. For multiple
            scoring metrics, the suffix _score in train_score changes
            to the specific metric i.e. train_r2 or train_accuracy.            

        train_score_stats : DataFrame of shape(n_splits, 7)
            Pandas DataFrame object containing train_score statistics
            over cv splits. Columns include 'Train Score', mean
            standard deviation, standard error, t_statistic, p_value
            and confidence interval for the train_scores.       

        fit_time : array_like of shape (n_splits,)     
            The time for training the estimator for each cv split

        fit_time_stats : DataFrame of shape(n_splits, 7)
            Pandas DataFrame object containing fit_time statistics
            over cv splits. Columns include 'Fit Time', mean
            standard deviation, standard error, t_statistic, p_value
            and confidence interval for the fit_times.

        score_time : array_like of shape (n_splits,)     
            The time for training the estimator for each cv split

        score_time_stats : DataFrame of shape(n_splits, 7)
            Pandas DataFrame object containing score_time statistics
            over cv splits. Columns include 'Score Time', mean
            standard deviation, standard error, t_statistic, p_value
            and confidence interval for the score_times.            

        parameters : DataFrame of shape (n_splits, n_features)
            Dataframe containing all parameter estimates for each
            cv split

        top_n_parameters : DataFrame of shape (n_splits, 
            min(max(1,min(top_n_features,20)),estimator.n_features_)
            DataFrame containing the top_n_parameters and
            their values.

        top_n_parameters_stats : DataFrame of shape(1,
            min(max(1,min(top_n_features,20)),estimator.n_features_)
            The mean, standard deviation, standard error, t_statistic,
            p_value and confidence interval for each of the top_n_parameters. 


    """    
    def _perform_cross_validation(self, X, y):
        """Performs KFold cross-validation using 'cv' folds."""
        cv_results = cross_validate(estimator=self.estimator, X=X, y=y, 
                        cv=self.cv, return_train_score=True, 
                        return_estimator=True)
        return cv_results

    def _create_dummy_feature_names(self, cv_results):
        """Creates dummy features of the format "Xn", where n is the feature index."""
        estimators = cv_results.get('estimator')
        n_features = estimators[0].n_features_
        feature_names = np.array(["X" + str(n) for n in range(n_features)])
        return feature_names

    def _create_parameters_df(self, cv_results):
        """Creates a dataframe of shape (n_splits, n_features) of parameter values."""
        df_parameters = pd.DataFrame()
        estimators = cv_results['estimator']
        for idx, estimator in enumerate(estimators):
            d = {}            
            split = "Split_" + str(idx)
            d[split] = estimator.theta_
            df_theta = pd.DataFrame.from_dict(data=d, orient='index',
                                    columns=self.feature_names) 
            df_parameters = pd.concat([df_parameters, df_theta], axis=0)
        return df_parameters

    def _extract_top_n_features(self, df_params):
        """Returns a dataframe containing the top_n features by importance."""
        # Compute importance as absolute value of mean parameter values
        feature_importances = np.abs(df_params.mean(axis=0).to_numpy())
        # Compute top_n as minimum of top_n parameter and df_params.shape[1]
        top_n = min(self.top_n, df_params.shape[1])
        # Extract an array of indices for top_n columns by importance
        top_n_feature_indices = np.argpartition(feature_importances, \
            -top_n)[-top_n:] 
        # Subset the parameters dataframe by these indices
        df_top_n_features = df_params.iloc[:,list(top_n_feature_indices)]        
        return df_top_n_features

    def _compute_feature_stats(self, feature_name, feature_parameters, alpha=0.05):
        """Computes descriptive statistics, and confidence interval for an individual feature"""
        # Compute degrees of freedom for one sample ttest        
        df = len(feature_parameters) - 1 
        # Compute critical values based upon significance and degrees of freedom
        cv = critical_values(feature_parameters, df, alpha)
        fs = pd.DataFrame()
        fs['feature'] = feature_name
        fs['mean'] = np.mean(feature_parameters)
        fs['std'] = np.std(feature_parameters)
        fs['se'] = fs['std'] / np.sqrt(len(feature_parameters))
        fs['t'], fs['p_value'] = one_sample_ttest(feature_parameters)
        fs['lower_ci'] = fs['mean'] - cv * fs['se']
        fs['upper_ci'] = fs['mean'] + cv * fs['se']
        return fs

    def _compute_parameter_stats(self, top_features):
        """Computes parameter statistics for all features and splits"""
        parameter_stats = pd.DataFrame()
        for (feature_name, feature_parameters) in top_features.iteritems():
            feature_stats = self._compute_feature_stats(feature_name, 
                                    feature_parameters)
            parameter_stats = pd.concat((parameter_stats, feature_stats), axis=0)
        return parameter_stats 

    def _compute_feature_data(self, cv_results):
        if self.feature_names is None:
            self.feature_names = self._create_dummy_feature_names(cv_results)
        
        df_params = self._create_parameters_df(cv_results)
        cv_results['top_features_params'] = self._extract_top_n_features(df_params)   
        cv_results['top_features_params_stats'] = \
            self._compute_parameter_stats(cv_results['top_features_params'])
        return cv_results
