# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \base.py                                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, June 30th 2020, 7:29:38 pm                         #
# Last Modified : Tuesday, June 30th 2020, 7:29:38 pm                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Base classes for model diagnostics, evaluation and selection classes."""
from abc import ABC, abstractmethod
from sklearn.base import BaseEstimator
from sklearn.metrics import get_scorer
from mlstudio.utils.validation import check_is_fitted
# --------------------------------------------------------------------------- #
class BaseNestedCV(BaseEstimator, ABC):
    """Abstract base class for all nested cross-validation classes.
    
    Parameters
    ----------
    estimators : dict of scikit-learn compatible estimators.
        Dictionary containing one or more key value pairs in which the
        key is an estimator name and the value is a scikit-learn
        compatible estimator.

    pipelines : dict of Pipeline objects.
        Dictionary of Pipeline objects where the key is the Pipeline
        object name and the value is the Pipeline object. 

    param_grids : dict of parameter grids.
        Dictionary where the key is parameter grid name and the value
        is a list containing a dictionary of parameter key value pairs.

    inner_cv : int, cross-validation generator or an iterable (default=None)
        Determines the cross validation splitting strategy. Possible inputs
        for inner_cv are:

            * None: Use the default 5-fold cross validation
            * int: The number of folds in (Stratified)KFold
            * CV Splitter: Cross-validation generator

    outer_cv : int, cross-validation generator or an iterable (default=None)
        Determines the cross validation splitting strategy. Possible inputs
        for outer_cv are:

            * None: Use the default 5-fold cross validation
            * int: The number of folds in (Stratified)KFold
            * CV Splitter: Cross-validation generator

    scoring : callable  (default=None)
        A scoring function used to evaluate the predictions on the 
        test set. If None, the estimator’s score method is used.     

    alpha : float (default=0.05)
        The level of significance to use when computing critical values
        and confidence intervals.            

    n_jobs : int (default=None)
        Number of jobs to run in parallel. None means 1 and -1 means use all
        processors.

    pre_dispatch : int, or str, (default=n_jobs)
        Controls the number of jobs that get dispatched during parallel 
        execution. Reducing this number can be useful to avoid an 
        explosion of memory consumption when more jobs get dispatched 
        than CPUs can process. This parameter can be:

        * None, in which case all the jobs are immediately created and 
            spawned. Use this for lightweight and fast-running jobs, 
            to avoid delays due to on-demand spawning of the jobs
        * int, giving the exact number of total jobs that are spawned
        * str, giving an expression as a function of n_jobs, as in ‘2*n_jobs’        
    
    refit : bool (default=True)
        If True, refit an estimator using the best found parameters on the 
        whole dataset.
        
    verbose : int (default=0)
        Controls the verbosity: the higher, the more messages.


    Attributes
    ----------
    gscv_results_ : dict of numpy arrays
        Dict contains GridsearchCV results including parameters, fit times, 
        score times, train and test scores. Mean and standard deviations of 
        scores and fit times are also included.

    gscv_best_estimator_ : estimator
        Estimator that was chosen by GridsearchCV based upon highest score.

    gscv_best_score_ : float
        Mean cross-validated sore of the best estimator

    gscv_best_params_ : dict
        Parameter setting that gave the best results during GridsearchCV

    gscv_best_index_ : int
        The index (of gscv_reults_ arrays) corresponding to the best
        candidate parameter setting.

    scorer_ : function 
        Scorer function used on validation data to choose the best parameters 
        for the model.         

    n_outer_repeats_ : int
        The number of repeated cross-validations performed on the outer loop.

    n_outer_splits_ : int
        The number of splits used on the outer loop of nested cross-validation.

    n_inner_repeats_ : int
        The number of repeated cross-validations performed on the inner loop.

    n_inner_splits_ : int
        The number of splits used on the inner GridsearchCV loop.        

    refit_time_ : float
        Seconds used to refit the best model on the whole dataset.

    """

    @abstractmethod
    def __init__(self, *estimators, *pipelines, *param_grids, inner_cv=5,
                 outer_cv=5, scoring=None, alpha=0.05, n_jobs=None, 
                 pre_dispatch='2*n_jobs', refit=True, verbose=0,
                 return_train_score=True, random_state=None):
        self.estimators = estimators
        self.pipelines = pipelines
        self.param_grids = param_grids
        self.inner_cv = inner_cv
        self.outer_cv = outer_cv
        self.scoring = scoring
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.refit = refit
        self.verbose = verbose
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, y):
        """Performs model selection and evaluates best model performance. 
        
        Parameters
        ----------
        X : array_like of shape, (n_samples, n_features)
            Input data. Note an intercept term is added by the estimator
            during training. Hence, estimator.n_features_ will include
            a bias term

        y : array_like of shape (n_samples,)
            Target value
        """
        pass

    def predict(self, X):
        """Calls predict on the best estimator.
        
        Parameters
        ----------
        X : array_like of shape(n_samples, n_features)
            Input data for which the predictions are computed.

        """
        return self.gscv_best_estimator_.predict(X)

    def predict_proba(self, X):
        """Calls predict_proba on the best estimator if supported.
        
        Parameters
        ----------
        X : array_like of shape(n_samples, n_features)
            Input data for which the predictions are computed.
                
        """
        return self.gscv_best_estimator_.predict_proba(X)        

    def predict_log_proba(self, X):
        """Calls predict_log_proba on the best estimator if supported.
        
        Parameters
        ----------
        X : array_like of shape(n_samples, n_features)
            Input data for which the predictions are computed.
        
        """
        return self.gscv_best_estimator_.predict_log_proba(X)        

    def score(self, X, y=None):
        """Returns score on the given data.
        
        This uses the score defined by the scoring parameter if provided;
        otherwise, use gscv_best_estimator_.score method.

        Parameters
        ----------
        X : array_like of shape (n_samples, n_features)
            Input data for which the score is being computed

        y : array_like of shape (n_samples,) or (n_samples, n_classes)
            Targets relative to X

        """
        if self.scorer_ is None:
            score = gscv_best_estimator_.score(X, y)        
        else:
            y_pred = self.predict(X)
            try:
                score = self.scorer_(y, y_pred)
            except:
                msg = "self.scorer_ is not a valid score function."
                raise ValueError(msg)
        return score
