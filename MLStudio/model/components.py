# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \component.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Sunday, July 12th 2020, 5:17:47 am                          #
# Last Modified : Sunday, July 12th 2020, 5:17:47 am                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Model Development classes"""
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import deepcopy


import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_validate, learning_curve, validation_curve
from sklearn.model_selection import StratifiedKFold, KFold, RepeatedKFold
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import check_X_y, check_array

from mlstudio.utils.format import proper
from mlstudio.utils.data_analyzer import one_sample_ttest, critical_values
# --------------------------------------------------------------------------- #
class Model(ABC):
    """Abstract base class for learning models.

    Models are mathematical representation of real-world processes. Distinct
    from a machine learning algorithm, machine learning models are specific
    incantations of a process and represents phenomena.

    The lifecycle of a model begins when it 'wins' a model selection 
    search. GridSearchCV, RandomSearchCV are two cross validation processes
    that can produce a model instance.
    
    Parameters
    ----------
    data : Data object containing the data upon which model was trained.

    estimator : Scikit-Learn compatible estimator object
        The algorithm to be evaluated

    method : Method object or subclass
        The object representing the process that created the Model.        

    environment : Environment object
        The system, hardware, and software environment in which the model was created.
        
    algorithm : Algorithm object 
        Object representing the algorithm and its hyperparameters.

    evaluation : Evaluation object
        Object containing the metrics and values from training and cross-validation.

    name : str
        Suggested a 35 character name for the object.

    type : str
        The model type: This pertains to the machine learning task for which
        the Model was created, for instance, 'regression'.

    desc : str (default=name of Estimator object.)
        Up to 80 character description of the object. 

    filepath : str
        The relative filename for object persistence.    

    analysis : Analysis object
        Statistics and visualizations

    estimation : Performance estimation object.
        Object containing empirical risk minimization and PAC performance
        estimation data.

    precedent : object
        The Model object from which this model was derived.

    decendents : dict
        Key value pairs of Model objects derived from this object. 

    """    

    def __init__(self, estimator, method, algorithm, evaluation=None,
                 name=None, model_type=None, desc=None, filepath=None, analysis=None,
                 environment=None, precedent=None):       

        self.estimator = estimator
        self.method = method
        self.algorithm = algorithm
        self.evaluation = evaluation
        self.name = name if name else self.estimator.name
        self.model_type = model_type
        self.desc = desc
        self.filepath = filepath
        self.analysis = analysis
        self.environment = environment
        self.precedent = precedent

    @property
    def algorithm_name(self):
        return self.algorithm.name

    @property
    def parameters(self):
        return self.estimator.get_params()

    @property
    def hyperparameters(self):
        return self.estimator.get_params()

    @property
    def best_score(self):
        return self.method.cv_results_.best_score
    
    @property
    def mean_test_score(self):
        return self.method.cv_results_.mean_test_score

    @property
    def mean_train_score(self):
        return self.method.cv_results_.mean_train_score        

    def fit(self, X, y):
        """Fits the model on the data."""
        self.estimator.fit(X, y)

    def predict(self, X):
        """Computes predictions."""
        return self.estimator.predict(X)

    def score(self, X, y):
        return self.estimator.score(X, y)





