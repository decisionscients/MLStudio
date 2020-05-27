#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : observers.py                                                      #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 21st 2020, 8:04:28 am                         #
# Last Modified : Thursday, May 21st 2020, 8:04:41 am                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes that observe and report performance of models."""
from abc import ABC, abstractmethod, ABCMeta
from collections import OrderedDict 
import copy
import datetime
import numpy as np
import types

from sklearn.base import BaseEstimator
from tabulate import tabulate

from mlstudio.supervised.core.scorers import R2
from mlstudio.utils.print import Printer
from mlstudio.utils.validation import validate_metric, validate_scorer
from mlstudio.utils.validation import validate_zero_to_one, validate_int
# --------------------------------------------------------------------------- #
#                          OBSERVER BASE CLASS                                #
# --------------------------------------------------------------------------- #
class Observer(ABC, BaseEstimator):
    """Abstract base class for all observer classes."""

    @abstractmethod
    def __init__(self):   
        pass

    @abstractmethod
    def initialize(self, logs=None):
        pass

    @abstractmethod
    def model_is_stable(self, logs=None):
        pass

# --------------------------------------------------------------------------- #
#                             STABILITY                                       #
# --------------------------------------------------------------------------- #
class Performance(Observer):
    """Monitors performance and signals when performance has not improved. 
    
    Performance is measured in terms of training or validation cost and scores.
    To ensure that progress is meaningful, it must be greater than a 
    quantity epsilon. If performance has not improved in a predefined number
    of epochs in a row, the evaluation method returns false to the 
    calling object.

    Parameters
    ----------
    metric : str, optional (default='train_cost')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'gradient_norm': The norm of the gradient of the objective function w.r.t. theta

    epsilon : float, optional (default=0.0001)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, metric='train_cost', scorer=None, epsilon=0.01, patience=5):        
        self.name = "Performance Observer"
        self.metric = metric        
        self.scorer = scorer
        self.epsilon = epsilon
        self.patience = patience
       
    def _validate(self):        
        validate_metric(self.metric)
        if 'score' in self.metric:
            validate_scorer(self.scorer)
        validate_zero_to_one(param=self.epsilon, param_name='epsilon')       
        validate_int(param=self.patience, param_name='patience')

    def initialize(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """        
        # Attributes
        self.best_performance_ = None
        # Instance variables
        self._iter_no_improvement = 0
        self._better = None   
        self._stabilized = False   
        # For 'score' and 'cost' metrics, we measure improvement by changes 
        # in a specific direction. For 'gradient' and 'theta', we don't 
        # care about the direction of the change in so much as we care about
        # the magnitude of the change.
        self._directional_metric = self.metric in ['train_cost', 'train_score',
                                                   'val_cost', 'val_score']    
        
        logs = logs or {}
        self._validate()
        # If 'score' is the metric and the scorer object exists, 
        # obtain the 'better' function from the scorer.        
        if 'score' in self.metric and self.scorer:            
            self._better = self.scorer.better
        # Otherwise, the better function is np.less since we improve be reducing
        # cost or the magnitudes of the parameters                
        else:
            self._better = np.less

    def _print_results(self, current):
        """Prints current, best and relative change."""
        relative_change = abs(current-self.best_performance_) / abs(self.best_performance_)
        print("Iteration #: {i}  Best : {b}     Current : {c}   Relative change : {r}".format(\
                i=str(self._iter_no_improvement),
                b=str(self.best_performance_), 
                c=str(current),
                r=str(relative_change)))            

    def _evaluate_non_directional_change(self, current):
        """Returns true if the magnitude of the improvement is greater than epsilon."""
        relative_change = abs(current-self.best_performance_) / abs(self.best_performance_)
        return relative_change > self.epsilon

    def _evaluate_directional_change(self, current):
        """Returns true if the direction and magnitude of change indicates improvement"""
        # Determine if change is in the right direction.
        if self._better(current, self.best_performance_):
            return self._evaluate_non_directional_change(current)
        else:
            return False

    def _process_improvement(self, current, logs):
        """Sets values of parameters and attributes if improved."""
        self._iter_no_improvement = 0
        self.best_performance_ = current        
        self._stabilized = False        

    def _process_no_improvement(self):
        """Sets values of parameters and attributes if no improved."""        
        self._iter_no_improvement += 1  
        if self._iter_no_improvement == self.patience:
            self._stabilized = True       

    def _get_current_value(self, logs):
        """Obtain the designated metric from the logs."""
        current = logs.get(self.metric)
        if not current:
            msg = "{m} was not found in the log.".format(m=self.metric)
            raise KeyError(msg)     
        return current

    def model_is_stable(self, epoch, logs=None):
        """Determines whether performance is improving or stabilized.

        Parameters
        ----------
        epoch : int
            The current epoch number

        logs : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  

        Returns
        -------
        Bool if True convergence has been achieved. 

        """        
        logs = logs or {}
        # Obtain current performance
        current = self._get_current_value(logs)

        # Handle first iteration as an improvement by default
        if self.best_performance_ is None:
            self._process_improvement(current, logs)        

        # Otherwise, evaluate directional or non-directional performance
        else:
            if self._directional_metric:
                if self._evaluate_directional_change(current):
                    self._process_improvement(current, logs)
                else:
                    self._process_no_improvement()
            else:
                if self._evaluate_non_directional_change(current):
                    self._process_improvement(current, logs)
                else:
                    self._process_no_improvement()  

        if self._stabilized:
            return epoch - self._iter_no_improvement
        else:
            return False


