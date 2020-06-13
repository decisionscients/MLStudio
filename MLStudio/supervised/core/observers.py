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
import pandas as pd
import types

from sklearn.base import BaseEstimator

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

    def __init__(self, metric='train_cost', scorer=None, epsilon=1e-3, patience=5): 
        super(Performance, self).__init__()       
        self.name = "Performance Observer"
        self.metric = metric        
        self.scorer = scorer
        self.epsilon = epsilon
        self.patience = patience

    @property
    def best_results(self):
        return self._best_results
       
    def _validate(self):        
        validate_metric(self.metric)
        if 'score' in self.metric:
            validate_scorer(self.scorer)
        validate_zero_to_one(param=self.epsilon, param_name='epsilon',
                             left='closed', right='closed')       
        validate_int(param=self.patience, param_name='patience',
                     minimum=0, left='open', right='open')

    def initialize(self, logs=None):                
        """Sets key variables at beginning of training.        
        
        Parameters
        ----------
        log : dict
            Contains no information
        """        
        logs = logs or {}        
        self._validate()
        # Private variables
        self._baseline = None        
        self._iter_no_improvement = 0
        self._better = None   
        self._stabilized = False   
        self._significant_improvement = False

        # log data
        self._epoch_log = []
        self._performance_log = []
        self._baseline_log = []
        self._relative_change_log = []
        self._improvement_log = []
        self._iter_no_improvement_log = []
        self._stability_log = []
        self._best_epochs_log = []                       
        
        # If 'score' is the metric and the scorer object exists, 
        # obtain the 'better' function from the scorer.        
        if 'score' in self.metric and self.scorer:            
            self._better = self.scorer.better
        # Otherwise, the better function is np.less since we improve be reducing
        # cost or the magnitudes of the gradient                
        else:
            self._better = np.less

    def _update_log(self, current, logs):
        """Creates log dictionary of lists of performance results."""
        self._epoch_log.append(logs.get('epoch'))
        self._performance_log.append(logs.get(self.metric))
        self._baseline_log.append(self._baseline)
        self._relative_change_log.append(self._relative_change)
        self._improvement_log.append(self._significant_improvement)
        self._iter_no_improvement_log.append(self._iter_no_improvement)
        self._stability_log.append(self._stabilized)
        self._best_epochs_log.append(self._best_epoch)

    def get_log(self):
        """Returns log in pandas dataframe format."""
        d = {'Epoch': self._epoch_log, 'Performance': self._performance_log,
             'Baseline': self._baseline_log, 'Relative Change': self._relative_change_log,
             'Improvement': self._improvement_log,
             'Iter No Improvement': self._iter_no_improvement_log,
             'Stability': self._stability_log,
             'Best Epochs': self._best_epochs_log        
        }
        df = pd.DataFrame(data = d)
        return df


    def _metric_improved(self, current):
        """Returns true if the direction and magnitude of change indicates improvement"""
        # Determine if change is in the right direction.
        if self._better(current, self._baseline):
            return True
        else:
            return False

    def _significant_relative_change(self, current):        
        self._relative_change = abs(current-self._baseline) / abs(self._baseline)
        return self._relative_change > self.epsilon                

    def _process_improvement(self, current, logs):
        """Sets values of parameters and attributes if improved."""
        self._iter_no_improvement = 0            
        self._stabilized = False
        self._baseline = current 
        self._best_epoch = logs.get('epoch')        

    def _process_no_improvement(self):
        """Sets values of parameters and attributes if no improved."""    
        self._iter_no_improvement += 1  
        if self._iter_no_improvement == self.patience:
            self._iter_no_improvement = 0
            self._stabilized = True               

    def _get_current_value(self, logs):
        """Obtain the designated metric from the logs."""
        current = logs.get(self.metric)
        if not current:
            msg = "{m} was not found in the log.".format(m=self.metric)
            raise KeyError(msg)     
        return current

    def _initialize_iteration(self):
        """Resets state for each iteration."""
        self._significant_improvement = False
        self._relative_change = 0
        self._stabilized = False

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

        # Initialize iteration
        self._initialize_iteration()

        # Handle first iteration as an improvement by default
        if self._baseline is None:
            self._significant_improvement = True
            self._process_improvement(current, logs)    

        # Otherwise, evaluate the direction and magnitude of the change        
        else:
            self._significant_improvement = self._metric_improved(current) and \
                self._significant_relative_change(current)

            if self._significant_improvement:
                self._process_improvement(current, logs)
            else:
                self._process_no_improvement()

        # Log results
        self._update_log(current, logs)

        return self._significant_improvement, self._stabilized, self._best_epoch


