#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : director.py                                                       #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Saturday, June 13th 2020, 12:27:23 pm                       #
# Last Modified : Saturday, June 13th 2020, 12:58:40 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes that observe subject (estimator) performance."""
from collections import OrderedDict 
import copy
import datetime
import numpy as np
import pandas as pd
import types

from mlstudio.supervised.observers.base import Observer
from mlstudio.utils.validation import validate_metric, validate_scorer
from mlstudio.utils.validation import validate_zero_to_one, validate_int
# --------------------------------------------------------------------------- #
#                             PERFORMANCE                                     #
# --------------------------------------------------------------------------- #
class Performance(Observer):
    """Performances and logs model performance, critical points and stability. 

    Performance of the estimator object is the domain of concern for this class.
    We define performance in terms of:

        * Metric : A metric to observe. This can be training error, validation
            score, gradient norm or the like 

        * Epsilon : A mininum amount of relative change in the observed
            metric required to consider the optimization in a productive
            state.

        * Patience : The number of consecutive epochs or iterations of
            non-improvement that is tolerated before considering an
            optimization stabilized.    

    Parameters
    ----------
    mode : str 'active' or 'passive' (Default='passive')
        In 'active' mode, this observer signals the estimator to suspend
        optimization when performance hasn't improved. In 'passive' mode,
        the observer collects, analyzes and stores performance data, but
        does not effect the subject's behavior. 

    metric : str, optional (default='train_cost')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'gradient_norm': The norm of the gradient of the objective function w.r.t. theta

    epsilon : float, optional (default=0.01)
        The amount of relative change in the observed metric considered to be
        a sufficient improvement in performance. 

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that is 
        tolerated before considering the optimization stable.

    All estimator performance considerations are managed and controlled
    by this class. 
    """

    def __init__(self, mode='passive', metric='train_cost', scorer=None, 
                 epsilon=1e-3, patience=5): 
        super(Performance, self).__init__()       
        self.name = "Performance"
        self.mode = mode
        self.metric = metric        
        self.scorer = scorer
        self.epsilon = epsilon
        self.patience = patience

    @property
    def best_results(self):
        return self._best_results

    @property
    def critical_points(self):
        return self._critical_points

    def get_performance_data(self):
        d = {'Epoch': self._epoch_log, 'Performance': self._performance_log,
             'Baseline': self._baseline_log, 'Relative Change': self._relative_change_log,
             'Improvement': self._improvement_log,'Iters No Change': self._iter_no_improvement_log,
             'Stability': self._stability_log, 'Best Epochs': self._best_epochs_log}
        df = pd.DataFrame(data=d)
        return df
       
    def _validate(self):        
        validate_zero_to_one(param=self.epsilon, param_name='epsilon',
                             left='closed', right='closed')       
        validate_int(param=self.patience, param_name='patience',
                     minimum=0, left='open', right='open')

    def on_train_begin(self, logs=None):                
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

        # Implicit dependencies
        if 'score' in self.metric:
            try:                
                self._scorer = self.model.scorer
                self._better = self._scorer.better
            except:
                e = "The Performance Observer requires a scorer object for 'score' metrics."
                raise TypeError(e)
        else:
            self._better = np.less

        # Validation
        validate_metric(self.metric)
        validate_zero_to_one(param=self.epsilon, param_name='epsilon',
                             left='open', right='open')
        validate_int(param=self.patience, param_name='patience')

        # log data
        self._epoch_log = []
        self._performance_log = []
        self._baseline_log = []
        self._relative_change_log = []
        self._improvement_log = []
        self._iter_no_improvement_log = []
        self._stability_log = []
        self._best_epochs_log = []                       

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

    def _process_improvement(self, current, logs=None):
        """Sets values of parameters and attributes if improved."""
        self._iter_no_improvement = 0            
        self._stabilized = False
        self._baseline = current 
        self._best_epoch = logs.get('epoch')        

    def _process_no_improvement(self, logs=None):
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

    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """                  
        logs = logs or {}   
        
        # Initialize state variables        
        self._significant_improvement = False
        self._relative_change = 0
        self._stabilized = False
        
        # Obtain current performance
        current = self._get_current_value(logs)

        # Handle first iteration as an improvement by default
        if self._baseline is None:                             # First iteration
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

        # If performance has stabilized and the observer is in 'active' mode,
        # direct the subject that the optimization has converged.
        if self._stabilized and self.mode == 'active':
            self.model.converged = True
        return self

    def on_train_end(self, logs=None):
        """Logic executed at the end of training.
        
        Parameters
        ----------        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """    
        self._best_results = self._best_epochs_log[-1]
        self._critical_points = np.where(self._stability_log)[0].tolist()
        self._critical_points = [self._best_epochs_log[i] for i in self._critical_points] 

        print(self._critical_points)
        return self


