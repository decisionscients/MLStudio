#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : early_stop.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, May 15th 2020, 9:16:41 pm                           #
# Last Modified : Sunday, June 14th 2020, 11:49:38 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Early stop classes."""
from abc import ABC, abstractmethod
import datetime
import numpy as np
import pandas as pd

from mlstudio.supervised.observers.base import Observer, PerformanceObserver
from mlstudio.utils.validation import validate_metric, validate_int
from mlstudio.utils.validation import validate_zero_to_one
# --------------------------------------------------------------------------- #
#                               PERFORMANCE                                   #
# --------------------------------------------------------------------------- #
class Performance(PerformanceObserver):
    """Performances and log model performance, critical points and stability. 

    Performance is defined in terms of:

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

    val_size : float (default=0.3)
        The proportion of the training set to allocate to the 
        validation set.

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

    best_or_final : str, optional (default='best')
        Indicates whether the composing estimator should use the best
        or final results from the Performance object when reporting results
        and computing predictions

    All estimator performance considerations are managed and controlled
    by this class. 
    """

    def __init__(self, mode='passive', val_size=0.3, metric='val_score', 
                 epsilon=0.001, patience=5, best_or_final='best'): 
        super(Performance, self).__init__(                   
            metric = metric,
            epsilon = epsilon,
            patience = patience
        )
        self.name = "Performance EarlyStop Observer"
        self.mode = mode       
        self.best_or_final = best_or_final 
        self.val_size = val_size

    def on_train_begin(self, log=None):
        """Logic executed at the beginning of training.
        
        Parameters
        ----------
        log: dict
            Currently not used
        """
        super(Performance, self).on_train_begin(log=log)
        self.model.val_size = self.val_size                          

    def on_epoch_end(self, epoch, log=None):
        """Logic executed at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                  
        super(Performance, self).on_epoch_end(epoch=epoch, log=log)
        if self._stabilized and self.mode == 'active':
            self.model.converged = True       

    def on_train_end(self, log=None):
        """Logic executed at the end of training.

        Updates the model with the best results and the critical points.
        
        Parameters
        ----------        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """           
        self.model.best_results_ = self._best_results        
        self.model.critical_points_ = self._critical_points

