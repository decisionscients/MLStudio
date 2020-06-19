#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : monitor.py                                                        #
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
"""Module containing observers that monitor and report on optimization."""
import datetime
import numpy as np
import pandas as pd

from mlstudio.supervised.observers.base import Observer, PerformanceObserver
from mlstudio.utils.validation import validate_metric, validate_int
from mlstudio.utils.validation import validate_zero_to_one
# --------------------------------------------------------------------------- #
#                                BLACKBOX                                     #
# --------------------------------------------------------------------------- #
class BlackBox(Observer):
    """Repository for data obtained during optimization."""

    def on_train_begin(self, log=None):
        """Sets instance variables at the beginning of training.
        
        Parameters
        ----------
        log : Dict
            Dictionary containing the X and y data
        """ 
        self.total_epochs = 0
        self.total_batches = 0
        self.start = datetime.datetime.now()
        self.epoch_log = {}
        self.batch_log = {}

    def on_train_end(self, log=None):        
        """Sets instance variables at end of training.
        
        Parameters
        ----------
        log : Dict
            Not used 
        """
        self.end = datetime.datetime.now()
        self.duration = (self.end-self.start).total_seconds() 
        if self.model.verbose:
            self.report()

    def on_batch_end(self, batch, log=None):
        """Updates data and statistics relevant to the training batch.
        
        Parameters
        ----------
        batch : int
            The current training batch
        
        log : dict
            Dictionary containing batch statistics, such as batch size, current
            weights and training cost.
        """
        self.total_batches += 1
        for k,v in log.items():
            self.batch_log.setdefault(k,[]).append(v)        

    def on_epoch_begin(self, epoch, log=None):
        """Updates data and statistics relevant to the training epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch
        
        log : dict
            Dictionary containing data and statistics for the current epoch,
            such as weights, costs, and optional validation set statistics
            beginning with 'val_'.
        """
        log = log or {}
        self.total_epochs += 1
        for k,v in log.items():
            self.epoch_log.setdefault(k,[]).append(v)

   

# --------------------------------------------------------------------------- #
#                                PROGRESS                                     #
# --------------------------------------------------------------------------- #              
class Progress(Observer):
    """Class that reports progress at designated points during training."""
    
    def on_epoch_begin(self, epoch, log=None):
        """Reports progress at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch

        log : Dict
            Statistics obtained at end of epoch
        """
        if self.model.verbose:
            if not isinstance(self.model.verbose, int):
                raise TypeError("Verbose must be False or an integer. The \
                    integer indicates the number of epochs between each \
                    progress update.")            
            else:
                if epoch % self.model.verbose == 0:
                    items_to_report = ('epoch', 'train', 'val')
                    log = {k:v for k,v in log.items() if k.startswith(items_to_report)}
                    progress = "".join(str(key) + ': ' + str(np.round(value,4)) + ' ' \
                        for key, value in log.items())
                    print(progress)
        
# --------------------------------------------------------------------------- #
#                             PERFORMANCE                                     #
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

    def __init__(self, mode='passive', val_size=0.3, metric='train_cost', 
                 scorer=None,  epsilon=0.01, patience=5, best_or_final='best'): 
        super(Performance, self).__init__(                   
            val_size = val_size,
            metric = metric,        
            scorer = scorer,
            epsilon = epsilon,
            patience = patience
        )
        self.name = "Performance Observer"
        self.mode = mode       
        self.best_or_final = best_or_final 

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


