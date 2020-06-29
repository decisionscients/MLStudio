#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : history.py                                                        #
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

    def __init__(self):
        super(BlackBox, self).__init__()
        self.name = "BlackBox"

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
        # Computes percent change in metrics in epoch_log.
        self._compute_pct_change()
        # Extracts final results and updates the model object.
        final_results = {}
        for k, v in self.epoch_log.items():
            final_results[k] = v[-1]
        self.model.final_results_ = final_results

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

    def _compute_pct_change(self):
        """Computes percent change in selected metrics."""
        metrics = ['train_cost', 'val_cost', 
                   'train_score', 'val_score', 'gradient_norm']
        for metric in metrics:
            new_metric = metric + '_pct_change'
            if self.epoch_log.get(metric):
                metric_series = pd.Series(self.epoch_log.get(metric))
                self.epoch_log[new_metric] = \
                    metric_series.pct_change().fillna(0).to_numpy()
   

# --------------------------------------------------------------------------- #
#                                PROGRESS                                     #
# --------------------------------------------------------------------------- #              
class Progress(Observer):
    """Class that reports progress at designated points during training."""

    def __init__(self):
        super(Progress, self).__init__()
        self.name = "Progress"    
    
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
        