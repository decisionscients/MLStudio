#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : monitoring.py                                                        #
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
from abc import ABC, abstractmethod
import datetime
import numpy as np
import pandas as pd

from mlstudio.supervised.algorithms.optimization.observers.base import Observer, PerformanceObserver
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

# --------------------------------------------------------------------------- #
#                                SUMMARY                                      #
# --------------------------------------------------------------------------- #           
class Summary(Observer):
    """Optimization summary class."""
    
    def __init__(self, printer=None):
        super(Summary, self).__init__()
        self.printer = printer
        self.name = "Summary"

    def _optimization_summary(self):
        """Prints optimization summary information."""
        
        bb = self.model.blackbox_

        optimization_summary = {'Name': self.model.description,
                                'Start': str(bb.start),
                                'End': str(bb.end),
                                'Duration': str(bb.duration) + " seconds.",
                                'Epochs': str(bb.total_epochs),
                                'Batches': str(bb.total_batches)}
        self.printer.print_dictionary(optimization_summary, "Optimization Summary")             


    def _early_stop_summary(self):        
        """Renders early stop optimization data."""

        log = self.model.blackbox_.epoch_log
        datasets = {'train': 'Training', 'val': 'Validation'}
        keys = ['train', 'val']
        metrics = ['cost', 'score']
        print_data = []
        # Format labels and data for printing from result parameter
        for performance in list(itertools.product(keys, metrics)):
            d = {}
            key = performance[0] + '_' + performance[1]
            if log.get(key):
                label = datasets[performance[0]] + ' ' + proper(performance[1]) 
                d['label'] = label
                if performance[1] == 'score' and hasattr(self.model, 'scorer'):                    
                    d['data'] = str(np.round(log.get(key)[-1],4)) + " " + self.model.scorer.name
                else:
                    d['data'] = str(np.round(log.get(key)[-1],4)) 
                print_data.append(d) 

        # Print performance statistics
        performance_summary = OrderedDict()
        for i in range(len(print_data)):
            performance_summary[print_data[i]['label']] = print_data[i]['data']
        title = "Performance Summary"
        self.printer.print_dictionary(performance_summary, title)        

    def _hyperparameter_summary(self):
        """Displays model hyperparameters."""

        hyperparameters = OrderedDict()
        def get_params(o):
            params = o.get_params()
            for k, v in params.items():
                if isinstance(v, (str, dict, bool, int, float, np.ndarray, \
                    np.generic, list)) or v is None:
                    k = o.__class__.__name__ + '__' + k
                    hyperparameters[k] = str(v)
                else:
                    get_params(v)
        get_params(self.model)

        self.printer.print_dictionary(hyperparameters, "Model HyperParameters")             

    def report(self):
          
        self._optimization_summary(model=self.model).report()
        self._early_stop_summary(model=self.model).report()
        self._hyperparameter_summary(model=self.model).report()
