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
from collections import OrderedDict
import datetime
import itertools
import sys

import numpy as np
import pandas as pd

from mlstudio.supervised.algorithms.optimization.observers import base, debug
from mlstudio.supervised.algorithms.optimization.services.activations import Activation
from mlstudio.supervised.algorithms.optimization.services.loss import Loss
from mlstudio.supervised.algorithms.optimization.services.optimizers import Optimizer
from mlstudio.supervised.algorithms.optimization.services.regularizers import Regularizer
from mlstudio.supervised.algorithms.optimization.services.tasks import Task
from mlstudio.supervised.performance.base import BaseMeasure, BaseMetric
from mlstudio.utils.data_manager import dict_search
from mlstudio.utils.validation import validate_metric, validate_int
from mlstudio.utils.validation import validate_zero_to_one
from mlstudio.utils.format import proper
from mlstudio.utils.print import Printer

# --------------------------------------------------------------------------- #
#                                BLACKBOX                                     #
# --------------------------------------------------------------------------- #
class BlackBox(base.Observer):
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
        log = log or {}
        self.total_batches += 1
        for k,v in log.items():
            self.batch_log.setdefault(k,[]).append(v)            

    def on_epoch_end(self, epoch, log=None):
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
class Progress(base.Observer):
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
class Summary(base.Observer):
    """Optimization summary class."""

    def __init__(self, printer=None):
        super(Summary, self).__init__()
        self.printer = printer
        self.name = "Summary"

    def _optimization_summary(self):
        """Prints optimization summary information."""
        
        bb = self.model.get_blackbox()

        optimization_summary = {'Name': self.model.description,
                                'Start': str(bb.start),
                                'End': str(bb.end),
                                'Duration': str(bb.duration) + " seconds.",
                                'Epochs': str(bb.total_epochs),
                                'Batches': str(bb.total_batches)}
        self.printer.print_dictionary(optimization_summary, "Optimization Summary")             

    def _data_summary(self):
        """Prints summary of the data used for training (and validation)."""
        print("**********************************")
        print(self.model.train_data_package['X_train']['metadata'])

        X_train_metadata = self.model.train_data_package['X_train']['metadata']
        self.printer.print_dictionary(X_train_metadata, "Training Set (X)")
        y_train_metadata = self.model.train_data_package['y_train']['metadata']
        self.printer.print_dictionary(y_train_metadata, "Training Set (y)") 

        if self.model.train_data_package.get('X_val'):       
            X_val_metadata = self.model.train_data_package['X_val']['metadata']
            self.printer.print_dictionary(X_val_metadata, "Validation Set (X)")
            y_val_metadata = self.model.train_data_package['y_val']['metadata']
            self.printer.print_dictionary(y_val_metadata, "Validation Set (y)") 

    def _performance_summary(self):        
        """Renders early stop optimization data."""
        #TODO: update once evaluator observer is done.
        pass

        # log = self.model.blackbox.epoch_log
        # datasets = {'train': 'Training', 'val': 'Validation'}
        # keys = ['train', 'val']
        # metrics = ['cost', 'score']
        # print_data = []
        # # Format labels and data for printing from result parameter
        # for performance in list(itertools.product(keys, metrics)):
        #     d = {}
        #     key = performance[0] + '_' + performance[1]
        #     if log.get(key):
        #         label = datasets[performance[0]] + ' ' + proper(performance[1]) 
        #         d['label'] = label
        #         if performance[1] == 'score' and hasattr(self.model, 'metric'):                    
        #             d['data'] = str(np.round(log.get(key)[-1],4)) + " " + self.model.metric.name
        #         else:
        #             d['data'] = str(np.round(log.get(key)[-1],4)) 
        #         print_data.append(d) 

        # # Print performance statistics
        # performance_summary = OrderedDict()
        # for i in range(len(print_data)):
        #     performance_summary[print_data[i]['label']] = print_data[i]['data']
        # title = "Performance Summary"
        # self.printer.print_dictionary(performance_summary, title)    
        # 
    _hdf = pd.DataFrame()
    def _update_params(self, level, k, v):
        """Adds a k,v pair of params to the hyperparameter dataframe."""        
        if v:
            obj = k.split("__")[0]                        
            d = {'level': level, 'object':obj, 'Hyperparameter': k, 'Value': v}
            df = pd.DataFrame(data=d, index=[0])
            self._hdf = pd.concat([self._hdf, df])
        else:
            obj = k.split("__")[0]                        
            d = {'level': level, 'object':obj, 'Hyperparameter': k, 'Value': None}
            df = pd.DataFrame(data=d, index=[0])
            self._hdf = pd.concat([self._hdf, df])            


    def _get_params(self, obj):
        """Gets the hyperparameters for an object."""        
        self._implicit_dependencies = (debug.GradientCheck, BlackBox, Printer, 
                              Progress, base.ObserverList, BaseMetric,
                              BaseMeasure, self.__class__)        
        object_name = obj.__class__.__name__        
        params = obj.get_params()
        for k, v in params.items():
            if isinstance(v, (str, dict, bool, int, float, np.ndarray, \
                np.generic, list)) or v is None:                                
                level = k.count("__")
                k = object_name + "__" + k                                
                self._update_params(level, k, v)                        
            else:
                if not isinstance(v, self._implicit_dependencies):                    
                    self._update_params(level=0, k=" ", v=" ")                        
                    k = v.__class__.__name__
                    level = k.count("__")
                    self._update_params(level, k, v=" ")                        
                    self._get_params(v)        

    def _hyperparameter_summary(self):        
        """Displays model hyperparameters."""        
        
        self.printer.print_title("Model Hyperparameters")
        self._get_params(self.model)
        hdf = self._hdf.loc[self._hdf['level'] == 0]
        print(hdf[['Hyperparameter', 'Value']].to_string(index=False))


    def report(self):
          
        self._optimization_summary()
        self._data_summary()
        self._performance_summary()
        self._hyperparameter_summary()
