#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : text.py                                                           #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, June 14th 2020, 11:34:29 pm                         #
# Last Modified : Sunday, June 14th 2020, 11:56:46 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Text-based visualizations."""
from abc import ABC, abstractmethod
from collections import OrderedDict
import itertools 
import numpy as np
import pandas as pd
from tabulate import tabulate

from mlstudio.utils.format import proper
from mlstudio.supervised.observers.early_stop import EarlyStop
from mlstudio.utils.print import Printer

class Summary(ABC):
    """Base class for all optimization summary classes."""

    def __init__(self, model):
        self.model = model
        self._printer = Printer()

    @abstractmethod
    def report(self):
        pass

class OptimizationSummary(Summary):
    """Reports summary information for an optimization."""

    def _extract_data(self):
        """Extracts required data from the model."""
        bb = self.model.blackbox_
        data = {}
        data['start'] = bb.start
        data['end'] = bb.end
        data['duration'] = bb.duration
        data['epochs'] = bb.total_epochs
        data['batches'] = bb.total_batches
        return data

    def report(self):
        data = self._extract_data()
        optimization_summary = {'Name': self.model.description,
                                'Start': str(data['start']),
                                'End': str(data['end']),
                                'Duration': str(data['duration']) + " seconds.",
                                'Epochs': str(data['epochs']),
                                'Batches': str(data['batches'])}
        self._printer.print_dictionary(optimization_summary, "Optimization Summary")          

class OptimizationEarlyStop(Summary):
    """Reports performance information for an optimization."""

    def report(self):        
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
        self._printer.print_dictionary(performance_summary, title)        


class OptimizationHyperparameters(Summary):
    """Reports the hyperparameters used for the optimization."""

    def report(self):
        """Renders report of hyperparameters used for the optimization."""
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

        self._printer.print_dictionary(hyperparameters, "Model HyperParameters")             
  

class OptimizationReport:
    """Prints and optimization report. 

    Parameters
    ----------
    reports : list default=['summary', 'performance', 'critical_points',
                            'features', 'hyperparameters']
        The reports in the order to be rendered. The valid report names are:
            'summary' : prints summary data for optimzation
            'hyperparameters' : prints the hyperparameters used for the optimization
            'performance' : prints performance in terms of cost and scores
            'critical_points' : prints cost and scores at critical points 
                during the optimization
            'features' : prints the best or final intercept and coeficients 
                by feature name if feature names are available. Best results 
                are printed if the Performance observer is used and the 
                'best_or_final' parameter = 'best'. Otherwise, final results
                will be printed.

    """
    def __init__(self, model):
        self.model = model        

    def report(self):
        OptimizationSummary(model=self.model).report()
        OptimizationEarlyStop(model=self.model).report()
        OptimizationHyperparameters(model=self.model).report()

