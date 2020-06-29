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
from mlstudio.supervised.observers.early_stop import Performance
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

class OptimizationPerformance(Summary):
    """Reports performance information for an optimization."""

    def _extract_data(self):
        """Extracts required data from the model."""
        final_results = self.model.final_results_
        try:
            best_results = self.model.best_results_ 
        except:
            best_results = {}
        return final_results, best_results 

    def _report(self, result, best_or_final):
        datasets = {'train': 'Training', 'val': 'Validation'}
        keys = ['train', 'val']
        metrics = ['cost', 'score']
        print_data = []
        # If printing 'best_results', extract the epoch
        if best_or_final == 'best':
            d = {'label': 'Best Epoch', 'data' : str(result.get('epoch'))}
            print_data.append(d)
        # Format labels and data for printing from result parameter
        for performance in list(itertools.product(keys, metrics)):
            d = {}
            key = performance[0] + '_' + performance[1]
            if result.get(key):
                label = proper(best_or_final) + ' ' + datasets[performance[0]] \
                    + ' ' + proper(performance[1]) 
                d['label'] = label
                if performance[1] == 'score' and hasattr(self.model, 'scorer'):                    
                    d['data'] = str(np.round(result.get(key),4)) + " " + self.model.scorer.name
                else:
                    d['data'] = str(np.round(result.get(key),4)) 
                print_data.append(d) 

        # Print performance statistics
        performance_summary = OrderedDict()
        for i in range(len(print_data)):
            performance_summary[print_data[i]['label']] = print_data[i]['data']
        title = proper(best_or_final) + " Weights Performance Summary"
        self._printer.print_dictionary(performance_summary, title)        

    def report(self):
        final_results, best_results  = self._extract_data()
        self._report(final_results, 'final')
        if best_results:
            self._report(best_results, 'best')

class OptimizationCriticalPoints(Summary):
    """Prints statistics at critical points during the optimization.
    
    Critical points are when the optimization changed state from stabilized
    to not stabilized.
    """
    def _extract_data(self):
        critical_points = None
        if hasattr(self.model, 'critical_points_'):
            critical_points = self.model.critical_points_
        return critical_points

    def report(self):
        critical_points = self._extract_data()
        if critical_points:        
            cp = []
            for p in critical_points:
                d = {}
                for k,v in p.items():
                    d[proper(k)] = v
                cp.append(d)                      
            self._printer.print_title("Critical Points")
            df = pd.DataFrame(cp) 
            df = df.drop(['Theta', 'Gradient'], axis=1)
            df.set_index('Epoch', inplace=True)
            print(tabulate(df, headers="keys"))
            print("\n")       


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


class OptimizationFeatures(Summary):
    """Reports the parameters with feature names if the feature names are available."""

    def __init__(self, model):
        super(OptimizationFeatures, self).__init__(
            model = model
        )            

    def report(self):
        theta = OrderedDict()
        theta['Intercept'] = str(np.round(self.model.intercept_, 4))      

        # If no features were provided to the estimator, create dummy features.
        if self.model.feature_names is None:
            self.features = []
            for i in np.arange(len(self.model.coef_)):
                self.features.append("Feature_" + str(i))

        for k, v in zip(self.features, self.model.coef_):
            theta[k]=str(np.round(v,4))  
        self._printer.print_dictionary(theta, "Model Parameters")        

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
    def __init__(self, model, reports=None):
        self.model = model
        self.reports = reports

    def report(self):

        valid_reports = ['summary', 'performance', 'critical_points', 
                           'features', 'hyperparameters']

        default_reports = ['summary', 'performance', 
                           'features', 'hyperparameters']                           

        reports = {'summary': OptimizationSummary(model=self.model), 
                   'performance': OptimizationPerformance(model=self.model),
                   'critical_points': OptimizationCriticalPoints(model=self.model), 
                   'features' : OptimizationFeatures(model=self.model), 
                   'hyperparameters' : OptimizationHyperparameters(model=self.model)}

        if self.reports is None:
            self.reports = default_reports
        if set(self.reports).issubset(set(default_reports)):            
            for report in self.reports:
                reports[report].report()
        else:
            msg = "One of your reports is not a valid report. Valid reports\
                 include {r}".format(str(default_reports))
            raise ValueError(msg)


def get_performance_observer(model):
    """Obtains performance observer if it was used."""
    observers = model.observers
    if observers:
        for name, observer in observers.items():
            if isinstance(observer, Performance):
                return observer
    return None        

