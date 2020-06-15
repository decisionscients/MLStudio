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
from tabulate import tabulate

from mlstudio.utils.format import proper
from mlstudio.supervised.observers.performance import Performance
from mlstudio.utils.print import Printer

TODO: Create summary class
class OptimizationSummary:
        def _report_hyperparameters(self):
        hyperparameters = OrderedDict()
        def get_params(o):
            params = o.get_params()
            for k, v in params.items():
                if isinstance(v, (str, bool, int, float, np.ndarray, np.generic, list)) or v is None:
                    k = o.__class__.__name__ + '__' + k
                    hyperparameters[k] = str(v)
                else:
                    get_params(v)
        get_params(self.model)

        self._printer.print_dictionary(hyperparameters, "Model HyperParameters")        


    def _report_features(self, features=None):
        theta = OrderedDict()
        theta['Intercept'] = str(np.round(self.model.intercept_, 4))      

        if features is None:
            # Try to get the features from the object
            features = self.model.features_

        # If no features were provided to the estimator, create dummy features.
        if features is None:
            features = []
            for i in np.arange(len(self.model.coef_)):
                features.append("Feature_" + str(i))

        for k, v in zip(features, self.model.coef_):
            theta[k]=str(np.round(v,4))  
        self._printer.print_dictionary(theta, "Model Parameters")        

    def _report_critical_points(self):
        if self.model.critical_points:    
            cp = []
            for p in self.model.critical_points:
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


    def _print_performance(self, result, best_or_final='final'):                
        datasets = {'train': 'Training', 'val': 'Validation'}
        keys = ['train', 'val']
        metrics = ['cost', 'score']
        print_data = []
        # Format keys, labels and data for printing based upon the results
        for performance in list(itertools.product(keys, metrics)):
            d = {}
            key = performance[0] + '_' + performance[1]
            if result.get(key):
                label = proper(best_or_final) + ' ' + datasets[performance[0]] \
                    + ' ' + proper(performance[1]) 
                d['label'] = label
                if performance[1] == 'score' and hasattr(self.model, 'scorer'):                    
                    d['data'] = str(np.round(result[key],4)) + " " + self.model.scorer.name
                else:
                    d['data'] = str(np.round(result[key],4)) 
                print_data.append(d)
        
        performance_summary = OrderedDict()
        for i in range(len(print_data)):
            performance_summary[print_data[i]['label']] = print_data[i]['data']
        title = proper(best_or_final) + " Weights Performance Summary"
        self._printer.print_dictionary(performance_summary, title)        

    def _report_performance(self):
        result = self.model.best_result
        self._print_performance(result, 'best')        
        result = self.model.final_result
        self._print_performance(result, 'final')

    def _report_summary(self):
        """Reports summary information for the optimization."""        
        optimization_summary = {'Name': self.model.description,
                                'Start': str(self.start),
                                'End': str(self.end),
                                'Duration': str(self.duration) + " seconds.",
                                'Epochs': str(self.total_epochs),
                                'Batches': str(self.total_batches)}
        self._printer.print_dictionary(optimization_summary, "Optimization Summary")        

    def report(self, features=None):
        """Summarizes performance statistics and parameters for model."""
        self._printer = Printer()
        self._report_summary()        
        self._report_performance()        
        self._report_critical_points()
        self._report_features(features)
        self._report_hyperparameters()   