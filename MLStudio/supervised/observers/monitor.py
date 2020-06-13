# %%
# =========================================================================== #
#                                MONITOR                                      #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \monitor.py                                                           #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday November 5th 2019, 8:47:45 pm                          #
# Last Modified: Saturday November 30th 2019, 10:37:20 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Module containing observers used to monitor and report training performance."""
from collections import OrderedDict 
import datetime
import itertools
import numpy as np
import pandas as pd
from tabulate import tabulate
import types

from mlstudio.supervised.observers.base import Observer
from mlstudio.utils.format import proper
from mlstudio.supervised.observers.performance import Performance
from mlstudio.utils.print import Printer
from mlstudio.utils.validation import validate_int, validate_zero_to_one
from mlstudio.utils.validation import validate_metric, validate_scorer


# --------------------------------------------------------------------------- #
#                             BLACKBOX CLASS                                  #
# --------------------------------------------------------------------------- #
class BlackBox(Observer):
    """Records history and metrics for training by epoch."""

    def on_train_begin(self, logs=None):
        """Sets instance variables at the beginning of training.
        
        Parameters
        ----------
        logs : Dict
            Dictionary containing the X and y data
        """ 
        self.total_epochs = 0
        self.total_batches = 0
        self.start = datetime.datetime.now()
        self.epoch_log = {}
        self.batch_log = {}

    def on_train_end(self, logs=None):        
        """Sets instance variables at end of training.
        
        Parameters
        ----------
        logs : Dict
            Not used 
        """
        self.end = datetime.datetime.now()
        self.duration = (self.end-self.start).total_seconds() 
        if self.model.verbose:
            self.report()

    def on_batch_end(self, batch, logs=None):
        """Updates data and statistics relevant to the training batch.
        
        Parameters
        ----------
        batch : int
            The current training batch
        
        logs : dict
            Dictionary containing batch statistics, such as batch size, current
            weights and training cost.
        """
        self.total_batches = batch
        for k,v in logs.items():
            self.batch_log.setdefault(k,[]).append(v)        

    def on_epoch_begin(self, epoch, logs=None):
        """Updates data and statistics relevant to the training epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch
        
        logs : dict
            Dictionary containing data and statistics for the current epoch,
            such as weights, costs, and optional validation set statistics
            beginning with 'val_'.
        """
        logs = logs or {}
        self.total_epochs += 1
        for k,v in logs.items():
            self.epoch_log.setdefault(k,[]).append(v)

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
          

# --------------------------------------------------------------------------- #
#                            PROGRESS CLASS                                   #
# --------------------------------------------------------------------------- #              
class Progress(Observer):
    """Class that reports progress at designated points during training."""
    
    def on_epoch_end(self, epoch, logs=None):
        """Reports progress at the end of each epoch.

        Parameters
        ----------
        epoch : int
            The current training epoch

        logs : Dict
            Statistics obtained at end of epoch
        """
        if self.model.verbose and (epoch % self.model.checkpoint == 0):
            items_to_report = ('epoch', 'train', 'val')
            logs = {k:v for k,v in logs.items() if k.startswith(items_to_report)}
            progress = "".join(str(key) + ': ' + str(np.round(value,4)) + ' ' \
                for key, value in logs.items())
            print(progress)
        
