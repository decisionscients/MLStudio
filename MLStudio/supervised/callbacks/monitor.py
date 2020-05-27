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

"""Module containing callbacks used to monitor and report training performance."""
import datetime
import numpy as np
import types
from collections import OrderedDict 

from mlstudio.supervised.callbacks.base import Callback
from mlstudio.utils.observers import Performance
from mlstudio.utils.print import Printer
from mlstudio.utils.validation import validate_int, validate_zero_to_one
from mlstudio.utils.validation import validate_metric

# --------------------------------------------------------------------------- #
#                             MONITOR                                         #
# --------------------------------------------------------------------------- #
class Monitor(Callback):
    """Monitors progress and signals the model when performance has stabilized. 
    
    This class delegates performance evaluation to an observer object. If
    performance has improved according to the patients and epsilon parameters,
    the observer returns False. If performance has not improved, the observer
    returns the epoch at the point of stabilization. 

    Parameters
    ----------
    metric : str, optional (default='train_score')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'gradient_norm': The norm of the gradient of the objective function w.r.t. theta

    epsilon : float, optional (default=0.001)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, metric='train_cost', epsilon=1e-2, patience=50):
        super(Monitor, self).__init__()
        self.name = "Monitor"
        self.metric = metric
        self.epsilon = epsilon
        self.patience = patience

    @property
    def best_results(self):
        try:
            results = self._observer.best_results
        except:
            msg = "Results aren't available until after training."
            raise Exception(msg)
        return results

    def _validate(self):        
        validate_metric(self.metric)
        if 'score' in self.metric:
            validate_scorer(self.model.scorer)
        validate_zero_to_one(param=self.epsilon, param_name='epsilon')       
        validate_int(param=self.patience, param_name='patience')

    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(Monitor, self).on_train_begin(logs)
        self._validate()        
        # Initialize state variables
        self._stabilized = False
        self._last_state = False
        self._critical_points = []
        # Obtain scorer from model if it has one
        scorer = None
        if hasattr(self.model, 'scorer'):
            scorer = self.model.scorer
        # Create and initialize the observer object.
        self._observer = Performance(metric=self.metric, scorer=scorer, \
            epsilon=self.epsilon, patience=self.patience)    
        self._observer.initialize()        

    def on_epoch_end(self, epoch, logs=None):
        """Determines whether convergence has been achieved.

        Parameters
        ----------
        epoch : int
            The current epoch number

        logs : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  

        Returns
        -------
        Bool if True convergence has been achieved. 

        """
        super(Monitor, self).on_epoch_end(epoch, logs)        
        logs = logs or {}                
        if self._observer.model_is_stable(epoch, logs):
            self._stabilized = True
            if self._stabilized != self._last_state:
                self._last_state = self._stabilized
                self._critical_points.append(logs)

# --------------------------------------------------------------------------- #
#                             HISTORY CLASS                                   #
# --------------------------------------------------------------------------- #
class BlackBox(Callback):
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
        # If a log has been passed, update the epoch log. This is used
        # to add epoch 0 evaluation data to the log before training
        if logs is not None:
            for k,v in logs.items():
                self.epoch_log.setdefault(k,[]).append(v)            


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

    def on_epoch_end(self, epoch, logs=None):
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
        self.total_epochs = epoch
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


    def _report_features(self):
        theta = OrderedDict()
        theta['Intercept'] = str(np.round(self.model.intercept_, 4))      

        if self.model.feature_names:
            features = self.model.feature_names
        else:
            features = []
            for i in np.arange(len(self.model.coef_)):
                features.append("Feature_" + str(i))

        for k, v in zip(features, self.model.coef_):
            theta[k]=str(np.round(v,4))  
        self._printer.print_dictionary(theta, "Model Parameters")        

    def _report_critical_points(self):
        if self.model.critical_points:
            print("\n")
            self._printer.print_title("Critical Points")
            print(tabulate(self.model.critical_points, headers="keys"))
            print("\n")        

    def _report_performance_cost(self):
        performance_summary = \
            {'Final Training Loss': str(np.round(self.epoch_log.get('train_cost')[-1],4))}

        self._printer.print_dictionary(performance_summary, "Performance Summary")                


    def _report_performance_with_validation(self):
        performance_summary = \
            {'Final Training Loss': str(np.round(self.epoch_log.get('train_cost')[-1],4)),
            'Final Training Score' : str(np.round(self.epoch_log.get('train_score')[-1],4))
                + " " + self.model.scorer.name,
            'Final Validation Loss': str(np.round(self.epoch_log.get('val_cost')[-1],4)),
            'Final Validation Score': str(np.round(self.epoch_log.get('val_score')[-1],4))
                    + " " + self.model.scorer.name}

        self._printer.print_dictionary(performance_summary, "Performance Summary")                    

    def _report_performance_wo_validation(self):
        performance_summary = \
            {'Final Training Loss': str(np.round(self.epoch_log.get('train_cost')[-1],4)),
             'Final Training Score' : str(np.round(self.epoch_log.get('train_score')[-1],4))
             + " " + self.model.scorer.name}

        self._printer.print_dictionary(performance_summary, "Performance Summary")        

    def _report_performance(self):
        if hasattr(self.model, 'X_val_'):    
            if self.model.X_val_ is not None:
                if self.model.X_val_.shape[0] > 0:
                    self._report_performance_with_validation()
                else:
                    self._report_performance_wo_validation()
            elif hasattr(self.model, 'scorer'):
                self._report_performance_wo_validation()        
            else:
                self._report_performance_cost()
        elif hasattr(self.model, 'scorer'):
            self._report_performance_wo_validation()        
        else:
            self._report_performance_cost()

    def _report_summary(self):
        """Reports summary information for the optimization."""        
        optimization_summary = {'Name': self.model.description,
                                'Start': str(self.start),
                                'End': str(self.end),
                                'Duration': str(self.duration) + " seconds.",
                                'Epochs': str(self.total_epochs),
                                'Batches': str(self.total_batches)}
        self._printer.print_dictionary(optimization_summary, "Optimization Summary")        

    def report(self):
        """Summarizes performance statistics and parameters for model."""
        self._printer = Printer()
        self._report_summary()        
        self._report_performance()        
        self._report_critical_points()
        self._report_features()
        self._report_hyperparameters()        
          

# --------------------------------------------------------------------------- #
#                            PROGRESS CLASS                                   #
# --------------------------------------------------------------------------- #              
class Progress(Callback):
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
        
