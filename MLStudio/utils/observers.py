#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : observers.py                                                      #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 21st 2020, 8:04:28 am                         #
# Last Modified : Thursday, May 21st 2020, 8:04:41 am                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes that observe and report performance of models."""
from abc import ABC, abstractmethod, ABCMeta
from collections import OrderedDict 
import copy
import datetime
import numpy as np
import types

from sklearn.base import BaseEstimator

from mlstudio.utils.print import Printer
from mlstudio.utils.validation import validate_zero_to_one
# --------------------------------------------------------------------------- #
#                          OBSERVER BASE CLASS                                #
# --------------------------------------------------------------------------- #
class Observer(ABC, BaseEstimator):
    """Abstract base class for all observer classes."""

    @abstractmethod
    def __init__(self):   
        pass

    @abstractmethod
    def initialize(self, logs=None):
        pass

    @abstractmethod
    def evaluate(self, logs=None):
        pass

    @abstractmethod
    def report(self, estimator, features=None):
        pass

# --------------------------------------------------------------------------- #
#                             STABILITY                                       #
# --------------------------------------------------------------------------- #
class Performance(Observer):
    """Monitors performance and signals when performance has not improved. 
    
    Performance is measured in terms of training or validation cost and scores.
    To ensure that progress is meaningful, it must be greater than a 
    quantity epsilon. If performance has not improved in a predefined number
    of epochs in a row, the evaluation method returns false to the 
    calling object.

    Parameters
    ----------
    metric : str, optional (default='val_score')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'theta': The norm of the parameters of the model
        'gradient': The norm of the gradient of the objective function w.r.t. theta

    epsilon : float, optional (default=0.0001)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, metric='val_score', scorer=MSE(), epsilon=0.01, patience=5):        
        self.name = "Performance Observer"
        self.metric = metric        
        self.scorer = scorer
        self.epsilon = epsilon
        self.patience = patience
       
    def _validate(self):
        if self.metric not in ['train_cost', 'train_score', 'val_cost', 'val_score',
                               'theta', 'gradient']:
            msg = "{m} is an invalid metric. The valid metrics include : {v}".\
                format(m=self.metric,
                       v=str(['train_cost', 'train_score', 'val_cost', 'val_score', 'theta', 'gradient']))
            raise ValueError(msg)
        validate_zero_to_one(p = self.epsilon)       

    def initialize(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """        
        # Attributes
        self.best_performance_ = None
        self.stalled = False
        self.best_weights_ = None        
        # Instance variables
        self._iter_no_improvement = 0
        self._better = None            
        
        logs = logs or {}
        self._validate()
        # Obtain the 'better' function from the scorer.
        # This is either np.less or np.greater        
        if 'score' in self.metric:            
            self._better = self.scorer.better
        else:
            self._better = np.less

    def _print_results(self, current):
        """Prints current, best and relative change."""
        relative_change = abs(current-self.best_performance_) / abs(self.best_performance_)
        print("Iteration #: {i}  Best : {b}     Current : {c}   Relative change : {r}".format(\
                i=str(self._iter_no_improvement),
                b=str(self.best_performance_), 
                c=str(current),
                r=str(relative_change)))            

    def _has_improved(self, current):
        """Returns true if the magnitude of the improvement is greater than epsilon."""
        relative_change = abs(current-self.best_performance_) / abs(self.best_performance_)
        return relative_change > self.epsilon

    def _process_improvement(self, current, logs):
        """Sets values of parameters and attributes if improved."""
        self._iter_no_improvement = 0
        self.best_performance_ = current
        self.best_weights_ = logs.get('theta')
        self._stalled=False        

    def _process_no_improvement(self):
        """Sets values of parameters and attributes if no improved."""        
        self._iter_no_improvement += 1  
        if self._iter_no_improvement == self.patience:
            self._stalled = True           

    def evaluate(self, epoch, logs=None):
        """Determines whether performance is improving.

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
        logs = logs or {}
        # Obtain current cost or score if possible.
        try:
            current = logs.get(self.metric)
        except:
            raise ValueError("{m} is not a valid metric for this optimization."\
                .format(m=self.metric))        

        # If the metric is 'gradient' or 'theta', get the magnitude of the vector
        if self.metric in ['gradient', 'theta']:
            current = np.linalg.norm(current)        

        # Handle first iteration
        if self.best_performance_ is None:
            self._process_improvement(current, logs)

        # Otherwise, if metric is negative i.e. R2, continue as if improved
        elif current < 0:
            self._process_improvement(current, logs)
        # Otherwise...
        else:                
            # Evaluate if there has been an improvement
            if self._better(current, self.best_performance_):
                # Check if improvement is significant
                if self._has_improved(current):
                    self._process_improvement(current, logs)
                else:
                    self._process_no_improvement()                        
            else:
                self._process_no_improvement()                       
        return self._stalled       

    def report(estimator, features=None):
        """Summarizes statistics for model.

        Parameters
        ----------
        estimator : an estimator object
            An fitted estimator object.
        """
        history = estimator.blackbox_
        # ----------------------------------------------------------------------- #
        printer = Printer()
        optimization_summary = {'Name': history.model.description,
                                'Start': str(history.start),
                                'End': str(history.end),
                                'Duration': str(history.duration) + " seconds.",
                                'Epochs': str(history.total_epochs),
                                'Batches': str(history.total_batches)}
        printer.print_dictionary(optimization_summary, "Optimization Summary")

        # ----------------------------------------------------------------------- #
        if history.model.early_stop:    
            performance_summary = \
                {'Final Training Loss': str(np.round(history.epoch_log.get('train_cost')[-1],4)),
                'Final Training Score' : str(np.round(history.epoch_log.get('train_score')[-1],4))
                    + " " + history.model.scorer.name,
                'Final Validation Loss': str(np.round(history.epoch_log.get('val_cost')[-1],4)),
                'Final Validation Score': str(np.round(history.epoch_log.get('val_score')[-1],4))
                        + " " + history.model.scorer.name}
        else:
            performance_summary = \
                {'Final Training Loss': str(np.round(history.epoch_log.get('train_cost')[-1],4)),
                'Final Training Score' : str(np.round(history.epoch_log.get('train_score')[-1],4))
                    + " " + history.model.scorer.name}

        printer.print_dictionary(performance_summary, "Performance Summary")
        
        # --------------------------------------------------------------------------- #
        if features is None:
            features = []
            for i in np.arange(len(history.model.coef_)):
                features.append("Feature_" + str(i))

        theta = OrderedDict()
        theta['Intercept'] = str(np.round(history.model.intercept_, 4))
        for k, v in zip(features, history.model.coef_):
            theta[k]=str(np.round(v,4))
        printer.print_dictionary(theta, "Model Parameters")
        # --------------------------------------------------------------------------- #
        hyperparameters = OrderedDict()
        def get_params(o):
            params = o.get_params()
            for k, v in params.items():
                if isinstance(v, (str, bool, int, float)) or v is None:
                    k = o.__class__.__name__ + '__' + k
                    hyperparameters[k] = str(v)
                else:
                    get_params(v)
        get_params(history.model)
        printer.print_dictionary(hyperparameters, "Model HyperParameters")

