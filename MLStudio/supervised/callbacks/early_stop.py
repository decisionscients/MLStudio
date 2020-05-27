# %%
# =========================================================================== #
#                             EARLY STOP CLASSES                              #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \early_stop.py                                                        #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Tuesday September 24th 2019, 3:16:03 am                        #
# Last Modified: Saturday November 30th 2019, 10:36:20 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

from abc import ABC, abstractmethod, ABCMeta
import copy
import numpy as np

from mlstudio.supervised.callbacks.base import Callback
from mlstudio.supervised.core.scorers import MSE
from mlstudio.utils.observers import Performance
from mlstudio.utils.validation import validate_metric, validate_zero_to_one
# --------------------------------------------------------------------------- #
#                          EARLY STOP BASE CLASS                              #
# --------------------------------------------------------------------------- #
class EarlyStop(Callback, ABC):
    """Abstract base class for all early stop classes."""

    @abstractmethod
    def __init__(self):
        super(EarlyStop, self).__init__()

    @abstractmethod
    def on_train_begin(self, logs=None):
        pass

    @abstractmethod
    def on_epoch_end(self, epoch, logs=None):
        pass

# --------------------------------------------------------------------------- #
#                             STABILITY                                       #
# --------------------------------------------------------------------------- #
class Stability(EarlyStop):
    """Stops training if performance hasn't improved.
    
    Observes performance and signals if performance has not improved. 
    Improvement is measured  with a tolerance parameter 'epsilon', so that 
    performance must improve by a relative factor greater than the 'epsilon', 
    to be considered significant enough. 'patience' parameter indicates how long 
    the optimizer has to show improvement before the algorithm signals that 
    performance has stabilized. 

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
        super(Stability, self).__init__()
        self.name = "Stability"
        self.metric = metric
        self.epsilon = epsilon
        self.patience = patience

    def _validate(self):        
        validate_metric(self.metric)
        validate_zero_to_one(param=self.epsilon, param_name='epsilon')       

    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(Stability, self).on_train_begin(logs)
        self._validate()        
        self._observer = Performance(metric=self.metric, scorer=self.model.scorer, \
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
        super(Stability, self).on_epoch_end(epoch, logs)        
        logs = logs or {}        
        
        if self._observer.model_is_stable(epoch, logs):
            self.model.converged = True
        else:            
            self.model.converged = False

