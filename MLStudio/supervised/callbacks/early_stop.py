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
from mlstudio.utils.validation import validate_zero_to_one
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
    
    Stops training if performance hasn't improved. Improvement is measured 
    with a 'tolerance', so that performance must improve by a factor greater
    than the tolerance, to be considered improved. A 'patience' parameter
    indicates how long non-performance has to occur, in epochs, to stop
    training.

    Parameters
    ----------
    metric : str, optional (default='val_score')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'theta': The parameters of the model
        'gradient': The gradient of the objective function w.r.t. theta

    val_size : float
        The proportion of the dataset to allocate to validation set.        

    epsilon : float, optional (default=0.001)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    

    mode : str (default="active")
        Indicates whether to stop training when performance stops improving
        or to just indicate that training has stalled.
    """

    def __init__(self, metric='val_cost', val_size=0.3, epsilon=1e-1, patience=100, mode="active"):
        super(Stability, self).__init__()
        self.name = "Stability"
        self.metric = metric
        self.val_size = val_size
        self.epsilon = epsilon
        self.patience = patience
        self.mode = mode
       

    def _validate(self):
        if self.metric not in ['train_cost', 'train_score', 'val_cost', 'val_score',
                               'theta', 'gradient']:
            msg = "{m} is an invalid metric. The valid metrics include : {v}".\
                format(m=self.metric,
                       v=str(['train_cost', 'train_score', 'val_cost', 'val_score', 'theta', 'gradient']))
            raise ValueError(msg)
        validate_zero_to_one(p = self.epsilon)       

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
        
        if self._observer.evaluate(epoch, logs):
            self.model.stalled = True
            if self.mode == "active":
                self.converged = True
        else:
            self.model.stalled = False
            self.converged = False

# --------------------------------------------------------------------------- #
class BCN1(EarlyStop):
    """Stopping Criteria #1: Based upon Bottou-Curtis-Nocedal Functions.

    Let :math:'\epsilon' > 0. Let :math: '{T_j}' be a sequence of positive-
    valued, strictly increasing, finite stopping times with respect to
    :math: '{F_k}'. Then, the SGD iterates are stopped at iterate 
    :math: 'T_j' where:

        :math: 'J = \text{min}\bigg\{j \ge 1: \lVert F(\Beta T_j)\rVert_2\le\epsilon\bigg}.'

    Parameters
    ----------
    val_size : float (default=0.3)
        The proportion of training set to allocate to the validation set.

    epsilon : float Default 0.0001
        The lower bound allowed for the percent change in the gradient norm.

    Reference: https://arxiv.org/abs/2004.00475

    """

    def __init__(self, n_size = 100, val_size = 0.3, epsilon=0.01):        
        super(BCN1, self).__init__()
        self.name = "BCN1"                 
        self.val_size = val_size
        self.epsilon = epsilon     

    def on_train_begin(self, logs=None):
        super(BCN1, self).on_train_begin(logs)
        j 

    def on_epoch_end(self, epoch, logs=None):        
        """Stops the epoch the L2 norm of the gradient drops below epsilon"""
        super(BCN1, self).on_epoch_end(epoch, logs)
        batch = logs.get('batch')
        norm_g = np.linalg.norm(logs.get('gradient')) 
        if norm_g <= self.epsilon and batch >= 1:
            J = np.min(batch, norm_g)
            if J == batch:
                self.model.converged = True
   

                self.model.converged = True