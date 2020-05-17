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
from mlstudio.utils.validation import validate_zero_to_one
# --------------------------------------------------------------------------- #
class EarlyStop(Callback):
    """Abstract base class for all early stop callbacks."""
    @abstractmethod
    def __init__(self, val_size=0.3, epsilon=0.0001):
        self.val_size = val_size    
        self.epsilon = epsilon

    @abstractmethod
    def on_train_begin(self, logs=None):    
        validate_zero_to_one(self.val_size)
        validate_zero_to_one(self.epsilon)

    @abstractmethod
    def on_epoch_end(self, epoch, logs=None):
        pass


# --------------------------------------------------------------------------- #
class Performance(Callback):
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

    val_size : float
        The proportion of the dataset to allocate to validation set.        

    epsilon : float, optional (default=0.01)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, metric='val_cost', val_size=0.3, epsilon=1e-6, patience=50):
        super(Performance, self).__init__()
        self.name = "Performance"
        self.metric = metric
        self.val_size = val_size
        self.epsilon = epsilon
        self.patience = patience
        self.n_iter_ = 0
        self.converged_ = False
        self.best_weights_ = None        
        # Instance variables
        self._iter_no_improvement = 0
        self._better = None    
        # Attributes
        self.best_performance_ = None
        

    def _validate(self):
        if self.metric not in ['train_cost', 'train_score', 'val_cost', 'val_score']:
            raise ValueError("metric must be in ['train_cost', 'train_score', 'val_cost', 'val_score']")
        validate_zero_to_one(self.epsilon)       


    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(Performance, self).on_train_begin(logs)
        logs = logs or {}
        self._validate()
        # We evaluate improvement against the prior metric plus or minus a
        # margin given by epsilon * the metric. Whether we add or subtract the margin
        # is based upon the metric. For metrics that increase as they improve
        # we add the margin, otherwise we subtract the margin.  Each metric
        # has a bit called a epsilon factor that is -1 if we subtract the 
        # margin and 1 if we add it. The following logic extracts the epsilon
        # factor for the metric and multiplies it by the epsilon for the 
        # improvement calculation.
        if 'score' in self.metric:
            scorer = copy.copy(self.model.scorer)
            self._better = scorer.better
            self.best_performance_ = scorer.worst
            self.epsilon *= scorer.epsilon_factor
        else:
            self._better = np.less
            self.best_performance_ = np.Inf
            self.epsilon *= -1 # Bit always -1 since it improves negatively

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
        super(Performance, self).on_epoch_end(epoch, logs)        
        logs = logs or {}
        # Obtain current cost or score
        current = logs.get(self.metric)

        # Handle the first iteration
        if self.best_performance_ in [np.Inf,-np.Inf]:
            self._iter_no_improvement = 0
            self.best_performance_ = current
            self.best_weights_ = logs.get('theta')
            self.converged_ = False
        # Evaluate performance
        elif self._better(current, 
                            (self.best_performance_+self.best_performance_ \
                                *self.epsilon)):            
            self._iter_no_improvement = 0
            self.best_performance_ = current
            self.best_weights_ = logs.get('theta')
            self.converged_=False
        else:
            self._iter_no_improvement += 1
            if self._iter_no_improvement == self.patience:
                self.converged_ = True            
        self.model.converged = self.converged_                     

# --------------------------------------------------------------------------- #
class Stability(Callback):
    """Stops when performance becomes stable.

    Stability is measured in the relative change in a metric i.e. the norm
    of the gradient or the validation score. 

    Parameters
    ----------
    metric : str (default = 'gradient')
        The metric to metric. Valid values include: 'gradient', 'theta',
        'val_cost', 'val_score'
    val_size : float (default=0.3)
        The proportion of training set to allocate to the validation set.

    epsilon : float Default 0.001
        The lower bound allowed for the percent change in the gradient norm.

    """

    def __init__(self, metric='val_cost', val_size = 0.3, epsilon=0.0001):        
        super(Stability, self).__init__()
        self.name = "Stability" 
        self.metric = metric 
        self.val_size = val_size
        self.epsilon = epsilon
        self._previous = None

    def on_train_begin(self, logs=None):
        super(Stability, self).on_train_begin(logs)
        if self.metric not in ['gradient', 'theta', 'train_cost', 'train_score',\
                               'val_cost', 'val_score']:
            msg = "Metric {m} is not supported. Valid values include: 'gradient',\
                 'theta', 'train_cost', 'train_score', 'val_cost', 'val_score.".format(m=self.metric)
            raise ValueError(msg)

    def on_epoch_end(self, epoch, logs=None):        
        """Stops when relative change in the metric is below epsilon"""
        super(Stability, self).on_epoch_end(epoch, logs)
        if self.metric in ['gradient', 'theta']:
            current = np.linalg.norm(logs.get(self.metric)) 
        else:
            current = logs.get(self.metric)

        if self._previous is None:
            self._previous = current
        elif (current - self._previous) \
            / self._previous * 100 < self.epsilon:
            self.model.converged = True
        else:
            self._previous = current

# --------------------------------------------------------------------------- #
class BCN1(Callback):
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

    def __init__(self, n_size = 100, val_size = 0.3, epsilon=0.0001):        
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