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

import numpy as np

from mlstudio.supervised.estimator.callbacks import Callback

# --------------------------------------------------------------------------- #
#                              EARLY STOP                                     #
# --------------------------------------------------------------------------- #
class EarlyStop(Callback):
    """Stops training if performance hasn't improved.
    
    Stops training if performance hasn't improved. Improvement is measured 
    with a 'tolerance', so that performance must improve by a factor greater
    than the tolerance, to be considered improved. A 'patience' parameter
    indicates how long non-performance has to occur, in epochs, to stop
    training.

    Parameters
    ----------
    monitor : str, optional (default='val_score')
        Specifies which statistic to monitor for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter

    precision : float, optional (default=0.01)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    
    """

    def __init__(self, early_stop=True, monitor='val_score', precision=0.01, patience=10):
        super(EarlyStop, self).__init__()
        self.early_stop = early_stop
        self.monitor = monitor
        self.precision = precision
        self.patience = patience
        self._epochs_trained = 0
        self.converged = False
        self.best_weights_ = None        
        # Instance variables
        self._iter_no_improvement = 0
        self._better = None    
        # Attributes
        self.best_performance_ = None
        

    def _validate(self):
        if self.monitor not in ['train_cost', 'train_score', 'val_cost', 'val_score']:
            raise ValueError("monitor must be in ['train_cost', 'train_score', 'val_cost', 'val_score']")
        elif not isinstance(self.precision, float):
            raise TypeError("precision must be a float.")
        elif self.precision < 0 or self.precision > 1:
            raise ValueError("precision must be between 0 and 1. A good default is 0.01 or 1%.")
        elif not isinstance(self.patience, (int)):
            raise TypeError("patience must be an integer.")
        elif 'score' in self.monitor and self.model.metric is None:
            raise ValueError("'score' has been selected for evaluation; however"
                             " no scoring metric has been provided for the model. "
                             "Either change the metric in the EarlyStop class to "
                             "'train_cost' or 'val_cost', or add a metric to the model.")


    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(EarlyStop, self).on_train_begin()
        if self.early_stop:
            logs = logs or {}
            self._validate()
            # We evaluate improvement against the prior metric plus or minus a
            # margin given by precision * the metric. Whether we add or subtract the margin
            # is based upon the metric. For metrics that increase as they improve
            # we add the margin, otherwise we subtract the margin.  Each metric
            # has a bit called a precision factor that is -1 if we subtract the 
            # margin and 1 if we add it. The following logic extracts the precision
            # factor for the metric and multiplies it by the precision for the 
            # improvement calculation.
            if 'score' in self.monitor:
                scorer = self.model.scorer
                self._better = scorer.better
                self.best_performance_ = scorer.worst
                self.precision *= scorer.precision_factor
            else:
                self._better = np.less
                self.best_performance_ = np.Inf
                self.precision *= -1 # Bit always -1 since it improves negatively

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
        if self.early_stop:
            logs = logs or {}
            # Obtain current cost or score
            current = logs.get(self.monitor)

            # Handle the first iteration
            if self.best_performance_ in [np.Inf,-np.Inf]:
                self._iter_no_improvement = 0
                self.best_performance_ = current
                self.best_weights_ = logs.get('theta')
                self.converged = False
            # Evaluate performance
            elif self._better(current, 
                                (self.best_performance_+self.best_performance_ \
                                    *self.precision)):            
                self._iter_no_improvement = 0
                self.best_performance_ = current
                self.best_weights_ = logs.get('theta')
                self.converged=False
            else:
                self._iter_no_improvement += 1
                if self._iter_no_improvement == self.patience:
                    self.converged = True            
            self.model.converged = self.converged                     
        else:
            self.model.converged = False






# %%
