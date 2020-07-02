#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : early_stop.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, May 15th 2020, 9:16:41 pm                           #
# Last Modified : Sunday, June 14th 2020, 11:49:38 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Early stop classes."""
from abc import ABC, abstractmethod
import datetime
import numpy as np
import pandas as pd

from mlstudio.supervised.observers.base import Observer, PerformanceObserver
from mlstudio.utils.validation import validate_metric, validate_int
from mlstudio.utils.validation import validate_zero_to_one
# --------------------------------------------------------------------------- #
#                               PERFORMANCE                                   #
# --------------------------------------------------------------------------- #
class Performance(PerformanceObserver):
    """Performances and log model performance, critical points and stability. 

    Performance is defined in terms of:

        * Metric : A metric to observe. This can be training error, validation
            score, gradient norm or the like 

        * Epsilon : A mininum amount of relative change in the observed
            metric required to consider the optimization in a productive
            state.

        * Patience : The number of consecutive epochs or iterations of
            non-improvement that is tolerated before considering an
            optimization stabilized.    

    Parameters
    ----------
    metric : str, optional (default='train_cost')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_score': Validation set scores based upon the model's metric parameter
        'gradient_norm': The norm of the gradient of the objective function w.r.t. theta

    epsilon : float, optional (default=0.01)
        The amount of relative change in the observed metric considered to be
        a sufficient improvement in performance. 

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that is 
        tolerated before considering the optimization stable.

    """

    def __init__(self, metric='val_score', epsilon=0.001, patience=5): 
        super(Performance, self).__init__(                   
            metric = metric,
            epsilon = epsilon,
            patience = patience
        )
        self.name = "Performance EarlyStop Observer"
        

    def on_train_begin(self, log=None):
        """Logic executed at the beginning of training.
        
        Parameters
        ----------
        log: dict
            Currently not used
        """
        super(Performance, self).on_train_begin(log=log)

    def on_epoch_end(self, epoch, log=None):
        """Logic executed at the end of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        log: dict
            Dictionary containing the data, cost, batch size and current weights
        """                  
        super(Performance, self).on_epoch_end(epoch=epoch, log=log)
        if self._stabilized:
            self.model.converged = True       
