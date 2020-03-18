# =========================================================================== #
#                               REGULARIZERS                                  #
# =========================================================================== #
# =========================================================================== #
# Project: ML Studio                                                          #
# Version: 0.1.14                                                             #
# File: \regularizers.py                                                      #
# Python Version: 3.8.0                                                       #
# ---------------                                                             #
# Author: John James                                                          #
# Company: Decision Scients                                                   #
# Email: jjames@decisionscients.com                                           #
# ---------------                                                             #
# Create Date: Monday August 19th 2019, 7:36:03 pm                            #
# Last Modified: Saturday November 30th 2019, 10:37:34 am                     #
# Modified By: John James (jjames@decisionscients.com)                        #
# ---------------                                                             #
# License: Modified BSD                                                       #
# Copyright (c) 2019 Decision Scients                                         #
# =========================================================================== #

"""Classes for L1, L2, and Elasticnet Regularization"""
from abc import ABC, abstractmethod
import numpy as np

class Regularizer(ABC):
    """Abstract base class for regularizers.
    
    Also functions as default regularizer when no regularization is used.
    """

    def __init__(self):
        self.callback = "Regularizer"

    def _validate_hyperparam(self, p):
        assert isinstance(p, float), "Regularization hyperparameter must be numeric."
        assert p >= 0 and p <= 1, "Regularization parameter must be between zero and 1."
    
    def __call__(self, w):
        return 0.0
    
    def gradient(self, w):
        return 0.0

 


class L1(Regularizer):
    """ Regularization for Lasso Regression """
    def __init__(self, alpha):
        super(L1,self).__init__()
        self._alpha = alpha
        self.name = "Lasso Regression (L1)"
    
    def __call__(self, w):
        self._validate_hyperparam(self._alpha)
        return self._alpha * np.linalg.norm(w, ord=1)

    def gradient(self, w):
        self._validate_hyperparam(self._alpha)
        return self._alpha * np.sign(w)


class L2(Regularizer):
    """ Regularization for Ridge Regression """
    def __init__(self, alpha):
        super(L2,self).__init__()
        self._alpha = alpha
        self.name = "Ridge Regression (L2)"
    
    def __call__(self, w): 
        self._validate_hyperparam(self._alpha)
        return self._alpha * np.linalg.norm(w)**2

    def gradient(self, w):
        self._validate_hyperparam(self._alpha)
        return self._alpha * 2.0 * w


class ElasticNet(Regularizer):
    """ Regularization for Elastic Net Regression """
    def __init__(self, alpha=1.0, ratio=0.5):
        super(ElasticNet,self).__init__()
        self._alpha = alpha
        self._ratio = ratio
        self.name = "Elastic Net Regression"

    def __call__(self, w):
        self._validate_hyperparam(self._alpha)
        self._validate_hyperparam(self._ratio)
        l1_contr = self._ratio * np.linalg.norm(w, ord=1)
        l2_contr = (1 - self._ratio) * 0.5 * np.linalg.norm(w)**2
        return self._alpha * (l1_contr + l2_contr)

    def gradient(self, w):
        self._validate_hyperparam(self._alpha)
        l1_contr = self._ratio * np.sign(w)
        l2_contr = (1 - self._ratio) * w
        alpha = np.asarray(self._alpha, dtype='float64')
        return np.multiply(alpha, np.add(l1_contr, l2_contr))



