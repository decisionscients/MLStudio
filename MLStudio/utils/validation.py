#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : validation.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 14th 2020, 1:33:31 pm                         #
# Last Modified : Thursday, May 14th 2020, 1:35:41 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Functions used to validate the state, parameters or data of an estimator."""
import numpy as np

# --------------------------------------------------------------------------  #
def validate_zero_to_one(self, p, left='open', right='open'):
    """Validates a parameter whose values should be [0,1]."""
    assert isinstance(p, (int,float)), "Regularization hyperparameter must be numeric."
    if left == 'open' and right == 'open':
        assert p >= 0 and p <= 1, "Regularization parameter must be between zero and 1."      
    elif left == 'open' and right != 'open':
        assert p >= 0 and p < 1, "Regularization parameter must be between zero and 1."      
    elif left != 'open' and right == 'open':
        assert p > 0 and p <= 1, "Regularization parameter must be between zero and 1."      
    else:
        assert p > 0 and p < 1, "Regularization parameter must be between zero and 1."      