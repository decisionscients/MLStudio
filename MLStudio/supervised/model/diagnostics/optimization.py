#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : optimization.py                                                   #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, May 25th 2020, 3:59:09 am                           #
# Last Modified : Monday, May 25th 2020, 3:59:09 am                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Functions used to diagnose optimization of a model."""
import numpy as np
import pandas as pd
from tabulate import tabulate

from mlstudio.utils.data_analyzer import describe_numeric_array

def diagnose_gradient(estimator):
    """Produces descriptive statistics for the gradient."""
    gradient_norms = estimator.get_blackbox().epoch_log.get('gradient_norm')    
    df = describe_numeric_array(gradient_norms, fmt='df')            
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
    return df


