#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : datasets.py                                                       #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, March 15th 2020, 2:02:06 pm                         #
# Last Modified : Tuesday, May 12th 2020, 4:22:12 am                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Datasets used for testing and evaluation."""
import os
from pathlib import Path

import numpy as np
import scipy.sparse as sp

homedir = Path(__file__).parents[0]

def load_urls():
    datadir = os.path.join(homedir,"data/URLs/")
    X_filepath = os.path.join(datadir, "X.npz")
    y_filepath = os.path.join(datadir, "y.npy")
    X = sp.load_npz(X_filepath)
    y = np.load(y_filepath)
    return X, y

