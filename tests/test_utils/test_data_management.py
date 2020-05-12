#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_data_management.py                                           #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, May 11th 2020, 8:33:38 pm                           #
# Last Modified : Monday, May 11th 2020, 8:33:38 pm                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests data management utilities."""
#%%
import numpy as np
import pytest
from pytest import mark

from mlstudio.utils.data_manager import MinMaxScaler, StandardScaler
# --------------------------------------------------------------------------  #
#                      TESTS MINMAX SCALER                                    #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
def test_minmax_scaler():
    x = np.array([[0,0,22],
                [0,1,17],
                [0,1,2]], dtype=float)
    x_new = np.array([[0,0,1],
                    [0,1,15/20],
                    [0,1,0]], dtype=float)
    scaler = MinMaxScaler()
    x_t = scaler.fit_transform(x)
    assert np.array_equal(x_new, x_t), "Minmax scaler not working"    
