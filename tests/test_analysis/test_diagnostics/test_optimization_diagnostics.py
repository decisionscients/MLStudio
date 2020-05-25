#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_optimization_diagnostics.py                                  #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, May 25th 2020, 5:53:51 am                           #
# Last Modified : Monday, May 25th 2020, 5:53:51 am                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests optimization diagnostic functions."""
import numpy as np
import pytest
from pytest import mark

from mlstudio.analysis.diagnostics.optimization import diagnose_gradient
# --------------------------------------------------------------------------  #
@mark.analysis
@mark.diagnostics
@mark.optimization
def test_diagnose_gradient(get_estimator):
    est = get_estimator
    diagnose_gradient(est)