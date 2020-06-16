#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_objectives.py                                                #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 3:45:31 pm                          #
# Last Modified : Monday, June 15th 2020, 3:45:31 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
import math
import numpy as np
import pytest
from pytest import mark

from mlstudio.supervised.core.regularizers import L1, L2, L1_L2
# --------------------------------------------------------------------------  #
@mark.regularizers
@mark.lasso
class LassoTests:

    def test_lasso(self):
        theta = {}
        alpha = 0.1
        theta = np.array([40, 6, -22, 31,26])      
        lasso = L1(alpha=alpha)
        # Lasso loss regularization        
        exp_result = 8.5
        act_result = lasso(theta)
        assert np.isclose(exp_result,act_result), "Lasso regularization error"
        # Lasso loss regularization gradient
        exp_result = np.array([0,0.1,-0.1,0.1,0.1])
        act_result = lasso.gradient(theta)
        assert np.allclose(exp_result, act_result), "Lasso gradient error"

# --------------------------------------------------------------------------  #
@mark.regularizers
@mark.ridge
class RidgeTests:

    def test_ridge(self):
        theta = {}
        alpha = 0.1
        theta = np.array([40, 6, -22, 31,26])      
        ridge = L2(alpha=alpha)
        # Ridge loss regularization        
        exp_result = 215.7
        act_result = ridge(theta)
        assert np.isclose(exp_result,act_result), "Ridge regularization error"
        # Ridge loss regularization gradient
        exp_result = np.array([0,0.6,-2.2,3.1,2.6])
        act_result = ridge.gradient(theta)
        assert np.allclose(exp_result, act_result), "Ridge gradient error"
    
# --------------------------------------------------------------------------  #
@mark.regularizers
@mark.elasticnet
class ElasticnetTests:

    def test_elasticnet(self):
        theta = {}
        alpha = 0.1
        ratio = 0.5
        theta = np.array([40, 6, -22, 31,26])      
        elasticnet = L1_L2(alpha=alpha, ratio=ratio)
        # Elasticnet loss regularization        
        exp_result = 58.175
        act_result = elasticnet(theta)
        assert np.isclose(exp_result,act_result), "Elasticnet regularization error"
        # Elasticnet loss regularization gradient
        exp_result = np.array([0,0.35,-1.15, 1.6,1.35])
        act_result = elasticnet.gradient(theta)
        assert np.allclose(exp_result, act_result), "Elasticnet gradient error"


   
      