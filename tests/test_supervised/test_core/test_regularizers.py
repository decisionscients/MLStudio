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
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=20)
        lasso = L1(alpha=alpha)
        act_result = lasso(theta)
        exp_result = np.linalg.norm(theta['weights'], ord=1) * alpha        
        assert np.isclose(act_result, exp_result), "Lasso not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))
        # Gradient
        act_result = lasso.gradient(theta)    
        exp_result = np.sign(theta['weights']) * alpha
        assert np.allclose(act_result, exp_result), "Lasso gradient not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))        

    def test_lasso_multiclass(self):
        theta = {}
        alpha = 0.1
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=(20,4))
        lasso = L1(alpha=alpha)
        act_result = lasso(theta)        
        exp_result = np.linalg.norm(theta['weights'], axis=0, ord=1) * alpha        
        assert np.allclose(act_result, exp_result), "Lasso not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))
        # Gradient
        act_result = lasso.gradient(theta)    
        exp_result = np.sign(theta['weights']) * alpha
        assert np.allclose(act_result, exp_result), "Lasso gradient not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))        

# --------------------------------------------------------------------------  #
@mark.regularizers
@mark.ridge
class RidgeTests:

    def test_ridge(self):
        theta = {}
        alpha = 0.1
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=20)
        ridge = L2(alpha=alpha)
        act_result = ridge(theta)
        exp_result = np.linalg.norm(theta['weights'])**2 * alpha        
        assert np.isclose(act_result, exp_result), "Ridge not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))
        # Gradient
        act_result = ridge.gradient(theta)    
        exp_result = theta['weights'] * alpha
        assert np.allclose(act_result, exp_result), "Ridge gradient not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))        

    def test_ridge_multiclass(self):
        theta = {}
        alpha = 0.1
        theta['weights'] = np.random.default_rng().uniform(low=0, high=1, size=(20,4))
        ridge = L2(alpha=alpha)
        act_result = ridge(theta)        
        exp_result = np.sum(np.linalg.norm(theta['weights'])**2) * alpha        
        assert np.allclose(act_result, exp_result), "Ridge not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))
        # Gradient
        act_result = ridge.gradient(theta)    
        exp_result = theta['weights'] * alpha
        assert np.allclose(act_result, exp_result), "Ridge gradient not working\n\
            actual result = {a}\nexpected result = {e}".format(a=str(act_result),
            e=str(exp_result))        

   
      