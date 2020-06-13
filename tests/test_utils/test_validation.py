#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_py                                                #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Sunday, May 24th 2020, 5:15:40 am                           #
# Last Modified : Sunday, May 24th 2020, 5:15:40 am                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests validation utilities."""
import sys
import numpy as np
import pytest
from pytest import mark
import scipy.sparse as sp


# --------------------------------------------------------------------------  #
#                          TEST VALIDATION                                    #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.validation
def test_validate_zero_to_one():
    from mlstudio.utils.validation import validate_zero_to_one
    with pytest.raises(AssertionError) as v:
        validate_zero_to_one(5, 'test_param')
        assert "assertion error"  in str(v.value)
    with pytest.raises(AssertionError) as v:
        validate_zero_to_one(0, left='open')
        assert "assertion error"  in str(v.value)    
    with pytest.raises(AssertionError) as v:
        validate_zero_to_one(1, right='open')
        assert "assertion error"  in str(v.value)    
    validate_zero_to_one(0)


# --------------------------------------------------------------------------  #
@mark.utils
@mark.validation
def test_validate_range():
    from mlstudio.utils.validation import validate_range
    with pytest.raises(AssertionError) as v:
        validate_range(param=1, minimum=0, maximum=1, param_name='test_param')
        assert "assertion error"  in str(v.value)
    with pytest.raises(AssertionError) as v:
        validate_range(param=0, minimum=0, maximum=1, param_name='test_param')
        assert "assertion error"  in str(v.value)        
    validate_range(param=1, minimum=0, maximum=1, right='closed')
    validate_range(param=0, minimum=0, maximum=1, left='closed')

# --------------------------------------------------------------------------  #    

@mark.utils
@mark.validation
def test_validate_string():
    from mlstudio.utils.validation import validate_string
    valid_values = ['epoch', 'batch']
    with pytest.raises(ValueError) as v:
        validate_string('hand', valid_values=valid_values)
        assert "assertion error"  in str(v.value)
    validate_string('batch', valid_values=valid_values)

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_activation():    
    from mlstudio.utils.validation import validate_activation
    from mlstudio.supervised.core.activations import Sigmoid
    with pytest.raises(ValueError) as v:
        validate_activation('hand')
        assert "value error"  in str(v.value)
    validate_activation(Sigmoid())

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_objective():    
    from mlstudio.utils.validation import validate_objective
    from mlstudio.supervised.core.objectives import MSE
    with pytest.raises(ValueError) as v:
        validate_objective('hand')
        assert "value error"  in str(v.value)
    validate_objective(MSE())    

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_optimizer():    
    from mlstudio.utils.validation import validate_optimizer
    from mlstudio.supervised.core.optimizers import Adam 
    with pytest.raises(ValueError) as v:
        validate_optimizer('hand')
        assert "value error"  in str(v.value)
    validate_optimizer(Adam())     

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_regularizer():    
    from mlstudio.utils.validation import validate_regularizer
    from mlstudio.supervised.core.regularizers import L1
    with pytest.raises(ValueError) as v:
        validate_regularizer('hand')
        assert "value error"  in str(v.value)
    validate_regularizer(L1())       

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_scorer():    
    from mlstudio.utils.validation import validate_scorer
    from mlstudio.supervised.core.scorers import MSE    
    with pytest.raises(ValueError) as v:
        validate_scorer('hand')
        assert "value error"  in str(v.value)
    validate_scorer(MSE())           

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_task():    
    from mlstudio.utils.validation import validate_task
    from mlstudio.supervised.core.tasks import LogisticRegression    
    with pytest.raises(ValueError) as v:
        validate_task('hand')
        assert "value error"  in str(v.value)
    validate_task(LogisticRegression())           

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_learning_rate_schedule():    
    from mlstudio.utils.validation import validate_learning_rate_schedule
    from mlstudio.supervised.observers.learning_rate import Improvement    
    with pytest.raises(ValueError) as v:
        validate_learning_rate_schedule('hand')
        assert "value error"  in str(v.value)
    validate_learning_rate_schedule(Improvement())           

# --------------------------------------------------------------------------  #    
@mark.utils
@mark.validation
def test_validate_performance():    
    from mlstudio.utils.validation import validate_performance
    from mlstudio.supervised.observers.performance import Performance
    with pytest.raises(ValueError) as v:
        validate_performance('hand')
        assert "value error"  in str(v.value)
    validate_performance(Performance())           