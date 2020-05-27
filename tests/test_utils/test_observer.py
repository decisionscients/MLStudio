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
"""Tests observer."""
import numpy as np
import pytest
from pytest import mark

from mlstudio.utils.observers import Performance

# --------------------------------------------------------------------------  #
#                          TEST OBSERVER                                      #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.observer
def test_observer_validation():    
    with pytest.raises(TypeError) as v:
        observer = Performance(metric=Performance())    
        observer.initialize()
    with pytest.raises(ValueError) as v:
        observer = Performance(metric='hair')
        observer.initialize()
    with pytest.raises(TypeError) as v:
        observer = Performance(scorer='hair')
        observer.initialize()
    with pytest.raises(TypeError) as v:
        observer = Performance(epsilon='hair')                
        observer.initialize()        
    with pytest.raises(ValueError) as v:
        observer = Performance(epsilon=1.1)                
        observer.initialize()
    with pytest.raises(TypeError) as v:
        observer = Performance(patience='hair')        
        observer.initialize()

@mark.utils
@mark.observer
def test_observer_train_cost(get_log):    
    observer = Performance(epsilon=0.1)
    observer.initialize()
    log = get_log
    critical_points = []
    for i in range(len(log)):
        result = observer.model_is_stable(i+1, log[i])
        if result:
            critical_points.append(result)
    assert np.array_equal(np.array(critical_points), np.array([6, 17])), \
        "Performance observer returned invalid critical points" 


