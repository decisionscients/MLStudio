#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : param_grid.py                                                      #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Wednesday, March 18th 2020, 2:11:31 am                       #
# Last Modified : Wednesday, March 18th 2020, 2:11:31 am                       #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
#%%
from sklearn.model_selection import ParameterGrid

param_grid = {'a': [1, 2], 'b': [True, False]}
grid = ParameterGrid(param_grid)
for params in grid:
    print("============")
    for param, value in params.items():
        print("----------")
        print(param)
        print(value)

# %%
