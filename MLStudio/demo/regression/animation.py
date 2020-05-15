#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : surface.py                                                        #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, April 10th 2020, 3:27:23 pm                         #
# Last Modified : Friday, April 10th 2020, 3:27:24 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
#%%
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
homedir = str(Path(__file__).parents[3])
sys.path.append(homedir)

from mlstudio.supervised.estimator.gradient import GradientDescentRegressor
from mlstudio.utils.data_manager import StandardScaler
from mlstudio.visual.animations.surface_line import SurfaceLine

# --------------------------------------------------------------------------  #
# Designate file locations
datadir = os.path.join(homedir,"mlstudio/data/Ames/")
filepath = os.path.join(datadir, "train.csv")
# --------------------------------------------------------------------------  #
# Obtain and scale data
cols = ["GrLivArea", "SalePrice"]
df = pd.read_csv(filepath, nrows=500, usecols=cols)
df_samples = df.head()
X = np.array(df['GrLivArea']).reshape(-1,1)
y = df['SalePrice']
scaler = StandardScaler()
X = scaler.fit_transform(X)
# --------------------------------------------------------------------------  #
# Train model
est = GradientDescentRegressor(theta_init=np.array([0,0]))
est.fit(X,y)
# --------------------------------------------------------------------------  #
v = SurfaceLine()
v.animate(est)