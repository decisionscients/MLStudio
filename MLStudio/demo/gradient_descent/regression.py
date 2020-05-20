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
demodir = str(Path(__file__).parents[1])
sys.path.append(homedir)

from mlstudio.supervised.machine_learning.gradient_descent import GradientDescentRegressor
from mlstudio.utils.data_manager import StandardScaler
from mlstudio.visual.animations.surface_line1 import SurfaceLine1
from mlstudio.visual.animations.gradient import MultiModelSearch3D

# --------------------------------------------------------------------------  #
# Designate file locations
datadir = os.path.join(homedir,"mlstudio/data/Ames/")
filepath = os.path.join(datadir, "train.csv")
figures = os.path.join(demodir, "figures")
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
bgd = GradientDescentRegressor(theta_init=np.array([0,0]), epochs=500)
sgd = GradientDescentRegressor(theta_init=np.array([0,0]), epochs=500, batch_size=1)
mbgd = GradientDescentRegressor(theta_init=np.array([0,0]), epochs=500, batch_size=64)
bgd.fit(X,y)
sgd.fit(X,y)
mbgd.fit(X,y)
models = [bgd, sgd, mbgd]
# models = {'Batch Gradient Descent': bgd, 'Stochastic Gradient Descent': sgd,
#           'Minibatch Gradient Descent': mbgd}

# --------------------------------------------------------------------------  #
v = SurfaceLine1()
v.animate(models=models, directory=demodir)
# v = MultiModelSearch3D()
# ani = v.search(models, frames=None, directory=figures)
