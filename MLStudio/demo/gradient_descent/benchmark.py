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
from collections import OrderedDict
from datetime import datetime
import os
from pathlib import Path
import sys

import pandas as pd
import numpy as np
homedir = str(Path(__file__).parents[3])
demodir = str(Path(__file__).parents[1])
sys.path.append(homedir)

from mlstudio.supervised.machine_learning.gradient_descent import GradientDescent
from mlstudio.visual.animations.surface import Surface
from mlstudio.visual.animations.surface_contour import SurfaceContour
from mlstudio.supervised.core.objectives import Adjiman, BartelsConn, Himmelblau
from mlstudio.supervised.core.objectives import Leon, Rosenbrock, StyblinskiTank
from mlstudio.supervised.core.objectives import SumSquares, ThreeHumpCamel
from mlstudio.supervised.core.optimizers import Classic, Momentum, Nesterov
from mlstudio.supervised.core.optimizers import Adagrad, Adadelta, RMSprop
from mlstudio.supervised.core.optimizers import Adam, AdaMax, AdamW
from mlstudio.supervised.core.optimizers import Nadam, AMSGrad, QHAdam
from mlstudio.supervised.core.optimizers import QuasiHyperbolicMomentum
from mlstudio.supervised.core.optimizers import AggMo
from mlstudio.supervised.callbacks.learning_rate import ExponentialSchedule, TimeDecay
from mlstudio.utils.file_manager import save_df

# --------------------------------------------------------------------------  #
# Designate file locations
figures = os.path.join(demodir, "figures")
# --------------------------------------------------------------------------  #
# Package up the objective functions
objectives = [Adjiman(), BartelsConn(), Himmelblau(), Leon(), Rosenbrock(),
              StyblinskiTank(), SumSquares(), ThreeHumpCamel()]

optimizers = [Momentum(), Nesterov(), Adagrad(), Adadelta(), RMSprop(), Adam(),
              AdaMax(), Nadam(), AMSGrad(), AdamW(), QHAdam(),
              QuasiHyperbolicMomentum()]

# --------------------------------------------------------------------------  #
# Train models
packages = OrderedDict()
objectives_results = OrderedDict()
results = []
for objective in objectives:
    estimators = OrderedDict()
    optimizers_results = OrderedDict()

    for optimizer in optimizers:
        estimators[optimizer.name] = GradientDescent(learning_rate=0.1,
                                        theta_init=objective.start, 
                                        epochs=500, objective=objective,
                                        schedule=TimeDecay(decay_factor=0.9),
                                        optimizer=optimizer)
        estimators[optimizer.name].fit()
        d = {}
        d['DateTime'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        d['Objective'] = estimators[optimizer.name].objective.name
        d['Optimizer'] = optimizer.name
        d['Epochs'] = estimators[optimizer.name].epochs        
        d['Starting Learning Rate'] = estimators[optimizer.name].learning_rate
        d['Final Learning Rate'] = estimators[optimizer.name].eta
        d['Schedule'] = estimators[optimizer.name].schedule.name
        d['Decay Factor'] = estimators[optimizer.name].schedule.decay_factor
        d['gradient'] = np.linalg.norm(estimators[optimizer.name].gradient_)
        d['gradient_min'] = np.min(estimators[optimizer.name].blackbox_.epoch_log.get('gradient_norm'))
        d['gradient_max'] = np.max(estimators[optimizer.name].blackbox_.epoch_log.get('gradient_norm'))        
        d['gradient_mean'] = np.mean(estimators[optimizer.name].blackbox_.epoch_log.get('gradient_norm'))        
        d['True'] = np.linalg.norm(objective.minimum)
        d['Size'] = np.linalg.norm(estimators[optimizer.name].theta_)
        d['Diff'] = d['Size'] - d['True']
        d['Cost'] = estimators[optimizer.name].blackbox_.epoch_log.get('train_cost')[-1]
        optimizers_results[optimizer.name] = d
        results.append(d)

    packages[objective.name] = estimators   
    objectives_results[objective.name] = optimizers_results 
df = pd.DataFrame(results)
df_filename = "Benchmark Optimizations.csv"
save_df(df, figures, df_filename)

# --------------------------------------------------------------------------  #
# Render plots
for title, package in packages.items():        
    v = Surface()
    v.animate(estimators=package, directory=figures, filename=title + " Optimization.html")        

