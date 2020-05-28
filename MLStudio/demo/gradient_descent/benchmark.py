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
from mlstudio.supervised.core.objectives import Adjiman, StyblinskiTank
from mlstudio.supervised.core.objectives import ThreeHumpCamel, Ursem01, Branin02
from mlstudio.supervised.core.optimizers import Classic, Momentum, Nesterov
from mlstudio.supervised.core.optimizers import Adagrad, Adadelta, RMSprop
from mlstudio.supervised.core.optimizers import Adam, AdaMax, AdamW
from mlstudio.supervised.core.optimizers import Nadam, AMSGrad, QHAdam
from mlstudio.supervised.core.optimizers import QuasiHyperbolicMomentum
from mlstudio.supervised.core.optimizers import AggMo
from mlstudio.supervised.callbacks.learning_rate import ExponentialSchedule, TimeDecay
from mlstudio.utils.data_analyzer import cosine
from mlstudio.utils.file_manager import save_df

# --------------------------------------------------------------------------  #
# Designate file locations
figures = os.path.join(demodir, "figures")
# --------------------------------------------------------------------------  #
# Package up the objective functions
optimizers = [Momentum(), Nesterov(), Adagrad(), Adadelta(), RMSprop(), Adam(),
              AdaMax(), Nadam(), AMSGrad(), AdamW(), QHAdam(),
              QuasiHyperbolicMomentum()]

objectives = [Adjiman(), Branin02(), Ursem01(),
              StyblinskiTank(), ThreeHumpCamel()]

objectives = [Adjiman()]

# --------------------------------------------------------------------------  #
# Train models
solutions = OrderedDict()
results = []
for objective in objectives:
    estimators = OrderedDict()   

    for optimizer in optimizers:
        estimators[optimizer.name] = {}
        model = GradientDescent(learning_rate=0.01,
                                theta_init=objective.start, 
                                epochs=500, objective=objective,
                                optimizer=optimizer)
        model.fit()
        sim = cosine(objective.minimum, model.theta_)
        d = {}
        d['DateTime'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        d['Objective'] = model.objective.name
        d['Optimizer'] = optimizer.name
        d['Epochs'] = model.epochs        
        d['Starting Learning Rate'] = model.learning_rate
        d['Final Learning Rate'] = model.eta
        if model.schedule:
            d['Schedule'] = model.schedule.name 
            d['Decay Factor'] = model.schedule.decay_factor 
        else:
            d['Schedule'] = None
            d['Decay Factor'] = None        
        d['gradient_min'] = np.min(model.blackbox_.epoch_log.get('gradient_norm'))
        d['gradient_max'] = np.max(model.blackbox_.epoch_log.get('gradient_norm'))        
        d['gradient_mean'] = np.mean(model.blackbox_.epoch_log.get('gradient_norm'))        
        d['True'] = objective.minimum
        d['est'] = model.theta_
        d['sim'] = sim
        results.append(d)
        estimators[optimizer.name]['model'] = model
        estimators[optimizer.name]['results'] = d        

    solutions[objective.name] = estimators       

df = pd.DataFrame(results)
df_filename = "Benchmark Optimizations.csv"
save_df(df, figures, df_filename)

# --------------------------------------------------------------------------  #
# Render plots
for title, solution in solutions.items():        
    v = SurfaceContour()
    v.animate(estimators=solution, directory=figures, filename=title + " Optimization.html", show=True)        

