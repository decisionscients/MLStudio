# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \observers.py                                                     #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Friday, July 17th 2020, 1:46:30 am                          #
# Last Modified : Friday, July 17th 2020, 1:46:31 am                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Inversion of Control: Dependency Injection and Containers for Observers."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.supervised.algorithms.optimization.observers import early_stop
from mlstudio.supervised.algorithms.optimization.observers import base, report
from mlstudio.supervised.algorithms.optimization.observers import learning_rate
from mlstudio.utils.print import Printer

# --------------------------------------------------------------------------- #
class EarlyStop(containers.DeclarativeContainer):
    """IoC Container for observer providers."""
    
    factory = providers.Factory(early_stop.EarlyStop,
                                    observer=base.PerformanceObserver) 

# --------------------------------------------------------------------------- #
class Summary(containers.DeclarativeContainer):
    """IoC Container for observer providers."""
    
    factory = providers.Factory(report.Summary,
                                    printer=providers.Factory(Printer)) 
# --------------------------------------------------------------------------- #
class LearningRate(containers.DeclarativeContainer):
    """IoC container for learning rate schedule providers."""

    step_decay_factory = providers.Factory(learning_rate.StepDecay,
                                    eta0=0.1, eta_min=1e-4, decay_factor=0.96,
                                    decay_steps=10)

    time_decay_factory = providers.Factory(learning_rate.TimeDecay,
                                    eta0=0.1, eta_min=1e-4, decay_factor='optimal')

    sqrt_time_decay_factory = providers.Factory(learning_rate.SqrtTimeDecay,
                                    eta0=0.1, eta_min=1e-4, decay_factor=0.5)

    exponential_decay_factory = providers.Factory(learning_rate.ExponentialDecay,
                                    eta0=0.1, eta_min=1e-4, decay_factor=0.1)      

    exponential_step_decay_factory = providers.Factory(learning_rate.ExponentialStepDecay,
                                    eta0=0.1, eta_min=1e-4, decay_factor=0.96,
                                    staircase=False)                         

    polynomial_decay_factory = providers.Factory(learning_rate.PolynomialDecay,
                                    eta0=0.1, eta_min=1e-4, power=1.0)                       

    polynomial_step_decay_factory = providers.Factory(learning_rate.PolynomialStepDecay,
                                    eta0=0.1, eta_min=1e-4, decay_steps=100,
                                    power=1.0)

    power_decay_factory = providers.Factory(learning_rate.PowerSchedule,
                                    eta0=0.1, eta_min=1e-4, decay_steps=100,
                                    power=1.0)                                                                 

    bottou_decay_factory = providers.Factory(learning_rate.BottouSchedule,
                                    eta0=0.1, eta_min=1e-4, decay_factor=0.5)                                                                                                                                                                   

    adaptive_decay_factory = providers.Factory(learning_rate.Adaptive,
                                    eta0=0.1, eta_min=1e-4, decay_factor=0.5,
                                    metric='train_cost', epsilon=0.001,
                                    patience=10, observer=base.PerformanceObserver)      
                        



