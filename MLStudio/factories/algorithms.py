# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \estimators.py                                                    #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Friday, July 17th 2020, 12:35:16 am                         #
# Last Modified : Friday, July 17th 2020, 12:35:16 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Inversion of Control: Dependency Injection and Containers for Estimators."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.supervised.algorithms.optimization.gradient_descent import GradientDescent
from mlstudio.supervised.algorithms.optimization.observers import base, debug
from mlstudio.supervised.algorithms.optimization.observers import report, history
from mlstudio.supervised.algorithms.optimization.services import optimizers
from mlstudio.factories.tasks import Task
from mlstudio.factories.observers import Summary


# --------------------------------------------------------------------------- #
class GradientDescent(containers.DeclarativeContainer):
    """IoC Container for gradient descent estimator providers."""                                    
    regression = providers.Factory(GradientDescent,
                                    task=Task.regression(),
                                    eta0=0.01,
                                    epochs=1000,
                                    batch_size=None,
                                    val_size=0.3,
                                    theta_init=None,
                                    optimizer=optimizers.GradientDescentOptimizer(),                                    
                                    early_stop=None,
                                    learning_rate=None,                           
                                    observer_list=base.ObserverList(),
                                    progress=report.Progress(),
                                    blackbox=history.BlackBox(),
                                    summary=Summary.factory(),
                                    verbose=False,
                                    random_state=None,
                                    check_gradient=False,
                                    gradient_checker=debug.GradientCheck()
                                    )

    binaryclass = providers.Factory(GradientDescent,
                                    task=Task.binaryclass(),
                                    eta0=0.01,
                                    epochs=1000,
                                    batch_size=None,
                                    val_size=0.3,
                                    theta_init=None,
                                    optimizer=optimizers.GradientDescentOptimizer(),
                                    early_stop=None,
                                    learning_rate=None,                           
                                    observer_list=base.ObserverList(),
                                    progress=report.Progress(),
                                    blackbox=history.BlackBox(),
                                    summary=Summary.factory(),
                                    verbose=False,
                                    random_state=None,
                                    check_gradient=False,
                                    gradient_checker=debug.GradientCheck()
                                    )

    multiclass = providers.Factory(GradientDescent,
                                    task=Task.multiclass(),
                                    eta0=0.01,
                                    epochs=1000,
                                    batch_size=None,
                                    val_size=0.3,
                                    theta_init=None,
                                    optimizer=optimizers.GradientDescentOptimizer(),
                                    early_stop=None,
                                    learning_rate=None,                           
                                    observer_list=base.ObserverList(),
                                    progress=report.Progress(),
                                    blackbox=history.BlackBox(),
                                    summary=Summary.factory(),
                                    verbose=False,
                                    random_state=None,
                                    check_gradient=False,
                                    gradient_checker=debug.GradientCheck()
                                    )
