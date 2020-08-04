# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \tasks.py                                                         #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Friday, July 17th 2020, 12:34:48 am                         #
# Last Modified : Friday, July 17th 2020, 12:34:49 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Inversion of Control: Dependency Injection and Containers for Tasks."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.utils.data_manager import GradientScaler
from mlstudio.factories.data import DataProcessors
# --------------------------------------------------------------------------- #
class Loss(containers.DeclarativeContainer):
    """IoC container for task providers."""

    quadratic = providers.Factory(loss.Quadratic,
                                  regularizer=None,
                                  gradient_scaling=False,
                                  gradient_scaler=GradientScaler())

    crossentropy = providers.Factory(loss.CrossEntropy,
                                     regularizer=None,
                                     gradient_scaling=False,
                                     gradient_scaler=GradientScaler())

    categorical_crossentropy = providers.Factory(loss.CategoricalCrossEntropy,
                                                 regularizer=None,
                                                 gradient_scaling=False,
                                                 gradient_scaler=GradientScaler())
