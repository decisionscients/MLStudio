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

from mlstudio.supervised.algorithms.optimization.services import activations
from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.supervised.algorithms.optimization.services import tasks
from mlstudio.supervised.performance import regression, binary_class
from mlstudio.factories.data_processor import DataProcessors
# --------------------------------------------------------------------------- #
class Task(containers.DeclarativeContainer):
    """IoC container for task providers."""

    linear_regression_factory = providers.Factory(tasks.LinearRegression,
                                          loss=loss.Quadratic(),
                                          scorer=regression.R2(),
                                          data_processor=DataProcessors.regression(),
                                          activation=None)

    logistic_regression_factory = providers.Factory(tasks.LogisticRegression,
                                          loss=loss.CrossEntropy(),
                                          scorer=classification.Accuracy(),
                                          data_processor=DataProcessors.binary_classification(),
                                          activation=activations.Sigmoid())     

    multiclass_classification_factory = providers.Factory(tasks.MulticlassClassification,
                                          loss=loss.CategoricalCrossEntropy(),
                                          scorer=classification.Accuracy(),
                                          data_processor=DataProcessors.multiclass_classification(),
                                          activation=activations.Softmax())

