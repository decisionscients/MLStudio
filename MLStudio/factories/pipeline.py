# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \pipeline.py                                                      #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 25th 2020, 5:56:27 pm                        #
# Last Modified : Saturday, July 25th 2020, 5:56:28 pm                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Factory containers for data processing pipelines."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import collections
import dependency_injector.containers as containers
import dependency_injector.providers as providers

from mlstudio.data_services.preprocessing import AbstractPipelineConfig
# --------------------------------------------------------------------------- #
class PipelineConfigFactory(AbstractPipelineConfig):
    """Creates configuration objects on the fly."""

    def __init__(self, name, config):
        self._name = name
        self._config = config

    def build(self):
        globals()[self._name] = type(self._name, (AbstractPipelineConfig,), self._config)
        symbol_table = globals()
        return symbol_table[self._name]
# --------------------------------------------------------------------------- #
class PipelineSteps(containers.DeclarativeContainer):
    """IoC container for pipeline step providers."""
    from mlstudio.data_services.preprocessing import AddBiasDataPipelineStep
    from mlstudio.data_services.preprocessing import AddBiasDataPipelineStep
    from mlstudio.data_services.preprocessing import ShuffleDataPipelineStep
    from mlstudio.data_services.preprocessing import SplitDataPipelineStep
    from mlstudio.data_services.preprocessing import EncodeLabelsDataPipelineStep
    from mlstudio.data_services.preprocessing import OneHotEncodeLabelsDataPipelineStep
    from mlstudio.utils.data_manager import AddBiasTerm, DataShuffler
    from mlstudio.utils.data_manager import OneHotLabelEncoder, DataSplitter
    from mlstudio.utils.data_manager import LabelEncoder    

    add_bias_term_factory = providers.Factory(AddBiasDataPipelineStep,
                                             transformer=AddBiasTerm())

    shuffle_factory = providers.Factory(ShuffleDataPipelineStep,
                                        transformer=DataShuffler())

    split_data_factory = providers.Factory(SplitDataPipelineStep,
                                transformer=DataSplitter())     

    encode_labels_factory = providers.Factory(EncodeLabelsDataPipelineStep,
                                          transformer=LabelEncoder())                        

    one_hot_encode_labels_factory = providers.Factory(OneHotEncodeLabelsDataPipelineStep,
                                          transformer=OneHotLabelEncoder())


# --------------------------------------------------------------------------- #
class Pipeline(containers.DeclarativeContainer):
    """IoC container for pipeline providers."""
    from mlstudio.data_services.preprocessing import DataPipeline
    factory = providers.Factory(DataPipeline)
