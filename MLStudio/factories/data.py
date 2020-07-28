# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \data.py                                                          #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Tuesday, July 28th 2020, 2:54:55 am                         #
# Last Modified : Tuesday, July 28th 2020, 2:54:55 am                         #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Container for data object factories."""
#%%
from pathlib import Path
import site
PROJECT_DIR = Path(__file__).resolve().parents[2]
site.addsitedir(PROJECT_DIR)

import dependency_injector.containers as containers
import dependency_injector.providers as providers
from sklearn.datasets import load_boston

# --------------------------------------------------------------------------- #
class DataTransformers(containers.DeclarativeContainer):
    from mlstudio.utils.data_manager import AddBiasTerm, DataSplitter
    from mlstudio.utils.data_manager import LabelEncoder, OneHotLabelEncoder
    
    add_bias_term = providers.Factory(AddBiasTerm)
    data_splitter = providers.Factory(DataSplitter)
    label_encoder = providers.Factory(LabelEncoder)
    one_hot_label_encoder = providers.Factory(OneHotLabelEncoder)

class DataProcessors(containers.DeclarativeContainer):
    from mlstudio.data_services.preprocessing import RegressionDataProcessor
    from mlstudio.data_services.preprocessing import BinaryClassDataProcessor
    from mlstudio.data_services.preprocessing import MultiClassDataProcessor

    regression = providers.Factory(RegressionDataProcessor,
                                   add_bias_transformer=DataTransformers.add_bias_term,
                                   split_transformer=DataTransformers.data_splitter)

    binary_class = providers.Factory(BinaryClassDataProcessor,
                                   add_bias_transformer=DataTransformers.add_bias_term,
                                   split_transformer=DataTransformers.data_splitter,
                                   label_encoder=DataTransformers.label_encoder)                                   

    multi_class = providers.Factory(MultiClassDataProcessor,
                                   add_bias_transformer=DataTransformers.add_bias_term,
                                   split_transformer=DataTransformers.data_splitter,
                                   label_encoder=DataTransformers.label_encoder,
                                   one_hot_label_encoder=DataTransformers.one_hot_label_encoder)                                   
