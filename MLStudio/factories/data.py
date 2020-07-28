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
class DataPackageTransformers(containers.DeclarativeContainer):
    from mlstudio.data_services.data_objects import DataPackage
    from mlstudio.data_services.data_objects import DataSetXTrain, DataSetYTrain
    from mlstudio.data_services.data_objects import DataSetXVal, DataSetYVal
    from mlstudio.data_services.data_objects import DataSetXTest, DataSetYTest    
    from mlstudio.data_services.data_objects import DataPackageTransformer
    # from mlstudio.data_services.preprocessing import MakePredictDataPackagePipelineStep
    # from mlstudio.data_services.preprocessing import MakeScoreDataPackagePipelineStep

    train_data_package = providers.Factory(DataPackageTransformer,
                                                       dataset_X = DataSetXTrain,
                                                       dataset_y = DataSetYTrain,
                                                       data_package=DataPackage())


X, y = load_boston(return_X_y=True)
print(DataPackageTransformers.train_data_package().transform(X, y))
# print("**********************")
# print(dp())
# d = dp().transform(X, y)
# print(d)
# print("**********************")
