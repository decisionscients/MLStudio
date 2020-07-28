# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \data_objects.py                                                  #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Sunday, July 26th 2020, 1:57:21 am                          #
# Last Modified : Sunday, July 26th 2020, 1:57:21 am                          #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Data objects for use during data preprocessing."""
from abc import ABC, abstractmethod
from collections import OrderedDict
import sys
from warnings import warn

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from mlstudio.utils.data_analyzer import get_features, get_target_info
# --------------------------------------------------------------------------- #              
class DataPackage(ABC):
    """Container of DataSet objects."""

    def __init__(self):
        self._datasets = {}

    def get_dataset(self, name):
        try:
            return self._datasets[name]
        except:
            msg = "Dataset {n} doesn't exist in the DataPackage.".format(n=name)
            warn(msg)    

    def add_dataset(self, dataset):
        if dataset.name in self._dataset.keys():
            msg = "Dataset {n} already exists. Use update method to update an existing DataSet object.".format(n=dataset.name)
            raise Exception(msg)
        self._datasets[dataset.name] = dataset

    def update_dataset(self, dataset):
        if dataset.name not in self._dataset.keys():
            msg = "Dataset {n} does not exit. Use add method to add a DataSet object.".format(n=dataset.name)
            raise Exception(msg)        
        self._datasets[dataset.name] = dataset

    def remove_dataset(self, name):
        try:
            del self._datasets[name]
        except:
            msg = "Dataset {n} doesn't exist in the DataPackage.".format(n=name)
            warn(msg)    

# --------------------------------------------------------------------------- #              
class AbstractDataSet(ABC):
    """Defines the interface for DataSet objects."""

    def __init__(self, data):
        self._data = data

    @property
    def data(self):
        return self._data
    
    @property
    def n_observations(self):
        return self._data.shape[0]

    @property
    def size(self):
        return sys.getsizeof(self._data)
# --------------------------------------------------------------------------- #              
#                               X DATASETS                                    #
# --------------------------------------------------------------------------- #              
class AbstractDataSetX(AbstractDataSet):
    """Defines interface for X data sets."""

    @abstractmethod
    def __init__(self, data):
        super(DataSetXTrain, self).__init__(data)
        self._name = 'X'    

    @property
    def name(self):
        return self._name

    @property
    def features(self):
        return get_features(self._data)

    @property
    def n_features(self):
        return self._data.shape[1]

# --------------------------------------------------------------------------- #              
class DataSetXTrain(AbstractDataSetX):

    def __init__(self, data):
        super(DataSetXTrain, self).__init__(data)
        self._name = 'X_train'

# --------------------------------------------------------------------------- #              
class DataSetXVal(AbstractDataSetX):

    def __init__(self, data):
        super(DataSetXTrain, self).__init__(data)
        self._name = 'X_val'

# --------------------------------------------------------------------------- #              
class DataSetXTest(AbstractDataSetX):

    def __init__(self, data):
        super(DataSetXTrain, self).__init__(data)
        self._name = 'X_test'

# --------------------------------------------------------------------------- #              
#                               y DATASETS                                    #
# --------------------------------------------------------------------------- #              
class AbstractDataSetY(AbstractDataSet):
    """Defines interface for X data sets."""

    @abstractmethod
    def __init__(self, data):
        super(DataSetYTrain, self).__init__(data)
        self._name = 'y'    
        self._metadata = get_target_info(data)

    @property
    def name(self):
        return self._name
    
    @property
    def data_type(self):
        return self._metadata['data_type']

    @property
    def data_class(self):
        return self._metadata['data_class']        

    @property
    def classes(self):
        return self._metadata['classes']                

    @property
    def n_classes(self):
        return self._metadata['n_classes']                
    

# --------------------------------------------------------------------------- #              
class DataSetYTrain(AbstractDataSetY):

    def __init__(self, data):
        super(DataSetYTrain, self).__init__(data)
        self._name = 'y_train'

# --------------------------------------------------------------------------- #              
class DataSetYVal(AbstractDataSetY):

    def __init__(self, data):
        super(DataSetYTrain, self).__init__(data)
        self._name = 'y_val'

# --------------------------------------------------------------------------- #              
class DataSetYTest(AbstractDataSetY):

    def __init__(self, data):
        super(DataSetXTrain, self).__init__(data)
        self._name = 'y_test'
# --------------------------------------------------------------------------- #
class DataPackageTransformer(BaseEstimator, TransformerMixin):
    """Defines interface for DataPackage Transformer objects."""
    def __init__(self, dataset_X=None, dataset_y=None, data_package=None):
        self._dataset_X = dataset_X
        self._dataset_y = dataset_y
        self._data_package = data_package

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = self._dataset_X(data=X).data
        print(self._dataset_X(data=X).data)
        y = self._dataset_y(data=y).data
        self._data_package.add_dataset(X)
        self._data_package.add_dataset(y)
        return self._data_package()

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X, y)        

