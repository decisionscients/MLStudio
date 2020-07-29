# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \preprocessing.py                                                 #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 25th 2020, 3:17:33 pm                        #
# Last Modified : Saturday, July 25th 2020, 3:17:34 pm                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Data preprocessing for machine learning and model development."""
from abc import ABC, abstractmethod
from collections import OrderedDict
from copy import copy

from sklearn.base import BaseEstimator
from mlstudio.utils.data_analyzer import get_feature_info, get_target_info
# --------------------------------------------------------------------------- #
class AbstractDataProcessor(ABC, BaseEstimator):
    """Defines interface for DataProcessor classes."""

    def __init__(self, add_bias_transformer=None, 
                 split_transformer=None, label_encoder=None, 
                 one_hot_label_encoder=None):
        
        self._add_bias_transformer = add_bias_transformer  
        self._split_transformer = split_transformer
        self._label_encoder = label_encoder
        self._one_hot_label_encoder = one_hot_label_encoder

    def _get_X_metadata(self, X):
        return get_feature_info(X)

    def _get_y_metadata(self, y):
        return get_target_info(y)   

    def _format_data_package(self, X_train=None, y_train=None, 
                                   X_val=None, y_val=None,
                                   X_test=None, y_test=None):     
        d = OrderedDict()
        if X_train is not None:
            d['X_train'] = OrderedDict()
            d['X_train']['data'] = X_train
            d['X_train']['metadata'] = self._get_X_metadata(X_train.copy())

        if X_val is not None:
            d['X_val'] = OrderedDict()
            d['X_val']['data'] = X_val
            d['X_val']['metadata'] = self._get_X_metadata(X_val.copy())                                             

        if X_test is not None:
            d['X_test'] = OrderedDict()
            d['X_test']['data'] = X_test
            d['X_test']['metadata'] = self._get_X_metadata(X_test.copy())  

        if y_train is not None:
            d['y_train'] = OrderedDict()
            d['y_train']['data'] = y_train
            d['y_train']['metadata'] = self._get_y_metadata(y_train.copy())

        if y_val is not None:
            d['y_val'] = OrderedDict()
            d['y_val']['data'] = y_val
            d['y_val']['metadata'] = self._get_y_metadata(y_val.copy())                                             

        if y_test is not None:
            d['y_test'] = OrderedDict()
            d['y_test']['data'] = y_test
            d['y_test']['metadata'] = self._get_y_metadata(y_test.copy())     

        return d                                                                                                                       

    @abstractmethod
    def process_train_data(self, X, y=None):
        """Processes data for training."""
        pass

    @abstractmethod
    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        """Processes training and validation data for training."""
        pass
    
    def process_X_test_data(self, X, y=None):
        X = self._add_bias_transformer.fit_transform(X)
        return self._format_data_package(X_test=X)
    
    def process_y_test_data(self, y):
        """Processes y test data."""
        return self._format_data_package(y_test=y)    
    
# --------------------------------------------------------------------------- #
class RegressionDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None):
        d = OrderedDict()
        X = self._add_bias_transformer.fit_transform(X)
        return self._format_data_package(X_train=X, y_train=y)

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)
        X_train, X_val, y_train, y_val = self._split_transformer.fit_transform(
            X, y, test_size=val_size, random_state=random_state)

        return self._format_data_package(X_train=X_train, X_val=X_val, 
                                         y_train=y_train, y_val=y_val)
        
# --------------------------------------------------------------------------- #
class BinaryClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None):
        X = self._add_bias_transformer.fit_transform(X)
        y = self._label_encoder.fit_transform(y)
        return self._format_data_package(X_train=X, y_train=y)

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)

        X_train, X_val, y_train, y_val = self._split_transformer.fit_transform(
            X, y, test_size=val_size, stratify=True, 
            random_state=random_state)

        y_train = self._label_encoder.fit_transform(y_train)
        y_val = self._label_encoder.transform(y_val)

        return self._format_data_package(X_train=X_train, X_val=X_val, 
                                         y_train=y_train, y_val=y_val)

    def process_y_test_data(self, y):
        y = self._label_encoder.fit_transform(y)
        return self._format_data_package(y_test=y)
        
# --------------------------------------------------------------------------- #
class MultiClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None):
        X = self._add_bias_transformer.fit_transform(X)
        y = self._label_encoder.fit_transform(y)
        y = self._one_hot_label_encoder.fit_transform(y)
        return self._format_data_package(X_train=X, y_train=y)

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)

        X_train, X_val, y_train, y_val = self._split_transformer.fit_transform(
            X, y, test_size=val_size, stratify=True, 
            random_state=random_state)

        y_train = self._label_encoder.fit_transform(y_train)
        y_train = self._one_hot_label_encoder.fit_transform(y_train)
        y_val = self._label_encoder.transform(y_val)
        y_val = self._one_hot_label_encoder.transform(y_val)

        return self._format_data_package(X_train=X_train, X_val=X_val, 
                                         y_train=y_train, y_val=y_val)

    def process_y_test_data(self, y):
        y = self._label_encoder.fit_transform(y)
        y = self._one_hot_label_encoder.fit_transform(y)
        return self._format_data_package(y_test=y)

        
