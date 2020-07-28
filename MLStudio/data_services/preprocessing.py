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
# --------------------------------------------------------------------------- #
class AbstractDataProcessor(ABC):
    """Defines interface for DataProcessor classes."""

    def __init__(self, add_bias_transformer=None, 
                 split_transformer=None, label_encoder=None, 
                 one_hot_label_encoder=None):
        
        self._add_bias_transformer = add_bias_transformer  
        self._split_transformer = split_transformer
        self._label_encoder = label_encoder
        self._one_hot_label_encoder = one_hot_label_encoder

    @abstractmethod
    def process_train_data(self, X, y=None, random_state=None):
        """Processes data for training."""
        pass

    @abstractmethod
    def process_train_val_data(self, val_size, X, y=None, random_state=None):
        """Processes training and validation data for training."""
        pass
    
    def process_X_test_data(self, X, y=None, random_state=None):
        return self._add_bias_transformer.fit_transform(X)
    
    def process_y_test_data(self, y, random_state=None):
        """Processes y test data."""
        return y    
    
# --------------------------------------------------------------------------- #
class RegressionDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)
        return X, y

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)
        X_train, X_val, y_train, y_val = self._split_transformer.fit_transform(
            X, y, test_size=val_size, random_state=random_state)

        return X_train, X_val, y_train, y_val
        
# --------------------------------------------------------------------------- #
class BinaryClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)
        y = self._label_encoder.fit_transform(y)
        return X, y

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)

        X_train, X_val, y_train, y_val = self._split_transformer.fit_transform(
            X, y, test_size=val_size, stratify=True, 
            random_state=random_state)

        y_train = self._label_encoder.fit_transform(y_train)
        y_val = self._label_encoder.transform(y_val)

        return X_train, X_val, y_train, y_val

    def process_y_test_data(self, y, random_state=None):
        return self._label_encoder.fit_transform(y)
        
# --------------------------------------------------------------------------- #
class MultiClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)
        y = self._label_encoder.fit_transform(y)
        y = self._one_hot_label_encoder.fit_transform(y)
        return X, y

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        X = self._add_bias_transformer.fit_transform(X)

        X_train, X_val, y_train, y_val = self._split_transformer.fit_transform(
            X, y, test_size=val_size, stratify=True, 
            random_state=random_state)

        y_train = self._label_encoder.fit_transform(y_train)
        y_train = self._one_hot_label_encoder.fit_transform(y_train)
        y_val = self._label_encoder.transform(y_val)
        y_val = self._one_hot_label_encoder.transform(y_val)

        return X_train, X_val, y_train, y_val

    def process_y_test_data(self, y, random_state=None):
        y = self._label_encoder.fit_transform(y)
        y = self._one_hot_label_encoder.fit_transform(y)
        return y

        
