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
from copy import copy, deepcopy

from sklearn.base import BaseEstimator
from mlstudio.utils.data_analyzer import get_feature_info, get_target_info
# --------------------------------------------------------------------------- #
class AbstractDataProcessor(ABC, BaseEstimator):
    """Defines interface for DataProcessor classes."""

    def __init__(self, add_bias_transformer=None, 
                 split_transformer=None, label_encoder=None, 
                 one_hot_label_encoder=None):
        
        self.add_bias_transformer = add_bias_transformer  
        self.split_transformer = split_transformer
        self.label_encoder = label_encoder
        self.one_hot_label_encoder = one_hot_label_encoder

    def _compile(self):
        """Compiles the class for scikit-learn compatibility."""
        self._add_bias_transformer = deepcopy(self.add_bias_transformer)
        self._split_transformer = deepcopy(self.split_transformer)
        self._label_encoder = deepcopy(self.label_encoder)
        self._one_hot_label_encoder = deepcopy(self.one_hot_label_encoder)

    def _check_in_X(self, X, name):
        """Initializes a data object for X input data and grabs some metadata."""
        data = OrderedDict()
        data[name] = OrderedDict()        
        data[name]['metadata'] = OrderedDict()
        data[name]['metadata']['orig'] = get_feature_info(X)
        return data

    def _check_out_X(self, X, name, data):
        """Wraps up preprocessing with addition of metadata on processed X ."""                        
        data[name]['data'] = X
        data[name]['metadata']['processed'] = get_feature_info(X)
        return data        

    def _check_in_y(self, y, name):
        """Initializes a data object for X input data."""
        data = OrderedDict()
        data[name] = OrderedDict()        
        data[name]['metadata'] = OrderedDict()
        data[name]['metadata']['orig'] = get_target_info(y)
        return data

    def _check_out_y(self, y, name, data):         
        """Wraps up preprocessing with additional metadata."""
        data[name]['data'] = y
        data[name]['metadata']['processed'] = get_target_info(y)
        return data

    def _transform_X(self, X):
        """Adds bias term to X."""
        return self._add_bias_transformer.fit_transform(X)

    def _transform_X_y_split(self, X, y, val_size=None, stratify=False, 
                                random_state=None):
        """Splits the data."""
        
        X_train, X_val, y_train, y_val = self._split_transformer.fit_transform(
            X, y, test_size=val_size, stratify=stratify, 
            random_state=random_state)        
        return X_train, X_val, y_train, y_val
    
    def process_X_test_data(self, X, y=None):        
        X = self._transform_X(X)
        return X
    
    @abstractmethod
    def process_y_test_data(self, y):
        """Default behavior for processing y_test data."""
        # Default behavior does nothing to y variable.
        return y

    def process_train_data(self, X, y=None):
        """Default behavior for processing training data."""
        # Default behavior adds bias to X, does nothing to y
        # Copies mutable parameters to private variables for scikit-learn
        self._compile()

        X_train = self._check_in_X(X, name='X_train')
        y_train = self._check_in_y(y, name='y_train')

        X = self._transform_X(X)        
        
        X_train = self._check_out_X(X, name='X_train', data=X_train)
        y_train = self._check_out_y(y, name='y_train', data=y_train)
        data = OrderedDict()
        data = {**X_train, **y_train}
        return data

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        """Default behavior for processing training and validation data."""
        # Copies mutable parameters to private variables for scikit-learn
        data = OrderedDict()
        self._compile()

        X_train = self._check_in_X(X, name='X_train')
        y_train = self._check_in_y(y, name='y_train')
        X_val = self._check_in_X(X, name='X_val')
        y_val = self._check_in_y(y, name='y_val')

        X = self._transform_X(X)

        X_train_xform, X_val_xform, y_train_xform, y_val_xform = \
            self._transform_X_y_split(X=X, y=y, val_size=val_size, 
                                      stratify=False, random_state=random_state)

        X_train = self._check_out_X(X_train_xform, name='X_train', data=X_train)
        y_train = self._check_out_y(y_train_xform, name='y_train', data=y_train)            

        # Confirm split was successful
        if X_val_xform is not None:
            X_val = self._check_out_X(X_val_xform, name='X_val', data=X_val)            
            y_val = self._check_out_y(y_val_xform, name='y_val', data=y_val)        
            data = {**X_train, **X_val, **y_train, **y_val}
        else:
            data = {**X_train, **y_train}
        return data        
    
# --------------------------------------------------------------------------- #
class RegressionDataProcessor(AbstractDataProcessor):
    """Performs preprocessing of data for regression."""

    def process_train_data(self, X, y=None):
        return super(RegressionDataProcessor, self).process_train_data(X, y)

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        return super(RegressionDataProcessor, self).process_train_val_data(X, y, \
            val_size, random_state)

    def process_X_test_data(self, X, y=None):     
        return super(RegressionDataProcessor, self).process_X_test_data(X)

    def process_y_test_data(self, y):           
        return super(RegressionDataProcessor, self).process_y_test_data(y)

        
# --------------------------------------------------------------------------- #
class BinaryClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None):
        X_train = self._check_in_X(X, name='X_train')
        y_train = self._check_in_y(y, name='y_train')

        X = self._transform_X(X)        
        y = self._label_encoder.fit_transform(y)
        
        X_train = self._check_out_X(X, name='X_train', data=X_train)
        y_train = self._check_out_y(y, name='y_train', data=y_train)
        data = OrderedDict()
        data = {**X_train, **y_train}
        return data

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        X_train = self._check_in_X(X, name='X_train')
        y_train = self._check_in_y(y, name='y_train')
        X_val = self._check_in_X(X, name='X_val')
        y_val = self._check_in_y(y, name='y_val')

        X = self._transform_X(X)        

        X_train_xform, X_val_xform, y_train_xform, y_val_xform = \
            self._transform_X_y_split(X=X, y=y, val_size=val_size, 
                                     stratify=True, random_state=random_state)

        y_train_xform = self._label_encoder.fit_transform(y_train_xform)
        X_train = self._check_out_X(X_train_xform, name='X_train', data=X_train)        
        y_train = self._check_out_y(y_train_xform, name='y_train', data=y_train)

        # Check if split was successful
        if X_val_xform is not None:
            y_val_xform = self._label_encoder.transform(y_val_xform)
            X_val = self._check_out_X(X_val_xform, name='X_val', data=X_val)
            y_val = self._check_out_y(y_val_xform, name='y_val', data=y_val)     
        
            data = {**X_train, **X_val, **y_train, **y_val}            
        else:
            data = {**X_train, **y_train}            
        return data           

    def process_X_test_data(self, X, y=None):     
        return super(BinaryClassDataProcessor, self).process_X_test_data(X)

    def process_y_test_data(self, y):           
        y_test = self._check_in_y(y, name='y_test')
        y = self._label_encoder.transform(y)
        y_test = self._check_out_y(y, name='y_test', data=y_test)
        return y_test

        
# --------------------------------------------------------------------------- #
class MultiClassDataProcessor(AbstractDataProcessor):

    def process_train_data(self, X, y=None):
        data = OrderedDict()

        X_train = self._check_in_X(X, name='X_train')
        y_train = self._check_in_y(y, name='y_train')

        X = self._transform_X(X)        
        y = self._label_encoder.fit_transform(y)
        y = self._one_hot_label_encoder.fit_transform(y)
        
        X_train = self._check_out_X(X, name='X_train', data=X_train)
        y_train = self._check_out_y(y, name='y_train', data=y_train)
        
        data = {**X_train, **y_train}
        return data

    def process_train_val_data(self, X, y=None, val_size=None, random_state=None):
        data = OrderedDict()

        X_train = self._check_in_X(X, name='X_train')
        y_train = self._check_in_y(y, name='y_train')
        X_val = self._check_in_X(X, name='X_val')
        y_val = self._check_in_y(y, name='y_val')

        X = self._transform_X(X)

        X_train_xform, X_val_xform, y_train_xform, y_val_xform = \
            self._transform_X_y_split(X=X, y=y, val_size=val_size, 
                                      random_state=random_state)

        y_train_xform = self._label_encoder.fit_transform(y_train_xform)
        y_train_xform = self._one_hot_label_encoder.fit_transform(y_train_xform)
        X_train = self._check_out_X(X_train_xform, name='X_train', data=X_train)        
        y_train = self._check_out_y(y_train_xform, name='y_train', data=y_train)

        # Check if split was successful
        if X_val_xform is not None:
            y_val_xform = self._label_encoder.transform(y_val_xform)
            y_val_xform = self._one_hot_label_encoder.transform(y_val_xform)
            X_val = self._check_out_X(X_val_xform, name='X_val', data=X_val)
            y_val = self._check_out_y(y_val_xform, name='y_val', data=y_val)     
        
            data = {**X_train, **X_val, **y_train, **y_val}            
        else:
            data = {**X_train, **y_train}            
        return data        

    def process_X_test_data(self, X, y=None):     
        return super(BinaryClassDataProcessor, self).process_X_test_data(X)

    def process_y_test_data(self, y):           
        y_test = self._check_in_y(y, name='y_test')
        y = self._label_encoder.transform(y)            
        y = self._one_hot_label_encoder.fit_transform(y)
        y_test = self._check_out_y(y, name='y_test', data=y_test)
        return y_test
