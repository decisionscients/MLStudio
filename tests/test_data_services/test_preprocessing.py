# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# File    : \test_preprocessing.py                                            #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------- #
# Author  : John James                                                        #
# Company : nov8.ai                                                           #
# Email   : jjames@nov8.ai                                                    #
# URL     : https://github.com/nov8ai/MLStudio                                #
# --------------------------------------------------------------------------- #
# Created       : Saturday, July 25th 2020, 9:54:15 pm                        #
# Last Modified : Saturday, July 25th 2020, 9:54:15 pm                        #
# Modified By   : John James (jjames@nov8.ai)                                 #
# --------------------------------------------------------------------------- #
# License : BSD                                                               #
# Copyright (c) 2020 nov8.ai                                                  #
# =========================================================================== #
"""Tests data preprocessing pipeline."""
#%%
import numpy as np
import pytest
from pytest import mark
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification, make_regression

from mlstudio.factories.data import DataProcessors

# --------------------------------------------------------------------------  #  
def check_add_bias(X, X_train, test):
    assert X_train.shape[1] == X.shape[1] + 1, test + ": bias term wasn't added."

def check_split(X, y, X_train, y_train, X_val, y_val, test):
    assert X_train.shape[1] == X.shape[1] + 1, test + ": bias term wasn't added."        
    assert X.shape[0] > X_train.shape[0],  test + ": split didn't happen."        
    assert X_train.shape[0] == y_train.shape[0], test + ": X, y shape mismatch."
    assert X_val.shape[0] == y_val.shape[0],  test + ": X, y shape mismatch."        
    assert X_train.shape[0] > X_val.shape[0],  test + ": Train size not greater than test."        

def check_label_encoder(y, test):
    assert all(y) in range(len(np.unique(y))), test + ": label encoding didn't work"

def check_one_hot_label_encoder(y, test):
    assert np.sum(y) == y.shape[0], test + ": one-hot-label encoding didn't binarize"
    assert y.shape[1] > 2, test + ": one-hot-label encoding didn't create vector."


@mark.data_processing
@mark.regression_data
class RegressionDataTests:

    _test = "Regression data"

    def test_regression_train_data(self, get_regression_data):
        X, y = get_regression_data
        data_processor = DataProcessors.regression
        X_train, y_train = data_processor().process_train_data(X, y)
        check_add_bias(X, X_train,test = self._test)        

    def test_regression_train_val_data(self, get_regression_data):
        X, y = get_regression_data
        data_processor = DataProcessors.regression
        X_train, X_val, y_train, y_val = data_processor().process_train_val_data(X, y, val_size=0.3)
        check_add_bias(X, X_train, test = self._test)        
        check_add_bias(X, X_val, test = self._test)        
        check_split(X, y, X_train, y_train, X_val, y_val, test=self._test)

    def test_regression_X_test_data(self, get_regression_data):
        X, y = get_regression_data
        data_processor = DataProcessors.regression
        X_train = data_processor().process_X_test_data(X)
        check_add_bias(X, X_train, test=self._test)

@mark.data_processing
@mark.binary_class_data
class BinaryClassDataTests:

    _test = "Binary classification data"

    def test_binary_class_train_data(self, get_logistic_regression_data):
        X, y = get_logistic_regression_data
        y = np.random.choice(["hat", "bowl"], size=y.shape[0])
        data_processor = DataProcessors.binary_class
        X_train, y_train = data_processor().process_train_data(X, y)
        check_add_bias(X, X_train,test = self._test)        

    def test_binary_class_train_val_data(self, get_logistic_regression_data):
        X, y = get_logistic_regression_data
        y = np.random.choice(["hat", "bowl"], size=y.shape[0])
        data_processor = DataProcessors.binary_class
        X_train, X_val, y_train, y_val = data_processor().process_train_val_data(X, y, val_size=0.3)
        check_add_bias(X, X_train, test = self._test)        
        check_add_bias(X, X_val, test = self._test)        
        check_split(X, y, X_train, y_train, X_val, y_val, test=self._test)
        check_label_encoder(y_train, test=self._test)
        check_label_encoder(y_val, test=self._test)

    def test_binary_class_X_test_data(self, get_logistic_regression_data):
        X, y = get_logistic_regression_data
        y = np.random.choice(["hat", "bowl"], size=y.shape[0])
        data_processor = DataProcessors.binary_class
        X_train = data_processor().process_X_test_data(X)
        check_add_bias(X, X_train, test=self._test)

    def test_binary_class_y_test_data(self, get_logistic_regression_data):
        X, y = get_logistic_regression_data
        y = np.random.choice(["hat", "bowl"], size=y.shape[0])        
        data_processor = DataProcessors.binary_class
        y_test = data_processor().process_y_test_data(y) 
        check_label_encoder(y_test, test=self._test)

@mark.data_processing
@mark.multi_class_data
class MultiClassDataTests:

    _test = "Multi classification data"

    def test_multi_class_train_data(self, get_multiclass_classification_data):
        X, y = get_multiclass_classification_data
        y = np.random.choice(["hat", "bowl", "junky", "riding", "happy"], size=y.shape[0])
        data_processor = DataProcessors.multi_class
        X_train, y_train = data_processor().process_train_data(X, y)
        check_add_bias(X, X_train,test = self._test)        

    def test_multi_class_train_val_data(self, get_multiclass_classification_data):
        X, y = get_multiclass_classification_data
        y = np.random.choice(["hat", "bowl", "junky", "riding", "happy"], size=y.shape[0])
        data_processor = DataProcessors.multi_class
        X_train, X_val, y_train, y_val = data_processor().process_train_val_data(X, y, val_size=0.3)
        check_add_bias(X, X_train, test = self._test)        
        check_add_bias(X, X_val, test = self._test)        
        check_split(X, y, X_train, y_train, X_val, y_val, test=self._test)
        check_one_hot_label_encoder(y_train, test=self._test)
        check_one_hot_label_encoder(y_val, test=self._test)

    def test_multi_class_X_test_data(self, get_multiclass_classification_data):
        X, y = get_multiclass_classification_data
        y = np.random.choice(["hat", "bowl", "junky", "riding", "happy"], size=y.shape[0])
        data_processor = DataProcessors.multi_class
        X_train = data_processor().process_X_test_data(X)
        check_add_bias(X, X_train, test=self._test)

    def test_multi_class_y_test_data(self, get_multiclass_classification_data):
        X, y = get_multiclass_classification_data
        y = np.random.choice(["hat", "bowl", "junky", "riding", "happy"], size=y.shape[0])
        data_processor = DataProcessors.multi_class
        y_test = data_processor().process_y_test_data(y)         
        check_one_hot_label_encoder(y_test, test=self._test)
        
        