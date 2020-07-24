#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : test_data_management.py                                           #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, May 11th 2020, 8:33:38 pm                           #
# Last Modified : Monday, May 11th 2020, 8:33:38 pm                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Tests data management utilities."""
#%%
import numpy as np
import pytest
from pytest import mark
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification

from mlstudio.utils.data_manager import MinMaxScaler, data_split, GradientScaler
from mlstudio.utils.data_manager import AddBiasTerm, ZeroBiasTerm, unpack_parameters
from mlstudio.factories.data_processor import DataProcessors
# --------------------------------------------------------------------------  #
#                       TEST ADD BIAS TERM TRANSFORMER                        #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.add_bias_term
def test_add_bias_term_np():
    X = np.random.rand(5,5)
    xformer = AddBiasTerm()
    X = xformer.fit_transform(X)
    assert X.shape[1] == 6, "Bias term not added."
    assert np.all(X[:,0] == 1.0), "Column zero not ones."
    # Inverse transform
    X = xformer.inverse_transform(X)
    assert X.shape[1] == 5, "Bias term not removed."
    

@mark.utils
@mark.data_manager
@mark.add_bias_term
def test_add_bias_term_csr():
    X = np.random.rand(5,5)
    X = csr_matrix(X)
    xformer = AddBiasTerm()
    X = xformer.fit_transform(X)
    assert X.shape[1] == 6, "Bias term not added."
    assert np.all(X.toarray()[:,0] == 1.0), "Column zero not ones."   
    # Inverse transform
    X = xformer.inverse_transform(X)
    assert X.shape[1] == 5, "Bias term not removed."    

# --------------------------------------------------------------------------  #
#                       TEST ZERO BIAS TERM TRANSFORMER                       #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.zero_bias_term
def test_zero_bias_term():
    X = np.random.rand(5)
    xformer = ZeroBiasTerm()
    X = xformer.fit_transform(X)    
    assert X[0] == 0.0, "Bias not zeroed out ."
    X = np.random.rand(5,5)
    xformer = ZeroBiasTerm()
    X = xformer.fit_transform(X)    
    assert np.array_equal(X[0,:], np.zeros(shape=X.shape[1])), "Bias not zeroed out ."


# --------------------------------------------------------------------------  #
#                        TEST GRADIENT SCALER                                 #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.gradient_scaler
@mark.gradient_scaler_1d
def test_gradient_scaler_1d():            
    lower_threshold = 1e-10
    upper_threshold = 1e10
    lows = [1e-20, 1e15, 1] 
    highs = [1e-10, 1e20, 5]
    for g in zip(lows, highs):    
        X = np.random.default_rng().uniform(low=g[0], high=g[1], size=20)                
        X_orig_norm = np.linalg.norm(X)        
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        X_new = scaler.fit_transform(X)        
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        X_old = scaler.inverse_transform(X_new)
        X_old_norm = np.linalg.norm(X_old)

        assert np.isclose(X_orig_norm, X_old_norm), \
            "Reverse transform didn't work\
                \nX_orig_norm = {n1}\nX_old_norm={n2}".format(n1=str(X_orig_norm),
                n2=str(X_old_norm))
        
@mark.utils
@mark.data_manager
@mark.gradient_scaler
@mark.gradient_scaler_2d
def test_gradient_scaler_2d():            
    lower_threshold = 1e-10
    upper_threshold = 1e10
    lows = [1e-20, 1e15, 1] 
    highs = [1e-10, 1e20, 5]
    for g in zip(lows, highs):    
        X = np.random.default_rng().uniform(low=g[0], high=g[1], size=(20,4))                
        X_orig_norm = (np.linalg.norm(X))                
        scaler = GradientScaler(lower_threshold=lower_threshold, 
                                upper_threshold=upper_threshold)                                        
        X_new = scaler.fit_transform(X)        
        X_new_norm = np.linalg.norm(X_new)
        assert X_new_norm>=lower_threshold and \
               X_new_norm<=upper_threshold, \
                   "Scaling didn't work. X_new_norm = {n}".format(
                   n=str(X_new_norm))        
        X_old = scaler.inverse_transform(X_new)
        X_old_norm = np.linalg.norm(X_old)

        assert np.allclose(X_orig_norm, X_old_norm), \
            "Reverse transform didn't work\
                \nX_orig_norm = {n1}\nX_old_norm={n2}".format(n1=str(X_orig_norm),
                n2=str(X_old_norm))

# --------------------------------------------------------------------------  #
#                       TEST MINMAX SCALER                                    #
# --------------------------------------------------------------------------  #
@mark.utils
@mark.data_manager
@mark.minmax
def test_minmax_scaler():
    x = np.array([[0,0,22],
                [0,1,17],
                [0,1,2]], dtype=float)
    x_new = np.array([[0,0,1],
                    [0,1,15/20],
                    [0,1,0]], dtype=float)
    scaler = MinMaxScaler()
    x_t = scaler.fit_transform(x)
    assert np.array_equal(x_new, x_t), "Minmax scaler not working"    
# --------------------------------------------------------------------------  #
#                        TEST DATA SPLIT                                      #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_manager
@mark.data_split  
def test_data_split():
    X, y = make_classification(n_classes=4, n_informative=3)
    X_train, X_test, y_train, y_test = data_split(X,y, stratify=True)
    n_train = y_train.shape[0]
    n_test = y_test.shape[0]
    train_values, train_counts = np.unique(y_train, return_counts=True)
    test_values, test_counts = np.unique(y_test, return_counts=True)
    train_proportions = train_counts / n_train
    test_proportions = test_counts / n_test
    assert np.allclose(train_proportions, test_proportions, rtol=0.1), "Data split stratification problem "

# --------------------------------------------------------------------------  #
#                        TEST UNPACK PARAMETERS                               #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.unpack_parameters
def test_unpack_parameters():
    X = np.random.rand(5)
    bias, weights = unpack_parameters(X)
    assert bias == X[0], "Bias not unpacked correctly"
    assert np.array_equal(weights, X[1:]), "weights not unpacked correctly"
    X = np.random.rand(5,3)
    bias, weights = unpack_parameters(X)
    assert bias.shape == (3,), "Bias not unpacked correctly"
    assert np.array_equal(bias, X[0,:]), "Bias not unpacked correctly "
    assert weights.shape == (4,3), "Weights not unpacked correctly"
    assert np.array_equal(weights, X[1:,:]), "Weights not unpacked correctly"


# --------------------------------------------------------------------------  #
#                        TEST DATA PROCESSORS                                 #
# --------------------------------------------------------------------------  #  
@mark.utils
@mark.data_processors
@mark.regression_data
class DataProcessorTests:

    _data_processor_types = {'Multiclass': 'Nominal', 'Binaryclass': 'Binary',
                            'Regression': 'Continuous'}

    def _evaluate_original_metadata(self, X, y, data, processor):
        assert len(data['original']['metadata']['Features']) == X.shape[1], processor + " metadata: features error."
        assert data['original']['metadata']['Num Features'] == X.shape[1], processor + " metadata: features error."
        assert data['original']['metadata']['Num Observations'] == X.shape[0], processor + " metadata: observations error."
        assert data['original']['metadata']['Size'] > 100, processor + " metadata: size error."
        assert data['original']['metadata']['Object Class X'] == 'ndarray', processor + " metadata: class error."
        assert data['original']['metadata']['Object Class y'] == 'ndarray', processor + " metadata: class error."
        assert data['original']['metadata']['Target Class'] == self._data_processor_types[processor], processor + " metadata: target type error."
        if processor != "Regression":
            assert len(data['original']['metadata']['Classes']) == data['original']['metadata']['Num Classes'], processor + " metadata: target class error."
        

    def _evaluate_training_metadata(self, X, y, data, processor):
        assert len(data['train']['metadata']['Features']) == X.shape[1]+1, processor + " metadata: features error."
        assert data['train']['metadata']['Num Features'] == X.shape[1]+1, processor + " metadata: features error."
        assert data['train']['metadata']['Num Observations'] == X.shape[0], processor + " metadata: observations error."
        assert data['train']['metadata']['Size'] > 100, processor + " metadata: size error."
        assert data['train']['metadata']['Object Class X'] == 'ndarray', processor + " metadata: class error."
        assert data['train']['metadata']['Object Class y'] == 'ndarray', processor + " metadata: class error."
        if processor != 'Multiclass':
            assert data['train']['metadata']['Target Class'] == self._data_processor_types[processor], processor + " metadata: target type error."
        if processor != "Regression":
            assert len(data['train']['metadata']['Classes']) == data['original']['metadata']['Num Classes'], processor + " metadata: target class error."
        

    def _evaluate_training_validation_metadata(self, X, y, data, processor):
        assert len(data['train']['metadata']['Features']) == X.shape[1]+1, processor + " metadata: features error."
        assert data['train']['metadata']['Num Features'] == X.shape[1]+1, processor + " metadata: features error."
        assert data['train']['metadata']['Num Observations'] < X.shape[0], processor + " metadata: observations error."
        assert data['train']['metadata']['Size'] > 100, processor + " metadata: size error."
        assert data['train']['metadata']['Object Class X'] == 'ndarray', processor + " metadata: class error."
        assert data['train']['metadata']['Object Class y'] == 'ndarray', processor + " metadata: class error."
        assert data['train']['metadata']['Target Class'] == self._data_processor_types[processor], processor + " metadata: target type error."
        if processor != "Regression":
            assert len(data['train']['metadata']['Classes']) == data['train']['metadata']['Num Classes'], processor + " metadata: target class error."
        

        assert len(data['validation']['metadata']['Features']) == X.shape[1]+1, processor + " metadata: features error."
        assert data['validation']['metadata']['Num Features'] == X.shape[1]+1, processor + " metadata: features error."
        assert data['validation']['metadata']['Num Observations'] < X.shape[0], processor + " metadata: observations error."
        assert data['validation']['metadata']['Size'] > 100, processor + " metadata: size error."
        assert data['validation']['metadata']['Object Class X'] == 'ndarray', processor + " metadata: class error."
        assert data['validation']['metadata']['Object Class y'] == 'ndarray', processor + " metadata: class error."
        assert data['validation']['metadata']['Target Class'] == self._data_processor_types[processor], processor + " metadata: target type error."
        if processor != "Regression":
            assert len(data['validation']['metadata']['Classes']) == data['validation']['metadata']['Num Classes'], processor + " metadata: target class error."
        

    def _evaluate_original_data(self, X, y, data, processor):
        assert np.array_equal(data['original']['X'], X), processor + " data: original X changed."
        assert np.array_equal(data['original']['y'], y), processor + " data: original y changed."    

    def _assert_arrays_not_equal(self, a1, a2, dataset1, dataset2, processor):
        assert not np.array_equal(a1, a2), processor + ": " + dataset1 + " and " + dataset2 + "should NOT be equal."
    def _assert_arrays_equal(self, a1, a2, dataset1, dataset2, processor):
        assert np.array_equal(a1, a2), processor + ": " + dataset1 + " and " + dataset2 + "should be equal."



    def _evaluate_training_data_no_shuffle(self, X, y, data, processor):
        # Evaluate X
        self._assert_arrays_not_equal(data['train']['X'], X, 'train_X', 'original_X', processor)
        assert data['train']['X'].shape[0] == X.shape[0], processor + " train_X shape error."
        assert data['train']['X'].shape[1] == X.shape[1]+1, processor + " train_X shape error."
        self._assert_arrays_equal(data['train']['X'][:,1:], X, 'train_y', 'original_y', processor)
        
        # Evaluate y
        if processor == "Regression":
            self._assert_arrays_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
        
        # Evaluate binary classification training data
        elif processor == "Binaryclass":
            # If integer, training and original targets should be the same.
            if data['original']['metadata']['Target Type'] == "Integer":
                self._assert_arrays_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
            else:
            # If not integer, training should have been encoded to integer, so different from original.
                self._assert_arrays_not_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
                assert np.issubdtype(data['train']['y'].dtype, np.number), processor + " data not encoded to numeric."
            assert len(np.unique(data['train']['y'])) == 2, processor + " does not include binary data"

        # Evaluate multiclass classification training data.    
        else:
            assert data['train']['y'].shape[0] == y.shape[0], processor + " Target length not correct."
            assert data['train']['y'].shape[1] == len(np.unique(y)), processor + " Target data not one-hot encoded correctly."
            assert np.sum(data['train']['y']) == y.shape[0], processor + " Target data one hot encoded data doesn't add to m."
            assert len(np.unique(data['train']['y'])) == 2, processor + " Target data not one-hot vector"
            assert np.issubdtype(data['train']['y'].dtype, np.number), processor + " Target data not encoded to numeric."


    def _evaluate_training_data_shuffle(self, X, y, data, processor):
        # Evaluate X
        self._assert_arrays_not_equal(data['train']['X'], X, 'train_X', 'original_X', processor)
        assert data['train']['X'].shape[0] == X.shape[0], processor + " train_X shape error."
        assert data['train']['X'].shape[1] == X.shape[1]+1, processor + " train_X shape error."
        self._assert_arrays_not_equal(data['train']['X'][:,1:], X, 'train_y', 'original_y', processor)
        
        # Evaluate y
        if processor == "Regression":
            self._assert_arrays_not_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
        
        # Evaluate binary classification training data
        elif processor == "Binaryclass":
            # Arrays will be different due to shuffling
            if data['original']['metadata']['Target Type'] == "Integer":
                self._assert_arrays_not_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
            else:
            # If not integer, training should have been encoded to integer, so different from original.
                self._assert_arrays_not_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
                assert np.issubdtype(data['train']['y'].dtype, np.number), processor + " data not encoded to numeric."
            assert len(np.unique(data['train']['y'])) == 2, processor + " does not include binary data"

        # Evaluate multiclass classification training data.    
        else:
            assert data['train']['y'].shape[0] == y.shape[0], processor + " Target length not correct."
            assert data['train']['y'].shape[1] == len(np.unique(y)), processor + " Target data not one-hot encoded correctly."
            assert np.sum(data['train']['y']) == y.shape[0], processor + " Target data one hot encoded data doesn't add to m."
            if processor == "Binaryclass":
                assert len(np.unique(data['train']['y'])) == 2, processor + " Target data has too few unique values"
            else:
                assert data['train']['y'].shape[1] > 2, processor + " Target data has too few unique values"
            assert np.issubdtype(data['train']['y'].dtype, np.number), processor + " Target data not encoded to numeric."




    def _evaluate_training_validation_data_no_shuffle(self, X, y, data, processor):
        # Evaluate X_y
        assert data['train']['X'].shape[0] == data['train']['y'].shape[0], processor + " X_y mismatch on training set " 
        assert data['validation']['X'].shape[0] == data['validation']['y'].shape[0], processor + " X_y mismatch on validation set " 
        # Evaluate X
        self._assert_arrays_not_equal(data['train']['X'], X, 'train_X', 'original_X', processor)
        self._assert_arrays_not_equal(data['train']['X'], data['validation']['X'], 'train_X', 'validation_X', processor)        
        assert data['train']['X'].shape[0] > data['validation']['X'].shape[0], processor + " validation set larger than training."
        assert data['train']['X'].shape[0] < X.shape[0], processor + " training set not smaller than original data."                
        assert data['train']['X'].shape[1] == X.shape[1]+1, processor + " train_X shape error."
        assert data['validation']['X'].shape[1] == X.shape[1]+1, processor + " validation_X shape error."
        assert data['train']['X'].shape[1] == data['validation']['X'].shape[1], processor + " train and validation dimension mismatch."
        assert data['train']['X'].shape[0] + data['validation']['X'].shape[0] <= X.shape[0], processor + " train and validation sets larger than original" 
        self._assert_arrays_not_equal(data['train']['X'][:,1:], X, 'train_y', 'original_y', processor)
        
        # Evaluate y
        if processor == "Regression":
            self._assert_arrays_not_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
            self._assert_arrays_not_equal(data['train']['y'], data['validation']['y'], 'train_y', 'validation_y', processor)
            assert np.issubdtype(data['train']['y'].dtype, np.number), processor + " Training target data numeric."
            assert np.issubdtype(data['validation']['y'].dtype, np.number), processor + " Validation target not numeric."
        # Evaluate binary classification training and validation data
        elif processor == "Binaryclass":
            self._assert_arrays_not_equal(data['train']['y'], y, 'train_y', 'original_y', processor)
            self._assert_arrays_not_equal(data['train']['y'], data['validation']['y'], 'train_y', 'validation_y', processor)
            assert np.issubdtype(data['train']['y'].dtype, np.number), processor + " training data not encoded to numeric."
            assert np.issubdtype(data['validation']['y'].dtype, np.number), processor + " validation data not encoded to numeric."
            assert len(np.unique(data['train']['y'])) == 2, processor + " training does not include binary data"
            assert len(np.unique(data['validation']['y'])) == 2, processor + " validation does not include binary data"

        # Evaluate multiclass classification training and validation data.    
        else:            
            assert data['train']['y'].shape[1] == len(np.unique(y)), processor + " Training target data not one-hot encoded correctly."
            assert data['validation']['y'].shape[1] == len(np.unique(y)), processor + " Validation target data not one-hot encoded correctly."            
            assert np.sum(data['train']['y']) == data['train']['y'].shape[0], processor + " Training target data one hot encoded data doesn't add to m."
            assert np.sum(data['validation']['y']) == data['validation']['y'].shape[0], processor + " Validation target data one hot encoded data doesn't add to m."
            assert data['train']['y'].shape[1] > 2, processor + " Training target data has too few unique values"
            assert np.issubdtype(data['train']['y'].dtype, np.number), processor + " Training target data not encoded to numeric."
            assert data['validation']['y'].shape[1] > 2, processor + " Validation target data has too few unique values"
            assert np.issubdtype(data['validation']['y'].dtype, np.number), processor + " Validation target data not encoded to numeric."            

    def _evaluate_test_data_no_shuffle(self, X, y, data, processor):
        # Evaluate X
        self._assert_arrays_not_equal(data['test']['X'], X, 'test_X', 'original_X', processor)
        assert data['test']['X'].shape[0] == X.shape[0], processor + " test_X shape error."
        assert data['test']['X'].shape[1] == X.shape[1]+1, processor + " test_X shape error."
        self._assert_arrays_equal(data['test']['X'][:,1:], X, 'test_y', 'original_y', processor)        
        
        # Evaluate y
        if processor == "Regression":
            self._assert_arrays_equal(data['test']['y'], y, 'test_y', 'original_y', processor)
        
        # Evaluate binary classification testing data
        elif processor == "Binaryclass":
            # If integer, testing and original targets should be the same.
            if data['original']['metadata']['Target Type'] == "Integer":
                self._assert_arrays_equal(data['test']['y'], y, 'test_y', 'original_y', processor)
        else:
            assert data['test']['y'].shape[1] > 2, processor + " does not include multiclass data"

         
    @mark.regression_data
    def test_regression_data_processor(self, get_regression_data):
        processor = 'Regression'
        X, y = get_regression_data
        dp = DataProcessors.regression()        
        # Training no shuffle        
        data = dp.fit_transform(X, y, dataset='train').copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_data_no_shuffle(X, y, data, processor)
        self._evaluate_training_metadata(X, y, data, processor=processor)        
        
        # Training w/ shuffle
        data = dp.fit_transform(X, y, dataset='train', shuffle=True).copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_data_shuffle(X, y, data, processor)
        self._evaluate_training_metadata(X, y, data, processor=processor)        

        # Training and validation
        data = dp.fit_transform(X, y, dataset='train', val_size=0.3).copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_validation_data_no_shuffle(X, y, data, processor)
        self._evaluate_training_validation_metadata(X, y, data, processor=processor)        

        # Test data
        data = dp.fit_transform(X, y, dataset='test').copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_test_data_no_shuffle(X, y, data, processor)


    @mark.binary_classification_data
    def test_binary_classification_data_processor(self, get_logistic_regression_data_categorical):
        processor = 'Binaryclass'
        X, y = get_logistic_regression_data_categorical
        dp = DataProcessors.binary_classification()        
        # Training no shuffle        
        data = dp.fit_transform(X, y, dataset='train').copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_data_no_shuffle(X, y, data, processor)
        self._evaluate_training_metadata(X, y, data, processor=processor)        
        
        # Training w/ shuffle
        data = dp.fit_transform(X, y, dataset='train', shuffle=True).copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_data_shuffle(X, y, data, processor)
        self._evaluate_training_metadata(X, y, data, processor=processor)        

        # Training and validation
        data = dp.fit_transform(X, y, dataset='train', val_size=0.3).copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_validation_data_no_shuffle(X, y, data, processor)
        self._evaluate_training_validation_metadata(X, y, data, processor=processor)        

        # Test data
        data = dp.fit_transform(X, y, dataset='test').copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_test_data_no_shuffle(X, y, data, processor)


    @mark.multiclass_classification_data
    def test_multiclass_classification_data_processor(self, get_multiclass_classification_data_categorical):
        processor = 'Multiclass'
        X, y = get_multiclass_classification_data_categorical
        dp = DataProcessors.multiclass_classification()        
        # Training no shuffle        
        data = dp.fit_transform(X, y, dataset='train').copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_data_no_shuffle(X, y, data, processor)
        self._evaluate_training_metadata(X, y, data, processor=processor)        
        
        # Training w/ shuffle
        data = dp.fit_transform(X, y, dataset='train', shuffle=True).copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_data_shuffle(X, y, data, processor)
        self._evaluate_training_metadata(X, y, data, processor=processor)        

        # Training and validation
        data = dp.fit_transform(X, y, dataset='train', val_size=0.3).copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_training_validation_data_no_shuffle(X, y, data, processor)
        self._evaluate_training_validation_metadata(X, y, data, processor=processor)        

        # Test data
        data = dp.fit_transform(X, y, dataset='test').copy()
        self._evaluate_original_data(X, y, data, processor=processor)
        self._evaluate_original_metadata(X, y, data, processor=processor)
        self._evaluate_test_data_no_shuffle(X, y, data, processor)
