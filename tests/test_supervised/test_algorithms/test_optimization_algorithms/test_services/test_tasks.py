#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.14                                                            #
# File    : test_activations.py                                               #
# Python  : 3.8.3                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, June 15th 2020, 10:24:08 pm                         #
# Last Modified : Monday, June 15th 2020, 10:24:25 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Test Activation Functions."""
import math
import numpy as np
import pytest
from pytest import mark

from mlstudio.factories.tasks import Task
from mlstudio.supervised.algorithms.optimization.services import activations
from mlstudio.supervised.algorithms.optimization.services import loss
from mlstudio.supervised.algorithms.optimization.services import regularizers
from mlstudio.utils.data_manager import RegressionData
from mlstudio.utils.data_manager import BinaryClassData
from mlstudio.utils.data_manager import MultiClassData
# --------------------------------------------------------------------------  #
@mark.tasks
@mark.linear_regression
class LinearRegressionTaskTests:

    def test_linear_regression_validation(self, get_regression_data):        
        with pytest.raises(TypeError):        
            task = Task.linear_regression_factory(loss="hat")
        with pytest.raises(TypeError):        
            task = Task.linear_regression_factory(loss=loss.CrossEntropy())            
        with pytest.raises(TypeError):        
            task = Task.linear_regression_factory(data_processor="hat")   
        with pytest.raises(TypeError):        
            task = Task.linear_regression_factory(data_processor=BinaryClassData())                        
        with pytest.raises(ValueError):        
            task = Task.linear_regression_factory(loss=\
                loss.Quadratic(regularizers.L1_L2(ratio=2)))                                    
        with pytest.raises(NotImplementedError):        
            task = Task.linear_regression_factory()
            X, y = get_regression_data
            theta = np.random.rand(X.shape[1],)
            task.predict_proba(theta, X)

    def test_linear_regression(self, get_regression_task_package):
        task = Task.linear_regression_factory()
        d = get_regression_task_package        
        # Test loss function
        assert isinstance(task.loss, loss.Quadratic), "LinearRegression loss value not correct"        
        task = Task.linear_regression_factory(loss=loss.Quadratic(regularizer=regularizers.L1(alpha=0.1)))
        assert isinstance(task.loss.regularizer, regularizers.L1), "LinearRegression loss regularizer value not correct"
        assert task.loss.regularizer.alpha == 0.1, "LinearRegression loss regularizer value not correct"
        # Test data processor
        assert isinstance(task.data_processor, RegressionData), "LinearRegression data processor error."
        # Test output / predict
        task = Task.linear_regression_factory()
        y_out = task.compute_output(theta=d['theta'], X=d['X'])
        y_pred = task.predict(theta=d['theta'], X=d['X'])
        assert np.allclose(y_out, y_pred), "LinearRegression output and prediction don't match."
        assert np.allclose(y_pred, d['y_pred']), "LinearRegression prediction error."


@mark.tasks
@mark.logistic_regression
class LogisticRegressionTaskTests:

    def test_logistic_regression_validation(self, get_logistic_regression_data):        
        with pytest.raises(TypeError):        
            task = Task.logistic_regression_factory(loss="hat")
        with pytest.raises(TypeError):        
            task = Task.logistic_regression_factory(loss=loss.CategoricalCrossEntropy())            
        with pytest.raises(TypeError):        
            task = Task.logistic_regression_factory(data_processor="hat")   
        with pytest.raises(TypeError):        
            task = Task.logistic_regression_factory(data_processor=RegressionData())                        
        with pytest.raises(ValueError):        
            task = Task.logistic_regression_factory(loss=loss.Quadratic(regularizer=regularizers.L1_L2(ratio=2)))                                    

    def test_logistic_regression(self, get_logistic_regression_task_package):
        task = Task.logistic_regression_factory()
        d = get_logistic_regression_task_package        
        # Test loss function
        assert isinstance(task.loss, loss.CrossEntropy), "LogisticRegression loss value not correct"        
        task = Task.logistic_regression_factory(loss=loss.CrossEntropy(regularizer=regularizers.L1(alpha=0.1)))
        assert isinstance(task.loss.regularizer, regularizers.L1), "LogisticRegression loss regularizer value not correct"
        assert task.loss.regularizer.alpha == 0.1, "LogisticRegression loss regularizer value not correct"
        # Test data processor
        assert isinstance(task.data_processor, BinaryClassData), "LogisticRegression data processor error."
        # Test output 
        task = Task.logistic_regression_factory()
        y_prob = task.predict_proba(theta=d['theta'], X=d['X'])
        y_pred = task.predict(theta=d['theta'], X=d['X'])
        assert np.allclose(y_pred, d['y_pred']), "LogisticRegression prediction error."
        assert np.allclose(y_prob, d['y_prob']), "LogisticRegression probability prediction error."



@mark.tasks
@mark.multiclass_classification
class MulticlassClassificationTaskTests:

    def test_multiclass_classification_validation(self):        
        with pytest.raises(TypeError):        
            task = Task.multiclass_classification_factory(loss="hat")
        with pytest.raises(TypeError):        
            task = Task.multiclass_classification_factory(loss=loss.Quadratic())            
        with pytest.raises(TypeError):        
            task = Task.multiclass_classification_factory(data_processor="hat")   
        with pytest.raises(TypeError):        
            task = Task.multiclass_classification_factory(data_processor=RegressionDataProcessor())                        
        with pytest.raises(ValueError):        
            task = Task.multiclass_classification_factory(loss=loss.CategoricalCrossEntropy(regularizers.L1_L2(ratio=2)))                                    

    def test_multiclass_classification(self, get_multiclass_classification_task_package):
        task = Task.multiclass_classification_factory()
        d = get_multiclass_classification_task_package        
        # Test loss function
        assert isinstance(task.loss, loss.CategoricalCrossEntropy), "MulticlassClassification loss value not correct"        
        task = Task.multiclass_classification_factory(loss=loss.CategoricalCrossEntropy(regularizer=regularizers.L1(alpha=0.1)))
        assert isinstance(task.loss.regularizer, regularizers.L1), "MulticlassClassification loss regularizer value not correct"
        assert task.loss.regularizer.alpha == 0.1, "MulticlassClassification loss regularizer value not correct"
        # Test data processor
        assert isinstance(task.data_processor, MultiClassData), "MulticlassClassification data processor error."
        # Test output 
        task = Task.multiclass_classification_factory()
        y_prob = task.predict_proba(theta=d['theta'], X=d['X'])
        y_pred = task.predict(theta=d['theta'], X=d['X'])
        assert np.allclose(y_prob, d['y_prob']), "MulticlassClassification probability prediction error."
        assert np.allclose(y_pred, d['y_pred']), "MulticlassClassification prediction error."
