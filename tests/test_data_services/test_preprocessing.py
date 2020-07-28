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

from mlstudio.data_services.preprocessing import DataPipeline
from mlstudio.data_services.preprocessing import AbstractDataPipelineStep
from mlstudio.factories.pipeline import PipelineConfigFactory, PipelineSteps
from mlstudio.utils.data_manager import AddBiasTerm

# --------------------------------------------------------------------------  #
#                        TEST PIPELINE STEPS                                  #
# --------------------------------------------------------------------------  #  
@mark.preprocessing
@mark.pipeline_config
class PipelineConfigTests:

    def test_data_pipeline_config(self):
        d = {}
        d['val_size'] = 0.2
        d['shuffle'] = True
        d['stratify'] = True
        d['length'] = 200
        d['width'] = 300
        d['encode_labels'] = True
        d['one_hot_encode_labels'] = True
        d['random_state'] = 55
        config = PipelineConfigFactory(name='DynamicPipelineConfig', config=d)
        settings = config.build()
        assert settings.val_size == d['val_size'], "Configuration attribute error"
        assert settings.shuffle == d['shuffle'], "Configuration attribute error"
        assert settings.stratify == d['stratify'], "Configuration attribute error"
        assert settings.length == d['length'], "Configuration attribute error"
        assert settings.width == d['width'], "Configuration attribute error"
        assert settings.encode_labels == d['encode_labels'], "Configuration attribute error"
        assert settings.one_hot_encode_labels == d['one_hot_encode_labels'], "Configuration attribute error"
        assert settings.random_state == d['random_state'], "Configuration attribute error"
        return settings        
        
@mark.preprocessing
@mark.pipeline_steps
class PipelineStepTests:

    def test_pipeline_step1_add_bias_term(self, get_regression_data):
        X, y = get_regression_data
        config = PipelineConfigTests().test_data_pipeline_config()                
        assert config.shuffle == True, "Config not obtained from prior class."
        step = PipelineSteps.add_bias_term_factory(config=config)
        X_new, y = step(X, y)
        assert X.shape[1] + 1 == X_new.shape[1], "Pipeline step error: input shape not changed" 

    def test_pipeline_step2_shuffle_data(self, get_regression_data):
        X, y = get_regression_data
        config = PipelineConfigTests().test_data_pipeline_config()        
        assert config.shuffle == True, "Config not obtained from prior class."
        step = PipelineSteps.shuffle_factory(config=config)
        X_new, y = step(X, y)
        assert not np.array_equal(X, X_new), "Pipeline step error: shuffle didn't work" 

    def test_pipeline_step3_split_data(self, get_regression_data):
        X, y = get_regression_data
        config = PipelineConfigTests().test_data_pipeline_config()        
        step = PipelineSteps.split_data_factory(config=config)
        X_train, X_val, y_train, y_val = step(X, y)
        assert not np.array_equal(X, X_train), "Pipeline step error: split didn't work" 
        assert X_train.shape[0] == y_train.shape[0] , "Pipeline step error: mismatch lengths " 
        assert X_val.shape[0] == y_val.shape[0] , "Pipeline step error: mismatch lengths " 
        assert X.shape[1] == X_val.shape[1] , "Pipeline step error: mismatch dimensions " 

    def test_pipeline_step4_encode_labels(self, get_regression_data):
        X, y = get_regression_data
        y = np.random.randint(low=10,high=14, size=X.shape[0], dtype='int')
        config = PipelineConfigTests().test_data_pipeline_config()        
        step = PipelineSteps.encode_labels_factory(config=config)
        X, y = step(X, y)
        assert len(np.unique(y)) == 4, "Pipeline step error: check data"
        assert np.all(y) in np.arange(0,4), "Pipeline step error: encoding didn't work" 

    def test_pipeline_step5_one_hot_encode_labels(self, get_regression_data):
        X, y = get_regression_data
        y = np.random.randint(low=0,high=4, size=X.shape[0], dtype='int')
        config = PipelineConfigTests().test_data_pipeline_config()        
        step = PipelineSteps.one_hot_encode_labels_factory(config=config)
        X, y = step(X, y)
        assert len(np.unique(y)) == 2, "Pipeline step error: check data"
        assert y.shape[1] == 4, "Pipeline step error: one hot encoding didn't work." 

@mark.preprocessing
@mark.pipeline
class DataPipelineTests:

    def test_data_pipeline_object(self, get_regression_data):
        X, y = get_regression_data
        y = np.random.randint(low=0,high=4, size=X.shape[0], dtype='int')
        config = PipelineConfigTests().test_data_pipeline_config()                
        pipe = DataPipeline()
        d = {}
        step_classes =  [cls for cls in AbstractDataPipelineStep.__subclasses__()]
        # Instantiate all step classes
        steps = []
        for step in step_classes:
            steps.append(step(config, AddBiasTerm()))

        # Add all steps
        for step in steps:
            pipe.add_step(step)
        # Get all steps
        for step in steps:            
            assert isinstance(pipe.get_step(step.name), AbstractDataPipelineStep), "Pipeline error: Not a step instance."
        # List all steps
        print(pipe.list_steps)
        # Try to add step that already exists:
        for step in steps:
            with pytest.raises(ValueError):
                pipe.add_step(step)
        # Delete all steps
        for step in steps:
            pipe.del_step(step.name)
        # Confirm all deleted
        for step in steps:            
            assert not pipe.get_step(step.name), "Pipeline deletion error"
        # Now add them again and run the pipeline
        for step in steps:
            pipe.add_step(step)
        X_new, y = pipe.run(X, y)
        assert X


        
        








