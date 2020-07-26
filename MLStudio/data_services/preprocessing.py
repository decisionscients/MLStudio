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

import numpy as np
from numpy.random import RandomState
from sklearn.base import TransformerMixin, BaseEstimator


from mlstudio.utils.validation import check_X_y, check_X
# --------------------------------------------------------------------------- #
class AbstractPipelineConfig(BaseEstimator):
    """Defines interface for pipeline configuration object.
    
    Parameters
    ----------
    val_size : float in [0,1) or None
        The proportion of the training set to allocate to cross-validation

    shuffle : Bool (Default=False)
        Specifies whether data shuffling should take place.

    stratify : Bool (Default=False)
        Specifies whether stratified data splitting should take place.

    encode_labels : Bool (Default=False)
        Specifies whether 1-of-k class encoding of target labels should take place.

    random_state : int
        Seed for reproducibility of pseudo-randomization
    """

    def __init__(self, val_size=None, shuffle=False, stratify=False, 
                 encode_labels=False, random_state=None):

        self._val_size = val_size
        self._shuffle = shuffle
        self._stratify = stratify
        self._encode_labels = encode_labels
        self._one_hot_encode_labels = one_hot_encode_labels
        self._random_state = random_state

    @property
    def val_size(self):
        return self._val_size

    @property
    def shuffle(self):
        return self._shuffle

    @property
    def stratify(self):
        return self._stratify

    @property
    def encode_labels(self):
        return self._encode_labels

    @property
    def one_hot_encode_labels(self):
        return self._one_hot_encode_labels

    @property
    def random_state(self):
        return self._random_state
        
# --------------------------------------------------------------------------- #
class AbstractDataPipeline(BaseEstimator):
    """Abstract base class for all pipeline subclasses."""

    def __init__(self):
        self._name = "abstract_data_pipeline"
        self._steps = OrderedDict()
        self._transformers = OrderedDict()

    @property
    def name(self):
        return self._name

    def get_step(self, step_name):
        """Gets the step referenced by the given 'step_name' parameter."""
        return self._steps[step_name]

    def add_step(self, step):
        """Adds a step to the pipeline."""
        self._steps[step.name] = step
        return self

    def del_step(self, step_name):
        """Removes a step from the pipeline."""
        try:
            del self._steps[step_name]
        except:
            msg = "Step named {s} did not exist in the pipeline.".format(s=step_name)
            warnings.warn(msg, category=UserWarning)
        return self

    def list_steps(self):
        """Lists the steps by name."""
        for k, v in self._steps.items():
            print(v.name)
        return self

    def run(self, X, y=None):
        """Executes the pipeline."""
        for name, step in self._steps.items():
            X, y = step(X, y)
        return X, y

# --------------------------------------------------------------------------- #
class DataPipeline(AbstractDataPipeline):
    """Data pipeline subclasses."""

    def __init__(self):
        from mlstudio.factories.pipeline import Pipeline
        super(DataPipeline, self).__init__()
        self._name = "data_pipeline"


# --------------------------------------------------------------------------- #
class AbstractDataPipelineStep(BaseEstimator):
    """Defines the interface for pipeline steps."""

    def __init__(self, config, transformer=None):
        self._name = "abstract_data_pipeline_step"
        self._config = config
        self._transformer = transformer
        from mlstudio.factories.pipeline import PipelineSteps

    @property
    def name(self):
        return self._name
    
    @property
    def config(self):
        return self._config

    @property
    def transformer(self):
        return self._transformer        

    @abstractmethod
    def __call__(self, X, y=None):        
        pass
        
# --------------------------------------------------------------------------- #
class AddBiasDataPipelineStep(AbstractDataPipelineStep):
    """Step that adds bias term to input data."""        

    def __call__(self, X, y=None):
        """Adds bias term to input X"""        
        X =  self._transformer.fit_transform(X)        
        return X, y

# --------------------------------------------------------------------------- #
class ShuffleDataPipelineStep(AbstractDataPipelineStep):
    """Step that shuffles the data."""        

    def __call__(self, X, y=None):
        """Shuffles data"""
        
        if self._config.shuffle:
            X, y = self._transformer.fit_transform(X=X, y=y, 
                            random_state=self._config.random_state)
        return X, y

# --------------------------------------------------------------------------- #
class SplitDataPipelineStep(AbstractDataPipelineStep):
    """Step that splits the data."""        

    def __call__(self, X, y=None):
        """Split the data"""
        X_train, X_val, y_train, y_val = self._transformer.fit_transform(X, y, 
                          test_size=self._config.val_size, 
                          stratify=self._config.stratify, 
                          random_state=self._config.random_state)
        return X_train, X_val, y_train, y_val

# --------------------------------------------------------------------------- #
class EncodeLabelsDataPipelineStep(AbstractDataPipelineStep):
    """Step that performs 1-of-k binary encoding of labels."""        

    def __call__(self, X, y=None):
        """Performs label encoding."""
        if self._transformer.is_fitted:
            y = self._transformer.transform(y)
        else:
            y = self._transformer.fit_transform(y)
        return X, y

# --------------------------------------------------------------------------- #
class OneHotEncodeLabelsDataPipelineStep(AbstractDataPipelineStep):
    """Step that performs 1-of-k binary encoding of labels."""        

    def __call__(self, X, y=None):
        """Performs one hot label encoding."""
        if self._transformer.is_fitted:
            y = self._transformer.transform(y)
        else:
            y = self._transformer.fit_transform(y)
        return X, y
        

# --------------------------------------------------------------------------- #
class AbstractDataPipelineBuilder(BaseEstimator):
    """Defines the ABC for data pipeline builders."""

    def __init__(self, config):
        self._config = config    

    def reset(self):
        self._pipeline = Pipeline.factory()

    @property
    def config(self):
        return self._config

    @abstractmethod
    def add_bias_term_step(self):        
        pass

    @abstractmethod
    def shuffle_data_step(self):
        pass   

    @abstractmethod
    def split_data_step(self):
        pass 

    @abstractmethod
    def encode_labels_step(self):
        pass    

    @abstractmethod
    def one_hot_encode_labels_step(self):
        pass        

    def pipeline(self):
        pipeline = self._pipeline
        self.reset()
        return pipeline
# --------------------------------------------------------------------------- #
class RegressionDataPipelineBuilder(AbstractDataPipelineBuilder):
    """Builds data pipelines for regression problems."""
    
    def add_bias_term_step(self):
        """Adds step that creates a bias term on the input matrix."""
        step = PipelineSteps.add_bias_term_factory(config=self._config)
        self._pipeline.add_step(step)

    def shuffle_data_step(self):
        """Adds step that shuffles the data."""
        step = PipelineSteps.shuffle_factory(config=self._config)
        self._pipeline.add_step(step)

    def split_data_step(self):
        """Adds step that splits the data."""
        step = PipelineSteps.split_data_factory(config=self._config)
        self._pipeline.add_step(step)

    def one_hot_encode_labels_step(self):
        """Adds step to encode targets to (0,k). k=n_classes"""
        pass

    def encode_labels_step(self):
        """Adds step that encodes the target data."""
        pass

# --------------------------------------------------------------------------- #
class BinaryClassDataPipelineBuilder(AbstractDataPipelineBuilder):
    """Builds data pipelines for binary classification problems."""
    
    def add_bias_term_step(self):
        """Adds step that creates a bias term on the input matrix."""
        step = PipelineSteps.add_bias_term_factory(config=self._config)
        self._pipeline.add_step(step)

    def shuffle_data_step(self):
        """Adds step that shuffles the data."""
        step = PipelineSteps.shuffle_factory(config=self._config)
        self._pipeline.add_step(step)

    def split_data_step(self):
        """Adds step that splits the data."""
        step = PipelineSteps.split_data_factory(config=self._config)
        self._pipeline.add_step(step)

    def one_hot_encode_labels_step(self):
        """Adds step to encode targets to (0,k). k=n_classes"""
        pass

    def encode_labels_step(self):
        """Adds step that encodes the target data."""
        pass

# --------------------------------------------------------------------------- #
class MultiClassDataPipelineBuilder(AbstractDataPipelineBuilder):
    """Builds data pipelines for multiclass classification problems."""
    
    def add_bias_term_step(self):
        """Adds step that creates a bias term on the input matrix."""
        step = PipelineSteps.add_bias_term_factory(config=self._config)
        self._pipeline.add_step(step)

    def shuffle_data_step(self):
        """Adds step that shuffles the data."""
        step = PipelineSteps.shuffle_factory(config=self._config)
        self._pipeline.add_step(step)

    def split_data_step(self):
        """Adds step that splits the data."""
        step = PipelineSteps.split_data_factory(config=self._config)
        self._pipeline.add_step(step)

    def one_hot_encode_labels_step(self):
        """Adds step to encode targets to (0,k). k=n_classes"""
        step = PipelineSteps.one_hot_encode_labels_factory(config=self._config)
        self._pipeline.add_step(step)

    def encode_labels_step(self):
        """Adds step that encodes the target data."""
        step = PipelineSteps.encode_labels_factory(config=self._config)
        self._pipeline.add_step(step)

# --------------------------------------------------------------------------- #
class DataPipelineDirector(BaseEstimator):
    """Responsible for executign the pipeline building steps in a specific order."""

    def __init__(self, config):        
        self._config = config

    @property
    def builder(self):
        return self._builder

    @property
    def config(self):
        return self._config

    @builder.setter
    def builder(self, builder):
        self._builder = builder

    def build_training_pipeline(self, X, y=None):
        self._builder.add_bias_term_step()
        self._builder.shuffle_data_step()
        self._builder.split_data_step()
        self._builder.one_hot_encode_labels_step()        
        self._builder.encode_labels_step()        
        return self._builder.pipeline()        

    def build_predict_pipeline(self, X, y=None):
        self._builder.add_bias_term_step()
        return self._builder.pipeline()

    def build_score_pipeline(self, X, y=None):
        self._builder.one_hot_encode_labels_step()        
        self._builder.encode_labels_step()                        
        return self._builder.pipeline()
    
