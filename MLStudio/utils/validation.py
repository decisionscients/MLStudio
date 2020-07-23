#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : validation.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 14th 2020, 1:33:31 pm                         #
# Last Modified : Thursday, May 14th 2020, 1:35:41 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Functions used to validate the state, parameters or data of an estimator."""
import sys

import numpy as np
import sklearn.utils.validation as skl
# --------------------------------------------------------------------------  #
def check_X(X):
    """Wraps sklearn's check array function."""
    return skl.check_array(array=X, accept_sparse=['csr'], 
                           accept_large_sparse=True)

def check_X_y(X, y):
    return skl.check_X_y(X, y, accept_sparse=['csr'],  
                        accept_large_sparse=True)    

def check_is_fitted(estimator):
    skl.check_is_fitted(estimator)
    
# --------------------------------------------------------------------------  #
def is_valid_array_size(x, lower=1e-10, upper=1e10):
    """Checks whether a vector or matrix norm is within lower and upper bounds.
    
    Parameters
    ----------
    x : array-like
        The data to be checked

    lower : float (default = 1e-10)
        The lower bound vector or matrix norm.

    upper : float (default = 1e10)
        The upper bound vector or matrix norm.

    Returns
    -------
    bools : True if greater than lower and less than upper, False otherwise.
    """
    r = np.linalg.norm(x)
    if r > lower and r < upper:
        return True
    else:
        return False
# --------------------------------------------------------------------------  #
def is_one_hot(x):
    """Returns true if a 2 dimensional matrix is in one-hot encoded format."""
    if np.ndim(x) == 1:
        return False
    try:
        return np.sum(x) / x.shape[0]  == 1
    except:
        return False
# --------------------------------------------------------------------------  #
def is_multilabel(y):
    """Returns true if y is multilabel"""
    y = np.asarray(y)
    if np.ndim(y) == 1:
        return False
    if is_one_hot(y):
        return False
    if y.shape[1] > 1:
        return True        
# --------------------------------------------------------------------------  #
def check_is_fitted(x):
    return x.converged and X.intercept_ and X.coef_

# --------------------------------------------------------------------------  #
def validate_bool(param, param_name=""):        
    if not isinstance(param, bool):
        msg = "{s} must be a boolean value.".format(s=param_name)
        raise TypeError(msg)
    else:
        return True             
     
# --------------------------------------------------------------------------  #
def validate_array_like(param, param_name=""):        
    if not isinstance(param, (np.ndarray, list, tuple, pd.Series)):
        msg = "{s} must be an array-like object.".format(s=param_name)
        raise TypeError(msg)
    else:
        return True             

# --------------------------------------------------------------------------  #        
def validate_gradient_check(param):
    from mlstudio.supervised.algorithms.optimization.observers import debug
    if param is None:
        raise ValueError("The GradientCheck observer from optimization.observers.monitor is required.")
    if not isinstance(param, debug.GradientCheck):
        raise TypeError("The GradientCheck observer from optimization.observers.monitor is required.")
    return True            
# --------------------------------------------------------------------------  #        
def validate_summary(param):
    from mlstudio.supervised.algorithms.optimization.observers import monitor    
    if param is None:
        raise ValueError("The Summary observer from optimization.observers.monitor is required.")
    if not isinstance(param, monitor.Summary):
        raise TypeError("The Summary observer from optimization.observers.monitor is required.")
    return True                
# --------------------------------------------------------------------------  #        
def validate_progress(param):
    from mlstudio.supervised.algorithms.optimization.observers import monitor    
    if param is None:
        raise ValueError("The Progress observer from optimization.observers.monitor is required.")    
    if not isinstance(param, monitor.Progress):
        raise TypeError("The Progress observer from optimization.observers.monitor is required.")
    return True        
# --------------------------------------------------------------------------  #        
def validate_black_box(param):
    from mlstudio.supervised.algorithms.optimization.observers import monitor    
    if param is None:
        raise ValueError("The BlackBox observer from optimization.observers.monitor is required.")    
    if not isinstance(param, monitor.BlackBox):
        raise TypeError("The BlackBox observer from optimization.observers.monitor is required.")
    return True        
# --------------------------------------------------------------------------  #        
def validate_regression_loss(param):
    from mlstudio.supervised.algorithms.optimization.services import loss
    valid_loss_classes = (loss.Quadratic,)
    if not isinstance(param, valid_loss_classes):
        raise TypeError("The loss parameter is not a valid regression loss class.")
    return True
# --------------------------------------------------------------------------  #        
def validate_binary_classification_loss(param):
    from mlstudio.supervised.algorithms.optimization.services import loss
    valid_loss_classes = (loss.CrossEntropy,)
    if not isinstance(param, valid_loss_classes):
        raise TypeError("The loss parameter is not a valid binary classification loss class.")
    return True
# --------------------------------------------------------------------------  #        
def validate_multiclass_classification_loss(param):
    from mlstudio.supervised.algorithms.optimization.services import loss
    valid_loss_classes = (loss.CategoricalCrossEntropy,)
    if not isinstance(param, valid_loss_classes):
        raise TypeError("The loss parameter is not a valid multiclass classification loss class.")
    return True    


# --------------------------------------------------------------------------  #        
def validate_regression_data_processor(param):
    from mlstudio.utils.data_manager import RegressionDataProcessor    
    if not isinstance(param, RegressionDataProcessor):
        raise TypeError("The data_processor parameter is not a valid regression data_processor class.")
    return True
# --------------------------------------------------------------------------  #        
def validate_binary_classification_data_processor(param):
    from mlstudio.utils.data_manager import LogisticRegressionDataProcessor    
    if not isinstance(param, LogisticRegressionDataProcessor):
        raise TypeError("The data_processor parameter is not a valid logistic regression data_processor class.")
    return True    
# --------------------------------------------------------------------------  #        
def validate_multiclass_classification_data_processor(param):
    from mlstudio.utils.data_manager import MulticlassDataProcessor    
    if not isinstance(param, MulticlassDataProcessor):
        raise TypeError("The data_processor parameter is not a valid multiclass classification data_processor class.")
    return True      


# --------------------------------------------------------------------------  #        
def validate_binary_classification_activation(param):
    from mlstudio.supervised.algorithms.optimization.services.activations import Sigmoid    
    if not isinstance(param, Sigmoid):
        raise TypeError("This class requires the Sigmoid activation function.")
    return True    
# --------------------------------------------------------------------------  #        
def validate_multiclass_classification_activation(param):
    from mlstudio.supervised.algorithms.optimization.services.activations import Softmax
    if not isinstance(param, Softmax):
        raise TypeError("This class requires the Softmax activation function.")
    return True      


# --------------------------------------------------------------------------  #        
def validate_observer_list(param):
    from mlstudio.supervised.algorithms.optimization.observers.base import ObserverList    
    if not isinstance(param, ObserverList):
        raise ValueError("This class requires an ObserverList object.")
    return True
# --------------------------------------------------------------------------  #
def validate_observers(param, param_name='observers'):
    from mlstudio.supervised.algorithms.optimization.observers.base import Observer
    if not isinstance(param, dict):
        msg = "The observer parameter must be a dictionary where each entry's \
            key is the name of the observer and the value is the observer object."
        raise TypeError(msg)
    for name, observer in param.items():
        if not isinstance(observer, Observer):
            msg = name + " is not a valid Observer object."
            raise TypeError(msg)
    return True
    
# --------------------------------------------------------------------------  #
def validate_metric(metric):    
    from mlstudio.supervised.performance.base import Metric
    valid_metrics = [cls.__name__ for cls in Metric.__subclasses__()]
    if not isinstance(metric, Metric):
        msg = "{s} is an invalid Metric object. The valid Metric classes include : \
            {v}".format(s=metric, v=str(valid_metrics))
        raise TypeError(msg)
    else:
        return True
# --------------------------------------------------------------------------  #
def validate_activation(activation):    
    from mlstudio.supervised.algorithms.optimization.services.activations import Activation
    valid_activations = [cls.__name__ for cls in Activation.__subclasses__()]
    if not isinstance(activation, Activation):
        msg = "{s} is an invalid Activation object. The valid Activation classes include : \
            {v}".format(s=activation, v=str(valid_activations))
        raise TypeError(msg)
    else:
        return True

# --------------------------------------------------------------------------  #
def validate_objective(objective):    
    from mlstudio.supervised.algorithms.optimization.services.objectives import Objective
    valid_objectives = [cls.__name__ for cls in Objective.__subclasses__()]
    if not isinstance(objective, Objective):
        msg = "{s} is an invalid Objective function object. The valid Objective \
        function classes include : {v}".format(s=objective, v=str(valid_objectives))
        raise TypeError(msg)
    else:
        return True      
# --------------------------------------------------------------------------  #
def validate_gradient_scaler(scaler):        
    from mlstudio.utils.data_manager import GradientScaler
    if not isinstance(scaler, GradientScaler):
        msg = "{s} is an invalid GradientScaler object. The valid Optimizer \
        classes include : {v}".format(s=scaler, v=str(data_manager.GradientScaler.__name__))
        raise TypeError(msg)
    else:
        return True     
# --------------------------------------------------------------------------  #
def validate_optimizer(optimizer):    
    from mlstudio.supervised.algorithms.optimization.services.optimizers import Optimizer
    valid_optimizers = [cls.__name__ for cls in Optimizer.__subclasses__()]
    if not isinstance(optimizer, Optimizer):
        msg = "{s} is an invalid Optimizer object. The valid Optimizer \
        classes include : {v}".format(s=optimizer, v=str(valid_optimizers))
        raise TypeError(msg)
    else:
        return True         
# --------------------------------------------------------------------------  #
def validate_regularizer(regularizer):    
    from mlstudio.supervised.algorithms.optimization.services.regularizers import Regularizer
    valid_regularizers = [cls.__name__ for cls in Regularizer.__subclasses__()]
    if not isinstance(regularizer, Regularizer):
        msg = "{s} is an invalid Regularizer object. The valid Regularizer \
        classes include : {v}".format(s=regularizer, v=str(valid_regularizers))
        raise TypeError(msg)
    else:
        return True   
# --------------------------------------------------------------------------  #
def validate_task(task):    
    from mlstudio.supervised.algorithms.optimization.services.tasks import Task
    valid_tasks = [cls.__name__ for cls in Task.__subclasses__()]
    if not isinstance(task, Task):
        msg = "{s} is an invalid Task object. The valid Task \
        classes include : {v}".format(s=task, v=str(valid_tasks))
        raise TypeError(msg)
    else:
        return True           
# --------------------------------------------------------------------------  #
def validate_metric(metric):
    valid_metrics = ['train_cost', 'train_score', 'val_cost', 'val_score',
                     'gradient_norm']
    if not isinstance(metric, str):
        msg = "The metric parameter must be a string including one of {v}.".\
            format(v=str(valid_metrics))
        raise TypeError(msg)
    elif metric not in valid_metrics:
        msg = "{m} is an invalid metric. The valid metrics include : {v}".\
            format(m=metric,
                   v=str(valid_metrics))
        raise ValueError(msg)
    else:
        return True

# --------------------------------------------------------------------------  #
def validate_string(param, param_name="", valid_values=None):
    if not isinstance(param, str) and not valid_values:
        msg = param_name + " must be a string."
        raise TypeError(msg)
    elif param not in valid_values:
        msg = param_name + " must be in {v}".format(v=str(valid_values))
        raise ValueError(msg)
    else:
        return True        

# --------------------------------------------------------------------------  #
def validate_range(param, param_name="", minimum=0, maximum=np.Inf, left='open', right='open'):        
    if not isinstance(param, (int,float)):
        msg = param_name + " hyperparameter must be numeric."
        raise TypeError(msg)
    elif left == 'open' and right == 'open':
        if param <= minimum or param >= maximum:
            msg = param_name + " hyperparameter must be in (" + str(minimum) + "," + str(maximum) + ")"      
            raise ValueError(msg)
    elif left == 'open' and right != 'open':
        if param <= minimum or param > maximum:
            msg = param_name + " hyperparameter must be in (" + str(minimum) + "," + str(maximum) + "]"      
            raise ValueError(msg)
    elif left != 'open' and right == 'open':
        if param < minimum or param >= maximum:
            msg = param_name + " hyperparameter must be in [" + str(minimum) + "," + str(maximum) + ")"                  
            raise ValueError(msg)
    else:
        if param < minimum or param > maximum:
            msg = param_name + " hyperparameter must be in [" + str(minimum) + "," + str(maximum) + "]"                  
            raise ValueError(msg)
# --------------------------------------------------------------------------  #
def validate_int(param, param_name="", minimum=0, maximum=np.inf, left="open",
                 right='open'):        
    if not isinstance(param, int):
        msg = "{s} must be an integer.".format(s=param_name)
        raise TypeError(msg)     
    else:
        validate_range(param=param, param_name=param_name, minimum=minimum, 
                       maximum=maximum, left=left, right=right)             
# --------------------------------------------------------------------------  #
def validate_zero_to_one(param, param_name = "", left='open', right='open'):
    """Validates a parameter whose values should be between 0 and 1."""
    if not isinstance(param, (int,float)):
        msg = param_name + " hyperparameter must be numeric."
        raise TypeError(msg)
    else:
        validate_range(param=param, param_name=param_name, minimum=0, maximum=1,
                       left=left, right=right)
# --------------------------------------------------------------------------  #
def search_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in search_all_subclasses(c)])  

# --------------------------------------------------------------------------  #
#                           CHECK ESTIMATOR                                   # 
# --------------------------------------------------------------------------  #     
def validate_estimator(estimator):
    """Validates an estimator object."""
    validate_task(estimator.task)
    validate_range(estimator.eta0, param_name="eta0", minimum=0, 
                                maximum=1, left='open', right='close')
    validate_int(estimator.epochs, param_name="epochs", minimum=0,
                            left='open', right='close')                 
    if estimator.batch_size:
        validate_int(estimator.batch_size, param_name="batch_size", 
                                minimum=0, left='open', right='close')            
    if estimator.val_size:
        validate_range(estimator.val_size, param_name="val_size", 
                                minimum=0, maximum=1, left='closed', 
                                right='open')
    validate_optimizer(estimator.optimizer)
    validate_metric(estimator.metric)               
    if estimator.early_stop:
        validate.observers(estimator.early_stop, 
                                        param_name='early_stop')
    if estimator.learning_rate:
        validate.observers(estimator.learning_rate, 
                                        param_name='learning_rate')   

    validate_observer_list(estimator.observer_list)
    validate_black_box(estimator.blackbox)
    validate_progress(estimator.progress)
    validate_summary(estimator.summary)
    validate_bool(param=estimator.check_gradient, param_name='check_gradient')
    validate_gradient_check(estimator.gradient_check)

    if estimator.verbose:
        validate_int(param=estimator.verbose, param_name='verbose',
                        minimum=0, left='open')
    if estimator.random_state:
        validate_int(param=estimator.random_state,
                                param_name='random_state')                                            


    if not isinstance(estimator.eta0, float):
        raise ValueError("The eta0 parameter must be a float in (0,1) object.")   

    if not isinstance(estimator.eta0, float):
        raise ValueError("The eta0 parameter must be a float in (0,1) object.")     
     