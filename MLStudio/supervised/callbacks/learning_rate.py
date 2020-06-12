#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : learning_rate.py                                                  #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Friday, May 15th 2020, 9:48:31 pm                           #
# Last Modified : Friday, May 15th 2020, 9:48:31 pm                           #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Learning rate schedules."""
from abc import ABC, abstractmethod
import math
import numpy as np

from mlstudio.supervised.callbacks.base import Callback
from mlstudio.supervised.core.scorers import MSE
from mlstudio.utils.observers import Performance
from mlstudio.utils.validation import validate_bool, validate_early_stop
from mlstudio.utils.validation import validate_int
from mlstudio.utils.validation import validate_learning_rate_schedule
from mlstudio.utils.validation import validate_objective, validate_optimizer
from mlstudio.utils.validation import validate_scorer, validate_string
from mlstudio.utils.validation import validate_zero_to_one, validate_int
from mlstudio.utils.validation import validate_metric
# --------------------------------------------------------------------------  #
#                      LEARNING RATE SCHEDULE HELPERS                         #
# --------------------------------------------------------------------------  #
def compute_total_batches(estimator):
    """Computes the total number of batches to be processed.

    Parameters
    ----------
    estimator : MLStudio estimator object.
    
    """
    m = estimator.X_train_.shape[0]
    batch_size = estimator.batch_size or m
    return estimator.epochs * math.ceil(m / batch_size)
# --------------------------------------------------------------------------  #
def compute_default_decay_factor(estimator):
    """Computes a default decay factor

     Parameters
    ----------
    estimator : MLStudio estimator object.
    
    """
    return estimator.learning_rate / compute_total_batches(estimator)

# --------------------------------------------------------------------------  #
#                      LEARNING RATE SCHEDULE FUNCTIONS                       #
# --------------------------------------------------------------------------  #
# Functions that perform the learning rate calculation in the callback. This 
# functionality has been pulled from the callback to facilitate unit testing.
# --------------------------------------------------------------------------  #
def constant(attributes, eta0, batch, **kwargs):
    """ Constant learning rate

    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.   

    """    
    return eta0
# --------------------------------------------------------------------------  #
def step_decay(attributes, eta0, batch, **kwargs):
    """ Step decay learning rate schedule as:
    .. math:: \eta_0 \times \gamma^{\text{floor((1+batch)/decay_steps)}}

    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.   

    """    
    decay_factor = attributes.get('decay_factor')
    decay_steps = attributes.get('decay_steps')
    assert decay_factor, "step_decay function requires a decay_factor attribute."
    assert decay_steps, "step_decay function requires a decay_steps attribute."
    return eta0 * np.power(decay_factor, math.floor((1+batch)/decay_steps))

# --------------------------------------------------------------------------  #
def time_decay(attributes, eta0, batch, **kwargs):
    """ Time decay learning rate schedule as:
    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot t} 

    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.        
        
    """
    decay_factor = attributes.get('decay_factor')    
    assert decay_factor, "time_decay function requires a decay_factor attribute."    
    return eta0 / (1 + decay_factor * batch)   

# --------------------------------------------------------------------------  #
def sqrt_time_decay(attributes, eta0, batch, **kwargs):
    """ Square root time decay learning rate schedule as:
    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot \sqrt{t}} 

    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.     

    """
    decay_factor = attributes.get('decay_factor')    
    assert decay_factor, "sqrt_time_decay function requires a decay_factor attribute."    
    return eta0 / (1 + decay_factor * np.sqrt(batch))      


# --------------------------------------------------------------------------  #
def exponential_decay(attributes, eta0, batch, **kwargs):
    """ Exponential decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.       
        
    """
    decay_factor = attributes.get('decay_factor')    
    assert decay_factor, "exponential_decay function requires a decay_factor attribute."    
    return eta0  * np.exp(-decay_factor * batch)    

# --------------------------------------------------------------------------  #
def polynomial_decay(attributes, eta0, batch, **kwargs):
    """ Exponential decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        total_batches : int
            The total number of anticipated batches to be processed.   
        
    """
    power = attributes.get('power')    
    total_batches = kwargs.get('total_batches')
    assert power, "polynomial_decay function requires a power attribute."    
    assert total_batches, "total_batches not sent to polynomial_decay function."
    decay = (1-(batch / float(total_batches))) ** power
    return eta0  * decay
# --------------------------------------------------------------------------  #
def exponential_step_decay(attributes, eta0, batch, **kwargs):
    """ Exponential step decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 \cdot 10^{\frac{-t}{r}}

    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.       

    """
    decay_factor = attributes.get('decay_factor')
    decay_steps = attributes.get('decay_steps')
    assert decay_factor, "decay_factor not sent to exponential_step_decay function."
    assert decay_steps, "decay_steps not sent to exponential_step_decay function."
    return eta0 * np.power(decay_factor, (batch / decay_steps))

# --------------------------------------------------------------------------  #
def power_schedule(attributes, eta0, batch, **kwargs):
    """ Power learning rate schedule as:
    .. math:: \eta_t=\eta_0 / (1+\frac{t}{r})^{c}
    
    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.       

    """
    power = attributes.get('power')
    decay_steps = attributes.get('decay_steps')
    assert power, "power not sent to power_schedule function."
    assert decay_steps, "decay_steps not sent to power_schedule function."
    return eta0 / (1 + batch/decay_steps)**power

# --------------------------------------------------------------------------  #
def bottou_schedule(attributes, eta0, batch, **kwargs):
    """ Learning rate schedule as described in:
    https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf
    
    Parameters
    ----------
    attributes : dict
        Dictionary containing the attributes of the LearningRateSchedule object

    eta0 : float
        The initial learning rate

    batch : int
        The batch number. For batch gradient descent, this is the same
        as the epoch number.

    kwargs : dict
        There are no additional keyword arguments for this function.       

    """
    decay_factor = attributes.get('decay_factor')    
    assert decay_factor, "decay_factor not sent to bottou_schedule function."
    return eta0 * (1 + eta0 * decay_factor * batch)**(-1)

# --------------------------------------------------------------------------  #
#                      LEARNING RATE SCHEDULE FACTORY                         #
# --------------------------------------------------------------------------  #
def get_learning_rate_function(schedule):
    """Returns the learning rate function for a given learning rate schedule class."""
    functions = {'Constant': constant, 'StepDecay': step_decay, 
                 'TimeDecay': time_decay, 
                 'SqrtTimeDecay': sqrt_time_decay, 
                 'ExponentialDecay': exponential_decay,
                 'PolynomialDecay': polynomial_decay,
                 'ExponentialStepDecay': exponential_step_decay,
                 'PowerSchedule': power_schedule,
                 'BottouSchedule': bottou_schedule}

    classname = schedule.__class__.__name__
    return functions[classname]


# --------------------------------------------------------------------------  #
#                      LEARNING RATE SCHEDULE CALLBACKS                       #
# --------------------------------------------------------------------------  #
class LearningRateSchedule(Callback):
    """Base class for learning rate schedules. """

    @abstractmethod
    def __init__(self, min_lr=1e-10):    
        super(LearningRateSchedule, self).__init__()    
        self.min_lr = min_lr        

    def on_train_begin(self, logs=None):
        super(LearningRateSchedule, self).on_train_begin(logs)        
        self._eta0 = self.model.learning_rate
        self._learning_rate_function = get_learning_rate_function(self)
        self._attributes = vars()

    def on_batch_end(self, batch, logs=None):
        super(LearningRateSchedule, self).on_batch_end(batch=batch, logs=logs)        
        self.model.eta = max(self._learning_rate_function(attributes=self._attributes, 
                                     eta0=self._eta0,
                                     batch=batch), self.min_lr)

# --------------------------------------------------------------------------  #
class Constant(LearningRateSchedule):        
    """Constant learning rate schedule."""

    def __init__(self):
        self.name = "Constant Learning Rate"        
        super(Constant, self).__init__()


# --------------------------------------------------------------------------  #
class StepDecay(LearningRateSchedule):
    """ Step decay learning rate schedule as:
    .. math:: \eta_0 \times \gamma^{\text{floor((1+batch)/decay_steps)}}

    Parameters
    ----------
    decay_factor : float (default=0.5)
        The factor by which the learning rate is reduced 
    
    decay_steps : int (default=10)
        The number of steps between each update

    """

    def __init__(self, decay_factor=0.5, decay_steps=10):        
        super(StepDecay, self).__init__()              
        self.name = "Step Decay Learning Rate Schedule"
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps       

# --------------------------------------------------------------------------  #
class TimeDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:
    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot t} 
    
    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of batches

    """

    def __init__(self, decay_factor=.005):        
        super(TimeDecay, self).__init__()
        self.name = "Time Decay Learning Rate Schedule"              
        self.decay_factor = decay_factor

# --------------------------------------------------------------------------  #
class SqrtTimeDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:
    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot \sqrt{t}} 
    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of batches

    """
    def __init__(self, decay_factor=.0005):        
        super(SqrtTimeDecay, self).__init__(
            decay_factor=decay_factor)              
        self.name = "Sqrt Time Decay Learning Rate Schedule"

# --------------------------------------------------------------------------  #
class ExponentialDecay(LearningRateSchedule):
    """ Exponential decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)
    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of batches

    """
    def __init__(self, decay_factor=.002):        
        super(ExponentialDecay, self).__init__(
            decay_factor=decay_factor)   
        self.name = "Exponential Decay Learning Rate Schedule"

# --------------------------------------------------------------------------  #
class PolynomialDecay(LearningRateSchedule):
    """ Polynomial decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    power : float (default=1)
        The power by which the learning rate will be annealed. 

    """

    def __init__(self, power=1.0):
        super(PolynomialDecay, self).__init__()
        self.name = "Polynomial Decay Learning Rate Schedule"
        self.power = power

    def _validate(self):
        """Performs hyperparameter validation"""
        validate_zero_to_one(self.power)        

    def _adjust_learning_rate(self, batch, logs):
        m = self.model.X_train_.shape[0]
        batch_size = self.model.batch_size or m
        batches = math.ceil(m / batch_size) * self.model.epochs
        decay = (1 - (batch / float(batches))) ** self.power                
        self.model.eta = self._eta0 * decay

# --------------------------------------------------------------------------  #
class ExponentialStepDecay(LearningRateSchedule):
    """ Exponential step decay learning rate schedule as:
    .. math:: \eta_t=\eta_0 \cdot 10^{\frac{-t}{r}}
    Parameters
    ----------
    decay_factor : float
        The factor by which the learning rate is decayed
    decay_steps : int
        The number of steps between each update
    """

    def __init__(self, decay_factor=0.0005, decay_steps=100):   
        super(ExponentialStepDecay, self).__init__(decay_factor=decay_factor)    
        self.name = "Exponential Schedule Learning Rate Schedule"
        self.decay_steps = decay_steps

    def _validate(self):
        """Performs hyperparameter """
        super(ExponentialStepDecay, self)._validate()
        validate_int(param=self.decay_steps, param_name='decay_steps')

    def _adjust_learning_rate(self, batch, logs):
        self.model.eta = self._eta0 * np.power(self.decay_factor, \
            (batch / self.decay_steps))

# --------------------------------------------------------------------------  #
class PowerSchedule(LearningRateSchedule):
    """ Power learning rate schedule as:
    .. math:: \eta_t=\eta_0 / (1+\frac{t}{r})^{c}
    Parameters
    ----------
    power : float (default=1.0)
        The factor by which the learning rate is decayed
    decay_steps : int
        The number of steps between each update
    """

    def __init__(self, power=1, decay_steps=1):
        super(PowerSchedule, self).__init__()
        self.name = "Power Schedule Learning Rate Schedule"              
        self.power = power
        self.decay_steps = decay_steps

    def _validate(self):
        """Performs hyperparameter """
        validate_zero_to_one(self.power)        
        validate_int(param=self.decay_steps, param_name='decay_steps')

    def _adjust_learning_rate(self, batch, logs):
        self.model.eta = self._eta0 / (1 + batch/self.decay_steps)**self.power

# --------------------------------------------------------------------------  #
class BottouSchedule(LearningRateSchedule):
    """ Learning rate schedule as described in:
    https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf
    
    Parameters
    ----------
    decay_factor : float (default=0.01)
        The factor by which the learning rate is decayed
    """

    def __init__(self, decay_factor=0.01):
        super(BottouSchedule, self).__init__(decay_factor=decay_factor)
        self.name = "Bottou Schedule Learning Rate Schedule"

    def _adjust_learning_rate(self, batch, logs):
        self.model.eta = self._eta0 * (1 + self._eta0 * \
            self.decay_factor * batch)**(-1)

# --------------------------------------------------------------------------- #
#                             STABILITY                                       #
# --------------------------------------------------------------------------- #
class Improvement(LearningRateSchedule):
    """Decays the learning rate if performance plateaus.
    Parameters
    ----------
    decay_factor : float (default - 0.3)
        The factor by which the learning rate is reduced. 
        math:: \alpha_{new} = \alpha_ * \text{decay_factor}
    metric : str, optional (default='train_cost')
        Specifies which statistic to metric for evaluation purposes.
        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'gradient_norm': The norm of the gradient of the objective function w.r.t. theta
    min_lr : float (default=0)
        The learning rate floor to which I will not be reduced.
    epsilon : float, optional (default=0.001)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.
    patience : int, optional (default=5)
        The number of consecutive batches of non-improvement that would 
        stop training.    
    """

    def __init__(self, decay_factor=0.5, metric='train_cost', min_lr=0,
                 epsilon=1e-4, patience=5):
        super(Improvement, self).__init__(decay_factor=decay_factor)
        self.name = "Improvement Learning Rate Schedule"        
        self.metric = metric        
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.patience = patience        

    def _validate(self):
        super(Improvement, self)._validate()        
        validate_metric(self.metric)
        validate_zero_to_one(param=self.min_lr, param_name='min_lr',
                             left='closed')
        validate_zero_to_one(param=self.epsilon, param_name='epsilon') 
        validate_int(param=self.patience, param_name='patience')      

    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(Improvement, self).on_train_begin(logs)
        self._validate()        
        self._observer = Performance(metric=self.metric, scorer=self.model.scorer, \
            epsilon=self.epsilon, patience=self.patience)    
        self._observer.initialize()        

    def _adjust_learning_rate(self, batch, logs):
        if self.model.eta * self.decay_factor > self.min_lr:
            self.model.eta = self.model.eta * self.decay_factor

    def on_batch_end(self, batch, logs=None):
        """Determines whether convergence has been achieved.
        Parameters
        ----------
        batch : int
            The current batch number
        logs : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  
        """
        super(Improvement, self).on_batch_end(batch, logs)        
        logs = logs or {}        
        
        if self._observer.model_is_stable(batch, logs):            
            self._adjust_learning_rate(batch, logs)
