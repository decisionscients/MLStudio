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
from mlstudio.utils.validation import validate_bool, validate_early_stop
from mlstudio.utils.validation import validate_gradient_check, validate_int
from mlstudio.utils.validation import validate_learning_rate_schedule
from mlstudio.utils.validation import validate_objective, validate_optimizer
from mlstudio.utils.validation import validate_scorer, validate_string
from mlstudio.utils.validation import validate_zero_to_one
# --------------------------------------------------------------------------  #
class LearningRateSchedule(Callback):
    """Base class for learning rate schedules. 
    
    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of iterations

    freq : str
        The unit of time associated with a single iteration. 

    """

    @abstractmethod
    def __init__(self, decay_factor=1, freq='epoch'):    
        super(LearningRateSchedule, self).__init__()
        self.decay_factor = decay_factor
        self.freq = freq                    

    def _default_decay_factor(self):
        """Computes a default decay factor.
        
        The default decay factor is given by:
        .. math:: \gamma=\frac{\alpha}{epochs}         
        
        """
        return self.model.eta / self.model.epochs

    def _validate(self):
        """Performs validation of hyperparameters."""
        validate_zero_to_one(self.decay_factor, 'decay_factor')
        validate_string(param=self.freq, param_name='freq', 
                        valid_values=['epoch', 'batch'])

    @abstractmethod
    def _adjust_learning_rate(self, iteration, logs):
        pass

    @property
    def iterations(self):
        """Computes anticipated number of iterations."""
        if self.freq == 'epoch':
            iterations = self.model.epochs
        else:
            if self.model.batch_size:                
                n_observations = self.model.X_train_.shape[0]
                n_batches_per_epoch = math.ceil(n_observations / \
                    self.model.batch_size)
                iterations = n_batches_per_epoch * self.model.epochs    
            else:
                iterations = self.model.epochs
        return iterations

    def on_train_begin(self, logs=None):
        super(LearningRateSchedule, self).on_train_begin(logs)
        self._eta0 = self.model.learning_rate
        if self.decay_factor is 'optimal':
            self.decay_factor = self._default_decay_factor()

    def on_epoch_begin(self, epoch, logs=None):
        if self.freq == 'epoch':
            self._adjust_learning_rate(iteration=epoch, logs=logs)
    
    def on_batch_begin(self, batch, logs=None):
        if self.freq != 'epoch':
            self._adjust_learning_rate(iteration=batch, logs=logs)
        

# --------------------------------------------------------------------------  #
class StepDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:

    .. math:: \eta_0 \times \gamma^{\text{floor((1+iteration)/decay_steps)}}

    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of iterations

    decay_steps : int (default=1)
        The number of steps between each update

    freq : str
        The unit of time associated with a single iteration. 

    """

    def __init__(self, decay_factor=1, decay_steps=1, freq='epoch'):        
        super(StepDecay, self).__init__(
            decay_factor=decay_factor,
            freq=freq)              
        self.name = "Step Decay Learning Rate Schedule"
        self.decay_steps = decay_steps

     

    def _validate(self):
        """Performs hyperparameter """
        super(StepDecay, self)._validate(decay_factor=self.decay_factor, 
                                         freq=self.freq)
        validate_int(param=self.decay_steps, param_name='decay_steps', minimum=1)

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self._eta0 * np.power(self.decay_factor, math.floor((1+iteration)/self.decay_steps))

# --------------------------------------------------------------------------  #
class TimeDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:

    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot t} 

    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of iterations

    freq : str
        The unit of time associated with a single iteration. 

    """

    def __init__(self, decay_factor=.005, freq='epoch'):        
        super(TimeDecay, self).__init__(
            decay_factor=decay_factor,
            freq=freq)
        self.name = "Time Decay Learning Rate Schedule"              

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self._eta0 / (1 + self.decay_factor * iteration)

# --------------------------------------------------------------------------  #
class SqrtTimeDecay(LearningRateSchedule):
    """ Time decay learning rate schedule as:

    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot \sqrt{t}} 

    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of iterations

    freq : str
        The unit of time associated with a single iteration. 

    """
    def __init__(self, decay_factor=.0005, freq='epoch'):        
        super(SqrtTimeDecay, self).__init__(
            decay_factor=decay_factor,
            freq=freq)              
        self.name = "Sqrt Time Decay Learning Rate Schedule"

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self._eta0 / (1 + self.decay_factor * \
            np.sqrt(iteration))        

# --------------------------------------------------------------------------  #
class ExponentialDecay(LearningRateSchedule):
    """ Exponential decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of iterations

    freq : str
        The unit of time associated with a single iteration. 

    """
    def __init__(self, decay_factor=.002, freq='epoch'):        
        super(ExponentialDecay, self).__init__(
            decay_factor=decay_factor,
            freq=freq)   
        self.name = "Exponential Decay Learning Rate Schedule"

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self._eta0 * np.exp(-self.decay_factor * iteration)

# --------------------------------------------------------------------------  #
class PolynomialDecay(LearningRateSchedule):
    """ Polynomial decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    power : float (default=1)
        The power to which 

    freq : str
        The unit of time associated with a single iteration. 

    """

    def __init__(self, power=1.0, freq='epoch'):
        super(PolynomialDecay, self).__init__(
            freq=freq
        )
        self.name = "Polynomial Decay Learning Rate Schedule"
        self.power = power

    def _validate(self):
        """Performs hyperparameter """
        validate_zero_to_one(self.power)        
        validate_string(param=self.freq, param_name='freq', 
                        valid_values=['epoch', 'batch'])     

    def _adjust_learning_rate(self, iteration, logs):
        decay = (1 - (iteration / float(self.model.epochs))) ** self.power                
        self.model.eta = self._eta0 * decay
# --------------------------------------------------------------------------  #
class ExponentialLearningRate(LearningRateSchedule):
    """ Exponential learning rate schedule as:

    .. math:: \eta_t=(1=\lambda\eta)^{-2t-1}\eta


    Reference : https://arxiv.org/abs/1910.07454

    Parameters
    ----------
    decay_factor : float (default=1) or 'optimal'
        If 'optimal', the decay rate will be computed based upon the 
        learning rate and the anticipated number of iterations

    freq : str
        The unit of time associated with a single iteration. 

    """

    def __init__(self, decay_factor=.005, freq='epoch'):   
        super(ExponentialLearningRate, self).__init__(
            decay_factor=decay_factor,
            freq=freq
        )
        self.name = "Exponential Learning Rate Schedule"

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = np.power((1- self.decay_factor*self._eta0), \
            (-2*iteration-1)) * self._eta0
# --------------------------------------------------------------------------  #
class ExponentialSchedule(LearningRateSchedule):
    """ Exponential decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 \cdot 10^{\frac{-t}{r}}

    Parameters
    ----------
    decay_factor : float
        The factor by which the learning rate is decayed

    decay_steps : int
        The number of steps between each update

    """

    def __init__(self, decay_factor=0.0005, decay_steps=100, freq='epoch'):   
        super(ExponentialSchedule, self).__init__(
            decay_factor=decay_factor,
            freq=freq
        )    
        self.name = "Exponential Schedule Learning Rate Schedule"
        self.decay_steps = decay_steps

    def _validate(self):
        """Performs hyperparameter """
        super(ExponentialSchedule, self)._validate(decay_factor=self.decay_factor,
                                                   freq=self.freq)
        validate_int(param=decay_steps, param_name='decay_steps')

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self._eta0 * np.power(self.decay_factor, \
            (iteration / self.decay_steps))

# --------------------------------------------------------------------------  #
class PowerSchedule(LearningRateSchedule):
    """ Exponential decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 / (1+\frac{t}{r})^{c}

    Parameters
    ----------
    power : float (default=1)
        The factor by which the learning rate is decayed

    decay_steps : int
        The number of steps between each update

    """

    def __init__(self, power=1, decay_steps=1, freq='epoch'):
        super(PowerSchedule, self).__init__(
            freq=freq
        )
        self.name = "Power Schedule Learning Rate Schedule"              
        self.power = power
        self.decay_steps = decay_steps

    def _validate(self):
        """Performs hyperparameter """
        validate_zero_to_one(self.power)        
        validate_int(param=decay_steps, param_name='decay_steps')
        validate_string(param=self.freq, param_name='freq', 
                        valid_values=['epoch', 'batch'])             

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self._eta0 / (1 + iteration/self.decay_steps)**self.power

# --------------------------------------------------------------------------  #
class BottouSchedule(LearningRateSchedule):
    """ Learning rate schedule as described in:

    https://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf

    Parameters
    ----------
    alpha : float (default=0.01)
        The factor by which the learning rate is decayed

    """

    def __init__(self, decay_factor=0.01, freq="epoch"):
        super(BottouSchedule, self).__init__(
            decay_factor=decay_factor,
            freq=freq
        )
        self.name = "Bottou Schedule Learning Rate Schedule"

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self._eta0 * (1 + self._eta0 * \
            self.decay_factor * iteration)**(-1)

# --------------------------------------------------------------------------- #
#                             STABILITY                                       #
# --------------------------------------------------------------------------- #
class Performance(LearningRateSchedule):
    """Decays the learning rate if performance plateaus.

    Parameters
    ----------
    decay_factor : float (default - 0.3)
        The factor by which the learning rate is reduced. 
        math:: \alpha_{new} = \alpha_ * \text{decay_factor}

    metric : str, optional (default='val_score')
        Specifies which statistic to metric for evaluation purposes.

        'train_cost': Training set costs
        'train_score': Training set scores based upon the model's metric parameter
        'val_cost': Validation set costs
        'val_score': Validation set scores based upon the model's metric parameter
        'theta': The parameters of the model
        'gradient': The gradient of the objective function w.r.t. theta

    scorer : Scorer object
        Computes training and validation scores.

    min_lr : float (default=0)
        The learning rate floor to which I will not be reduced.

    epsilon : float, optional (default=0.001)
        The factor by which performance is considered to have improved. For 
        instance, a value of 0.01 means that performance must have improved
        by a factor of 1% to be considered an improvement.

    patience : int, optional (default=5)
        The number of consecutive epochs of non-improvement that would 
        stop training.    

    freq : str
        The unit of time associated with one iteration. This will be
        'epoch' or 'batch'.

    """

    def __init__(self, decay_factor=0.5, metric='cost', min_lr=0,
                 epsilon=1e-3, patience=5, freq='epoch'):
        super(Performance, self).__init__()
        self.name = "Performance Learning Rate Schedule"
        self.decay_factor = decay_factor
        self.metric = metric        
        self.epsilon = epsilon
        self.patience = patience
        self.freq = freq       

    def _validate(self):
        super(Performance, self)._validate(decay_factor=self.decay_factor,
                                           freq=self.freq)
        validate_zero_to_one(param=self.decay_factor, param_name='decay_factor')       
        validate_metric(self.metric)
        validate_zero_to_one(param=min_lr, param_name='min_lr',
                             left='open')
        validate_zero_to_one(param=self.epsilon, param_name='epsilon') 
        validate_int(param=patience, param_name='patience')      

    def on_train_begin(self, logs=None):        
        """Sets key variables at beginning of training.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(Performance, self).on_train_begin(logs)
        self._validate()        
        self._observer = Performance(metric=self.metric, scorer=self.model.scorer, \
            epsilon=self.epsilon, patience=self.patience)    
        self._observer.initialize()        

    def _adjust_learning_rate(self, iteration, logs):
        self.model.eta = self.mode.eta * self.decay_factor

    def on_epoch_begin(self, epoch, logs=None):
        """Determines whether convergence has been achieved.

        Parameters
        ----------
        epoch : int
            The current epoch number

        logs : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  

        Returns
        -------
        Bool if True convergence has been achieved. 

        """
        super(Performance, self).on_epoch_begin(epoch, logs)        
        logs = logs or {}        
        
        if self._observer.evaluate(epoch, logs):            
            if self.model.eta * self.decay_factor > self.min_lr:
                self._adjust_learning_rate(logs)
