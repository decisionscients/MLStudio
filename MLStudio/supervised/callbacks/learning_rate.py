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
import numpy as np

from mlstudio.supervised.callbacks.base import Callback
# --------------------------------------------------------------------------  #
class Constant(Callback):
    """Constant learning rate schedule

    Parameters
    ----------
    eta0 : float (default=0.01) 
        The fixed learning rate

    """
    def __init__(self, eta0=0.01):
        """Callback class constructor."""        
        self.params = None
        self.model = None
        self.eta0 = eta0

    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        self.model.eta = self.eta0

# --------------------------------------------------------------------------  #
class TimeDecay(Callback):
    """ Time decay learning rate schedule as:

    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot t} 

    Parameters
    ----------
    eta0 : float (default=0.01)
        The initial learning rate

    decay_factor : float
        The factor by which the learning rate is decayed

    """

    def __init__(self, eta0=0.01, decay_factor=None):
        """Callback class constructor."""        
        self.params = None
        self.model = None
        self.eta0 = eta0
        self.decay_factor = decay_factor

    def on_train_begin(self, logs=None):
        """Sets decay factor"""
        if not self.decay_factor:
            self.decay_factor = self.eta0 / self.model.epochs


    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        self.model.eta = self.eta0 / (1 + self.decay_factor * epoch)

# --------------------------------------------------------------------------  #
class SqrtTimeDecay(Callback):
    """ Time decay learning rate schedule as:

    .. math:: \eta_t=\frac{\eta_0}{1+b\cdot \sqrt{t}} 

    Parameters
    ----------
    eta0 : float (default=0.01)
        The initial learning rate

    decay_factor : float
        The factor by which the learning rate is decayed

    """

    def __init__(self, eta0=0.01, decay_factor=None):
        """Callback class constructor."""        
        self.params = None
        self.model = None
        self.eta0 = eta0
        self.decay_factor = decay_factor

    def on_train_begin(self, logs=None):
        """Sets decay factor"""
        if not self.decay_factor:
            self.decay_factor = self.eta0 / self.model.epochs


    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        self.model.eta = self.eta0 / (1 + self.decay_factor * np.sqrt(epoch))        

# --------------------------------------------------------------------------  #
class ExponentialDecay(Callback):
    """ Exponential decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    eta0 : float (default=0.01)
        The initial learning rate

    decay_factor : float
        The factor by which the learning rate is decayed

    """

    def __init__(self, eta0=0.01, decay_factor=None):
        """Callback class constructor."""        
        self.params = None
        self.model = None
        self.eta0 = eta0
        self.decay_factor = decay_factor

    def on_train_begin(self, logs=None):
        """Sets decay factor"""
        if not self.decay_factor:
            self.decay_factor = self.eta0 / self.model.epochs


    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        self.model.eta = self.eta0 * np.exp(-self.decay_factor * epoch)

# --------------------------------------------------------------------------  #
class PolynomialDecay(Callback):
    """ Polynomial decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 \cdot \text{exp}(-b\cdot t)

    Parameters
    ----------
    eta0 : float (default=0.01)
        The initial learning rate

    power : float (default=1)
        The power/exponential of the polynomial

    """

    def __init__(self, eta0=0.01, power=1.0):
        """Callback class constructor."""        
        self.params = None
        self.model = None
        self.eta0 = eta0
        self.power = power

    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """
        decay = (1 - (epoch / float(self.model.epochs))) ** self.power                
        self.model.eta = self.eta0 * decay

# --------------------------------------------------------------------------  #
class ExponentialSchedule(Callback):
    """ Exponential decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 \cdot 10^{\frac{-t}{r}}

    Parameters
    ----------
    eta0 : float (default=0.01)
        The initial learning rate

    decay_factor : float
        The factor by which the learning rate is decayed

    """

    def __init__(self, eta0=0.01):
        """Callback class constructor."""        
        self.params = None
        self.model = None
        self.eta0 = eta0

    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        self.model.eta = self.eta0 * 10**(-epoch / self.model.epochs)

# --------------------------------------------------------------------------  #
class PowerSchedule(Callback):
    """ Exponential decay learning rate schedule as:

    .. math:: \eta_t=\eta_0 (1+\frac{t}{r})^{-c}

    Parameters
    ----------
    eta0 : float (default=0.01)
        The initial learning rate

    power : float (default=1)
        The factor by which the learning rate is decayed

    """

    def __init__(self, eta0=0.01, power=1):
        """Callback class constructor."""        
        self.params = None
        self.model = None
        self.eta0 = eta0
        self.power = power

    def on_epoch_begin(self, epoch, logs=None):
        """Logic executed at the beginning of each epoch.
        
        Parameters
        ----------
        epoch : int
            Current epoch
        
        logs: dict
            Dictionary containing the data, cost, batch size and current weights
        """                
        self.model.eta = self.eta0 * (1 + epoch/self.model.epochs)**(-self.power)