#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : objectives.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Tuesday, May 19th 2020, 1:06:30 pm                          #
# Last Modified : Tuesday, May 19th 2020, 1:06:31 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Objective functions and their gradients."""
from abc import ABC, abstractmethod

import autograd.numpy as np
from autograd import grad
import numpy as np
from sklearn.base import BaseEstimator
# --------------------------------------------------------------------------  #
class Objective(ABC, BaseEstimator):
    """Base class for objective functions."""

    @property
    def density(self):
        """The linspace density per point of difference between min and max.""" 
        return 2

    @abstractmethod
    @property
    def start(self):
        """Returns a good start point."""
        pass

    @abstractmethod
    @property
    def minimum(self):
        """Returns a good start point."""
        pass    

    @abstractmethod 
    @property
    def _range(self):
        """Returns x and y ranges for plotting."""
        pass

    def mesh(self):
        """Returns the mesh grid for the function."""       
        x, y = self._range
        density_x = (x['max']-x['min']) * self.density
        density_y = (y['max']-y['min']) * self.density
        density = np.max(density_x, density_y)
        x = np.linspace(x['min'], x['max'], density)
        y = np.linspace(y['min'], y['max'], density)
        x, y = np.meshgrid(x, y)
        z = np.array([self.__call__(THETA)
                    for THETA in zip(np.ravel(x), np.ravel(y))])
        z = z.reshape(x.shape)  
        d = {'x': x, 'y': y, 'z':z}
        return d    
    
    @abstractmethod
    def __call__(self, theta):
        """Computes the objective function value"""
        pass
    
    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        return map(grad(self.__call__), theta)
# --------------------------------------------------------------------------  #
class Adjiman(Objective):
    """Base class for objective functions.""" 

    @property   
    def start(self):
        return np.array([-4,0])

    @property
    def minimum(self):
        return np.array([0,0])

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -4, 'max': 4}
        y = {'min': -5, 'max': 5}
        return x, y
    
    def __call__(self, theta):
        """Computes the objective function value"""
        return np.sum(np.multiply(np.sqrt(theta, np.sin(theta)))) 

# --------------------------------------------------------------------------  #
class BartelsConn(Objective):
    """Base class for objective functions."""    
    @property
    def start(self):
        return np.array([-500,500])

    @property
    def minimum(self):
        return np.array([0,0])        

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -500, 'max': 500}
        y = {'min': -500, 'max': 500}
        return x, y
    
    def __call__(self, theta):
        """Computes the objective function value"""
        return np.sum(np.multiply(np.sqrt(theta, np.sin(theta))))         

# --------------------------------------------------------------------------  #
class SumSquares(Objective):
    """Base class for objective functions."""    

    @property
    def start(self):
        return np.array([10,10])
        
    @property
    def minimum(self):
        return np.array([0,0])           
    
    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -10, 'max': 10}
        y = {'min': -10, 'max': 10}
        return x, y
    
    def __call__(self, theta):
        """Computes the objective function value"""        
        return np.sum(theta[0]**2 + 2 * theta[1]**2)

# --------------------------------------------------------------------------  #
class GoldsteinPrice(Objective):
    """Base class for objective functions."""    

    @property
    def start(self):
        return np.array([2,-2])

    @property
    def minimum(self):
        return np.array([0,-1])           

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -2, 'max': 2}
        y = {'min': -2, 'max': 2}
        return x, y
    
    def __call__(self, theta):
        """Computes the objective function value"""
        a = (theta[0] + theta[1] + 1)**2
        b = (19 - 14 * theta[0] + 3 * theta[0]**2 - 14 * theta[1] + \
            6 * np.multiply(theta[0], theta[1]) + 3 * theta[1]**2)
        c = (2 * theta[0] - 3 * theta[1])**2
        d = (18 - 32 * theta[0] + 12 * theta[0]**2 + 48 * theta[1] -\
            36 * np.multiply(theta0, theta1) + 27 * theta[1]**2)
        return (1 + (a * b)) * (30 + (c * d))

# --------------------------------------------------------------------------  #
class Himmelblau(Objective):
    """Base class for objective functions."""    

    @property
    def start(self):
        return np.array([6,-6])

    @property
    def minimum(self):
        return np.array([3,2], [-2.805118, 3.131312], \
                        [-3.779310,-3.283186], [3.584428,-1.848126])           

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -6, 'max': 6}
        y = {'min': -6, 'max': 6}
        return x, y    

    def __call__(self, theta):
        """Computes the objective function value"""
        return (theta[0]**2 + theta[1] - 11)**2 + (theta[0]+theta[1]**2-7)**2

# --------------------------------------------------------------------------  #
class Leon(Objective):
    """Base class for objective functions."""    

    @property
    def start(self):
        return np.array([6,6])

    @property
    def minimum(self):
        return np.array([1,1])  

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -6, 'max': 6}
        y = {'min': -6, 'max': 6}
        return x, y            

    def __call__(self, theta):
        """Computes the objective function value"""
        return 100 * (theta[1] - theta[0]**3)**2 + (1 - theta[0])**2

   

# --------------------------------------------------------------------------  #
class Rosenbrock(Objective):
    """Base class for objective functions."""

    @property
    def start(self):
        return np.array([10,10])

    @property
    def minimum(self):
        return np.array([1,1])  

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -10, 'max': 10}
        y = {'min': -10, 'max': 10}
        return x, y            

    
    def __call__(self, theta):
        """Computes the objective function value"""
        return (1-theta[0])**2 + (100 * (theta[1] - theta[0]**2)**2)        

# --------------------------------------------------------------------------  #
class StyblinskiTank(Objective):
    """Styblinksi-Tank objective functions."""

    @property
    def start(self):
        return np.array([5,-5])

    @property
    def minimum(self):
        return np.array([−2.903534, −2.903534])  

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -5, 'max': 5}
        y = {'min': -5, 'max': 5}
        return x, y            

    
    def __call__(self, theta):
        """Computes the objective function value"""
        return 1/2 * np.sum(theta**4 - (16 * theta**2) + (5 * theta))            