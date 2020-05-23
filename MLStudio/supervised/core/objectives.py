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

import numpy as np
from sklearn.base import BaseEstimator

from mlstudio.supervised.core.regularizers import Nill
from mlstudio.utils.data_manager import VectorScaler
# --------------------------------------------------------------------------  #
#                        OBJECTIVE BASE CLASS                                 #
# --------------------------------------------------------------------------  #
class Objective(ABC, BaseEstimator):
    """Base class for all objective functions.
    
    Parameters
    ----------
    regularizer : a Regularizer object
        Object to perform L1, L2, or L1_L2 regularization of the gradient

    gradient_scaler : a GradientScaler object
        Performs gradient scaling 
    """

    def __init__(self, regularizer=None, gradient_scaler=None):
        self.regularizer = regularizer
        self.gradient_scaler = gradient_scaler
        if not regularizer:
            self.regularizer = Nill()       
        

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def __call__(self, theta, **kwargs):
        """Computes the objective function.

        Parameters
        ----------
        theta : array-like
            The parameters from the model

        kwargs : Arbitrary keyword arguments.

        Returns
        -------
        real number
        """
        pass

    @abstractmethod
    def gradient(self, theta, **kwargs):
        """Computes the derivative of the objective function w.r.t. theta.
        
        Parameters
        ----------
        theta : array-like
            The parameters from the model

        kwargs : Arbitrary keyword arguments.

        Returns
        -------
        gradient : array-like in the shape of theta 
            The gradient of the objective function
        """
        pass
         

# --------------------------------------------------------------------------  #
class Cost(Objective):
    """Base class for all cost classes."""
    
    @property
    def name(self):
        return "Cost Base Class"        

    @abstractmethod
    def __call__(self, theta, y, y_out):
        pass

    @abstractmethod
    def gradient(self, theta, X, y, y_out):
        pass

# --------------------------------------------------------------------------  #
class MSE(Cost):

    @property
    def name(self):
        return "Mean Squared Error Cost Function"        

    def __call__(self, theta, y, y_out):
        """Computes the mean squared error cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        n_samples = y.shape[0]
        J = 0.5 * np.mean((y-y_out)**2) 
        # Add regularizer of weights
        J += self.regularizer(theta)  / n_samples
        return J

    def gradient(self, theta, X, y, y_out):
        """Computes quadratic costs gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = X.shape[0]
        dZ = y_out-y
        dW = float(1. / n_samples) * X.T.dot(dZ) 
        # Add the gradient of regularizer of weights 
        dW += self.regularizer.gradient(theta) / n_samples        
        # Check scale of gradient if requested
        if self.gradient_scaler:
            dW = self.gradient_scaler.fit_transform(dW)                
        return(dW)        

# --------------------------------------------------------------------------  #
class CrossEntropy(Cost):

    @property
    def name(self):
        return "Cross Entropy Cost Function"        

    def __call__(self, theta, y, y_out):
        """Computes cross entropy cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)        
        J = -1*(1/n_samples) * np.sum(np.multiply(y, np.log(y_out)) + \
            np.multiply(1-y, np.log(1-y_out))) 
        # Add regularizer of weights 
        J += self.regularizer(theta) / n_samples        
        return J   

    def gradient(self, theta, X, y, y_out):
        """Computes cross entropy cost  gradient with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = X.shape[0]
        dZ = y_out-y
        dW = float(1./n_samples) * (dZ).dot(X)
        dW += self.regularizer.gradient(theta) / n_samples        
        # Check scale of gradient if requested
        if self.gradient_scaler:
            dW = self.gradient_scaler.fit_transform(dW)                         
        return(dW)          

# --------------------------------------------------------------------------  #
class CategoricalCrossEntropy(Cost):


    @property
    def name(self):
        return "Categorical Cross Entropy Cost Function"        

    def __call__(self, theta, y, y_out):
        """Computes categorical cross entropy cost.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters            

        Returns
        -------
        cost : The quadratic cost 

        """
        
        n_samples = y.shape[0]
        # Prevent division by zero
        y_out = np.clip(y_out, 1e-15, 1-1e-15)    
        # Obtain unregularized cost
        J = np.mean(-np.sum(np.log(y_out) * y, axis=1))
        # Add regularizer of weights 
        J += self.regularizer(theta) / n_samples
        return J 

    def gradient(self, theta, X, y, y_out):
        """Computes gradient of cross-entropy cost with respect to weights.
        
        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_out : array of shape (n_features,)
            Output from the model 

        theta : array of shape (n_features,)  
            The model parameters                        

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        n_samples = y.shape[0]
        dZ = y_out-y
        dW = 1/n_samples * X.T.dot(dZ)
        # Add regularizer of weights 
        dW += self.regularizer.gradient(theta) / n_samples        
        # Check scale of gradient if requested
        if self.gradient_scaler:
            dW = self.gradient_scaler.fit_transform(dW)                             
        return(dW)                  
# --------------------------------------------------------------------------  #
#                         BENCHMARK FUNCTIONS                                 #        
# --------------------------------------------------------------------------  #
class Benchmark(Objective):
    """Base class for objective functions."""

    @property
    def name(self):
        return "Benchmark Base Class"

    @property
    def density(self):
        """The linspace density per point of difference between min and max.""" 
        return 2

    @property
    def start(self):
        """Returns a good start point."""
        pass

    @property
    def minimum(self):
        """Returns a good start point."""
        pass    

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
    
    @abstractmethod
    def gradient(self, theta, **kwargs):
        """Computes the gradient of the objective function."""
        pass
# --------------------------------------------------------------------------  #
class Adjiman(Benchmark):
    """Base class for objective functions.""" 

    @property
    def name(self):
        return "Adjiman Objective Function"    

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
        return np.cos(theta[0]) * np.sin(theta[1]) - (theta[0] / (theta[1]**2 + 1))

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        dfdx = -(1/(theta[1]**2+1))*((theta[1]**2+1)*np.sin(theta[0])*np.sin(theta[1])+1)
        dfdy = 2*theta[0]*theta[1] /(theta[1]**2+1)**2 + np.cos(theta[0])*np.cos(theta[1])
        return np.array([dfdx, dfdy])

# --------------------------------------------------------------------------  #
class BartelsConn(Benchmark):
    """Base class for objective functions."""    

    @property
    def name(self):
        return "Bartels Conn Objective Function"
    
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
        a = abs(theta[0]**2 + theta[1]**2 + theta[0] * theta[1])
        b = abs(np.sin(theta[0]))
        c = abs(np.cos(theta[1]))
        return a + b + c

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        a = (2 * theta[0] + theta[1]) * np.sign(theta[0]**2 + theta[0] * theta[1] + theta[1]**2)
        b = np.cos(theta[0]) * np.sign(np.sin(theta[0]))
        dfdx = a + b
        a = (theta[0] + 2 * theta[1]) * np.sign(theta[0]**2 + theta[0] * theta[1] + theta[1]**2)
        b = np.sin(theta[1]) * np.sign(np.cos(theta[1]))
        dfdy = a - b
        return np.array([dfdx, dfdy])

# --------------------------------------------------------------------------  #
class GoldsteinPrice(Benchmark):
    """Base class for objective functions."""    

    @property
    def name(self):
        return "Goldstein-Price Function"    

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
            36 * np.multiply(theta[0], theta[1]) + 27 * theta[1]**2)
        return (1 + (a * b)) * (30 + (c * d))

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        ax = 9 * ((2/3)*theta[0]-theta[1])**2 
        bx =24*theta[0]-36*theta[1]-32
        cx = 8*theta[0]-12*theta[1]
        dx = 12*theta[0]**2-36*theta[0]*theta[1]-32*theta[0]+\
                27*theta[1]**2+48*theta[1]+18
        ex = (theta[0]+theta[1]+1)**2
        fx = 3*theta[0]**2+6*theta[0]*theta[1]-14*theta[0]+\
            3*theta[1]**2-14*theta[1]+19
        gx = 1
        hx = 9*((2/3)*theta[0]-theta[1])**2
        ix = 12*theta[0]**2-36*theta[0]*theta[1]-\
            32*theta[0]+27*theta[1]**2+48*theta[1]+18
        jx = 30
        kx = (theta[0]+theta[1]+1)**2
        lx = 6*theta[0]+6*theta[1]-14
        mx = 2*theta[0]+2*theta[1]+2
        nx = 3*theta[0]**2+6*theta[0]*theta[1]-14*theta[0]+3*theta[1]**2-14*theta[1]+19
        dfdx = (ax*bx+cx*dx) * (ex*fx+gx) + (hx*ix+jx) * (kx*lx+mx*nx)

        ay = -12*theta[0]+18*theta[1]
        by = dx
        cy = ax
        dy = -36*theta[0]+54*theta[1]+48
        ey = ex
        fy = fx
        gy = gx
        hy = hx
        iy = ix
        jy = jx
        ky = kx
        ly = lx
        my = mx
        ny = nx
        a = (theta[0] + 2 * theta[1]) * np.sign(theta[0]**2 + theta[0] * theta[1] + theta[1]**2)
        b = np.sin(theta[1]) * np.sign(np.cos(theta[1]))
        dfdy = (ay*by+cy*dy) * (ey*fy+gy) + (hy*iy+jy) * (ky*ly+my*ny)
        return np.array([dfdx, dfdy])        

# --------------------------------------------------------------------------  #
class Himmelblau(Benchmark):
    """Base class for objective functions."""   

    @property
    def name(self):
        return "Himmelblau Objective Function"     

    @property
    def start(self):
        return np.array([6,-6])

    @property
    def minimum(self):
        return np.array([3,2])           

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -6, 'max': 6}
        y = {'min': -6, 'max': 6}
        return x, y    

    def __call__(self, theta):
        """Computes the objective function value"""
        return (theta[0]**2 + theta[1] - 11)**2 + (theta[0]+theta[1]**2-7)**2

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        dfdx = 4*theta[0]*(theta[0]**2+theta[1]-11)+2*theta[0]+2*theta[1]**2-14
        dfdy = 2*theta[0]**2 + 4*theta[1] * (theta[0]+theta[1]**2-7)+2*theta[1]-22
        return np.array([dfdx, dfdy])        

# --------------------------------------------------------------------------  #
class Leon(Benchmark):
    """Base class for objective functions."""    

    @property
    def name(self):
        return "Leon Objective Function"    

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
        return 100 * (theta[1] - theta[0]**2)**2 + (1 - theta[0])**2

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        dfdx = -400*(-theta[0]**2+theta[1]) + 2*theta[0] - 2
        dfdy = -200*theta[0]**2 + 200 * theta[1]
        return np.array([dfdx, dfdy])             

   

# --------------------------------------------------------------------------  #
class Rosenbrock(Benchmark):
    """Base class for objective functions."""

    @property
    def name(self):
        return "Rosenbrock Objective Function"    

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
        a = 1
        b = 100
        return b*(theta[1]-theta[0]**2)**2+(a-theta[0])**2
        n = 2
        score = 0
        for i in range(n-1):
            score += (b*(theta[i+1]-theta[i]**2)+(a-theta[i])**2)
        return score

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        dfdx = -400*theta[0]*(-theta[0]**2+theta[1])+2*theta[0]-2
        dfdy = -200*theta[0]**2 + 200 * theta[1]
        return np.array([dfdx, dfdy])      

# --------------------------------------------------------------------------  #
class StyblinskiTank(Benchmark):
    """Styblinksi-Tank objective functions."""

    @property
    def name(self):
        return "Styblinski-Tank Objective Function"    

    @property
    def start(self):
        return np.array([5,-5])

    @property
    def minimum(self):
        return np.array([-2.903534, -2.903534])  

    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -5, 'max': 5}
        y = {'min': -5, 'max': 5}
        return x, y            

    
    def __call__(self, theta):
        """Computes the objective function value"""        
        return 1/2 * np.sum(theta**4 - (16 * theta**2) + (5 * theta))     

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        dfdx = 2*theta[0]**3 - 16 * theta[0] + 5/2
        dfdy = 2*theta[1]**3 - 16*theta[1] + 5/2
        return np.array([dfdx, dfdy])                  

# --------------------------------------------------------------------------  #
class SumSquares(Benchmark):
    """Base class for objective functions."""    

    @property
    def name(self):
        return "Sum Squares Objective Function"    

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
        return np.sum(theta[0]**2 + theta[1]**2)

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        dfdx = 2 * theta[0]
        dfdy = 2 * theta[1]
        return np.array([dfdx, dfdy])        

# --------------------------------------------------------------------------  #
class ThreeHumpCamel(Benchmark):
    """Three hump camel objective functions."""    

    @property
    def name(self):
        return "Three Hump Camel Objective Function"    

    @property
    def start(self):
        return np.array([2,2])
        
    @property
    def minimum(self):
        return np.array([0,0])           
    
    @property
    def _range(self):
        """Returns the x and y ranges for plotting."""
        x = {'min': -2, 'max': 2}
        y = {'min': -2, 'max': 2}
        return x, y
    
    def __call__(self, theta):
        """Computes the objective function value"""        
        return 2*theta[0]**2 - 1.05*theta[0]**4 + ((theta[0]**6)/6)+theta[0]*theta[1]+theta[1]**2

    def gradient(self, theta):
        """Computes the gradient of the objective function."""
        dfdx = theta[0]**5-((21*theta[0]**3)/5)+4*theta[0]+theta[1]
        dfdy = theta[0]+2*theta[1]
        return np.array([dfdx, dfdy]) 
