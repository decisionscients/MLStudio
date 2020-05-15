#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : base_regression.py                                                #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 14th 2020, 10:27:39 pm                        #
# Last Modified : Thursday, May 14th 2020, 10:27:39 pm                        #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Base class for all regression algorithms."""
from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array
# --------------------------------------------------------------------------  #
class BaseModel(ABC, BaseEstimator):
    """Base class for all regression subclasses."""

    @abstractmethod
    def __init__(self):      
        raise Exception("Instantiation of the BaseModel class is prohibited.")  

        
    def _validate_hyperparam(self, p):
        """Validates a parameter. Used for validating regularization parameters."""
        assert isinstance(p, (int,float)), "Regularization hyperparameter must be numeric."
        assert p >= 0 and p <= 1, "Regularization parameter must be between zero and 1."

    @abstractmethod
    def compute_output(self, X, theta):
        """Computes output based upon inputs and model parameters.

        For linear regression, this is the linear combination of the inputs
        and the model parameters, theta. For logistic and multiclass logistic
        regression, this is the output from the sigmoid and softmax functions
        respectively.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The model inputs. Note the number of features includes the coefficient
            for the bias term

        theta : array of shape [n_features,] or [n_features, n_classes]
            Model parameters

        Returns
        -------
        output : Model output
        
        """
        pass

    @abstractmethod
    def predict(self, X, theta):
        """Computes the prediction using final model parameters.        

        Parameter
        ---------
        X : np.array of shape [n_samples, n_features]

        theta : array of shape [n_features,] or [n_features, n_classes]
            Model parameters
             
        Returns
        -------
        prediction 

        """ 
        pass
        

    @abstractmethod
    def compute_cost(self, y, y_pred, theta):
        """Implements the cost function.

        Parameters
        ----------
        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions 

        theta : array of shape (n_features,) or (n_features, n_classes)        
            The model parameters            

        Returns
        -------
        cost : Computed cost of the objective function. 

        """
        pass

    @abstractmethod
    def compute_gradient(self, X, y, y_pred, theta):
        """Computes the gradient of cost function.

        Parameters
        ----------
        X : array of shape (m_observations, n_features)
            Input data

        y : array of shape (n_features,)
            Ground truth target values

        y_pred : array of shape (n_features,)
            Predictions         

        theta : array of shape (n_features,) or (n_features, n_classes)           

        Returns
        -------
        gradient of the cost function w.r.t. the parameters.

        """
        pass
