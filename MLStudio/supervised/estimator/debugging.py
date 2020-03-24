#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : debugging.py                                                      #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Monday, March 23rd 2020, 4:06:07 pm                         #
# Last Modified : Monday, March 23rd 2020, 4:35:57 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Call back that performs gradient checking."""
import numpy as np
import pandas as pd

from mlstudio.supervised.estimator.callbacks import Callback

# --------------------------------------------------------------------------- #
#                              GRADIENT CHECK                                 #
# --------------------------------------------------------------------------- #
class GradientCheck(Callback):
    """Performs gradient checking."""

    def __init__(self, iterations=50, epsilon=1e-4):
        super(GradientCheck, self).__init__()
        self.epsilon = epsilon
        self.iterations = iterations
        self._algorithm = None
        
    def on_train_begin(self, logs=None):        
        """Initializes gradient check parameters.
        
        Parameters
        ----------
        log : dict
            Contains no information
        """
        super(GradientCheck, self).on_train_begin()
        self._algorithm = self.model.algorithm
        self._n = 0
        self._iteration = []
        self._gradients = []
        self._approximations = []
        self._differences = []
        self._results = []        

    def on_epoch_end(self, epoch, logs=None):
        """Checks gradient each self.iterations number of iterations.

        Parameters
        ----------
        epoch : int
            The current epoch number

        logs : dict
            Dictionary containing training cost, (and if metric=score, 
            validation cost)  

        """
        if self.model.gradient_check:
            if logs.get('epoch') % self.iterations == 0:
                X = logs.get('X')
                y = logs.get('y')
                epoch = logs.get('epoch')
                learning_rate = logs.get('learning_rate')
                theta = logs.get('theta')

                grad_approx = []
                for i in np.arange(len(theta)):

                    # Compute theta differentials
                    theta_plus = theta.copy()
                    theta_minus = theta.copy()
                    theta_plus[i] = theta_plus[i] + self.epsilon
                    theta_minus[i] = theta_minus[i] - self.epsilon
                    # Compute associated costs
                    y_pred = self._algorithm.predict(X, theta_plus)
                    J_plus = self._algorithm.compute_cost(y, y_pred, theta_plus)
                    y_pred = self._algorithm.predict(X, theta_minus)
                    J_minus = self._algorithm.compute_cost(y, y_pred, theta_minus)

                    # Estimate the gradient
                    grad_approx_i = (J_plus - J_minus) / (2 * self.epsilon)         
                    grad_approx.append(grad_approx_i)
                
                # Compute gradient via back-propagation
                y_pred = self._algorithm.predict(X, theta)
                grad = self._algorithm.compute_gradient(X, y, y_pred, theta) 

                grad = np.array(grad)
                grad_approx = np.array(grad_approx)

                # Evaluate
                numerator = np.linalg.norm(grad-grad_approx)
                denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approx)
                difference = numerator / denominator
                result = difference < self.epsilon

                # Update results
                self._n += 1
                self._iteration.append(self._n)
                self._gradients.append(grad)
                self._approximations.append(grad_approx)
                self._differences.append(difference)
                self._results.append(result)

    def on_train_end(self, logs=None):
        d = {"Iteration": self._iteration, "Difference": self._differences,
             "Result": self._results}
        df = pd.DataFrame(d)
        failures = len(df[df['Result']== False])
        successes = len(df[df['Result']== True])
        avg_difference = df['Difference'].mean(axis=0)
        print("\n","* ",40*"=", " *")
        print("Gradient Checks for {desc}".format(desc=self.model.description))
        print("  Num Failures : {failures}".format(failures=failures))
        print(" Num Successes : {successes}".format(successes=successes))
        print(" Pct Successes : {pct}".format(pct=successes/(self._n)*100))
        print("Avg Difference : {diff}".format(diff=avg_difference))
        