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
# Created       : Sunday, March 22nd 2020, 3:54:52 am                         #
# Last Modified : Sunday, March 22nd 2020, 3:55:06 am                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
""" Debugging classes."""
import numpy as np

class GradientCheck:
    """Performs gradient checks."""
    def __init__(self, algorithm, iterations=50, epsilon=1e-7):
        self.epsilon = epsilon
        self.iterations = iterations
        self.algorithm = algorithm
        self._n = 0
        self._iteration = []
        self._gradients = []
        self._approximations = []
        self._differences = []
        self._results = []

    def check_gradient(self, X, y, theta, learning_rate):
        """Performs the actual gradient check."""
        grad_approx = []
        for i in np.arange(len(theta)):

            # Compute theta differentials
            theta_plus = theta
            theta_minus = theta
            theta_plus[i] = theta_plus[i] + self.epsilon
            theta_minus[i] = theta_minus[i] - self.epsilon
            # Compute associated costs
            _, J_plus = self.algorithm.propagate_forward(X, y, theta_plus)
            _, J_minus = self.algorithm.propagate_forward(X, y, theta_minus)
            # Estimate the gradient            
            grad_approx.append((J_plus - J_minus) / (2 * self.epsilon))
        
        # Compute gradient via back-propagation
        y_pred = self.algorithm.predict(X, theta)
        grad = self.algorithm.propagate_backward(X, y, y_pred, theta, learning_rate) 

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

    def check_results(self):
        d = {"Iteration": self._iteration, "Gradient": self._gradients,
             "Approximations": self._approximations, "Difference": self._differences,
             "Result": self._results}
        df = pd.DataFrame(d)
        failures = len(df[df['Result']== False])
        successes = len(df[df['Result']== True])
        avg_difference = df['Difference'].mean(axis=0)
        print("\n",40*"*")
        print("\n  Num Failures : {failures}".format(failures=failures))
        print("\n Num Successes : {successes}".format(successes=successes))
        print("\n Pct Successes : {pct}".format(pct=successes/(self._n)*100))
        print("\nAvg Difference : {diff}".format(diff=avg_difference))
        return failures






