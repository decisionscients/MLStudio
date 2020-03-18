#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : logistic_regression.py                                             #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Monday, March 16th 2020, 3:06:46 am                          #
# Last Modified : Monday, March 16th 2020, 3:06:46 am                          #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Classes supporting binary and multinomial classification ."""
import numpy as np

from ml_studio.supervised_learning.training.cost import Cost
from ml_studio.supervised_learning.training.cost import BinaryClassificationCostFunction
from ml_studio.supervised_learning.training.cost import BinaryClassificationCostFactory
from ml_studio.supervised_learning.training.cost import MultinomialClassificationCostFunction
from ml_studio.supervised_learning.training.cost import MultinomialClassificationCostFactory
from ml_studio.supervised_learning.training.metrics import ClassificationScorer
from ml_studio.supervised_learning.training.metrics import ClassificationScorerFactory
from ml_studio.supervised_learning.training.estimator import Estimator
from ml_studio.utils.data_manager import data_split, one_hot

# --------------------------------------------------------------------------- #
#                          LOGISTIC CLASSIFICATION                            #
# --------------------------------------------------------------------------- #            
class LogisticRegression(GradientDescent):
    """Trains models for binary classification using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, optional (default='binary_cross_entropy')
        The string name for the cost function

        'binary_cross_entropy':
            Computes binary cross entropy 
        'hinge':
            Computes Hinge Loss
        'squared_hinge':
            Computes Squared Hinge Loss

    metric : str, optional (default='accuracy')
        Metrics used to evaluate classification scores:

        'accuracy': 
            Accuracy - Total Accurate Predictions / Total Predictions
        'auc': 
            Compute Area Under the Curve (AUC)
        'confusion_matrix':
            Compute confusion matrix to evaluate accuracy of a classification
        'f1':
            Compute F1 score.
        'precision':
            Compute the precision
        'recall':
            Compute the recall
        'roc':
            Compute Reciever Operating Characteristics (ROC)

    early_stop : None or EarlyStop subclass, optional (default=None)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.3
        The proportion of the training set to allocate to the validation set.
        Must be between 0 and 1. Only used when early_stop is not False.

    patience : int, default=5
        The number of consecutive iterations with no improvement to wait before
        early stopping.

    precision : float, default=0.01
        The stopping criteria. The precision with which improvement in training
        cost or validation score is measured e.g. training cost at time k+1
        has improved if it has dropped training cost (k) * precision.

    verbose : bool, optional (default=False)
        If true, performance during training is summarized to sysout.

    checkpoint : None or int, optional (default=100)
        If verbose, report performance each 'checkpoint' epochs

    name : None or str, optional (default=None)
        The name of the model used for plotting

    random_state : None or int, optional (default=None)
        Random state random_state        

    Attributes
    ----------
    coef_ : array-like shape (n_features,1) or (n_features, n_classes)
        Coefficient of the features in X. 'coef_' is of shape (n_features,1)
        for binary problems. For multi-class problems, 'coef_' corresponds
        to outcome 1 (True) and '-coef_' corresponds to outcome 0 (False).

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 
        'intercept_' is of shape (1,) for binary problems. For multi-class
        problems, `intercept_` corresponds to outcome 1 (True) and 
        `-intercept_` corresponds to outcome 0 (False).

    epochs_ : int
        Total number of epochs executed.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    classification.MultinomialLogisticRegression : Multinomial Classification
    """    
    _DEFAULT_METRIC = 'accuracy'

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None,
                 epochs=1000, cost='binary_cross_entropy',                 
                 metric='accuracy',  early_stop=False, 
                 val_size=0.3, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, random_state=None):
        super(LogisticRegression,self).__init__(learning_rate=learning_rate, 
                 batch_size=batch_size, theta_init=theta_init, 
                 epochs=epochs, cost=cost, metric=metric,  
                 early_stop=early_stop, val_size=val_size, patience=patience, 
                 precision=precision, verbose=verbose, checkpoint=checkpoint, 
                 name=name, random_state=random_state)                 

    def _sigmoid(self, z):
        """Computes the sigmoid for a scalar or vector z."""
        s = 1/(1+np.exp(-z))
        return s                 

    def set_name(self):
        """Set name of model for plotting purposes."""
        self._get_algorithm_name()
        self.task = "Logistic Regression"
        self.name = self.name or self.task + ' with ' + self.algorithm

    def _get_cost_function(self):
        """Obtains the cost function associated with the cost parameter."""
        cost_function = BinaryClassificationCostFactory()(cost=self.cost)
        if not isinstance(cost_function, BinaryClassificationCostFunction):
            msg = str(self.cost) + ' is not a supported binary classification cost function.'
            raise ValueError(msg)
        else:
            return cost_function


    def _get_scorer(self):
        """Obtains the scoring function associated with the metric parameter."""
        
        if self.metric:
            try:
                scorer = ClassificationScorerFactory()(metric=self.metric)
            except ValueError:
                msg = str(self.metric) + ' is not a supported classification metric.'
                print(msg)
            return scorer


    def score(self, X, y):
        """Computes a score for the current model, given inputs X and output y.

        The score uses the class associated the metric parameter from class
        instantiation.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        y : numpy array, shape (n_samples,)
            Target values             

        Returns
        -------
        float
            Returns the score for the designated metric.
        """
        check_X_y(X, y)
        y_pred = self.predict(X)
        if self.metric:
            score = self.scorer_(y=y, y_pred=y_pred)    
        else:
            score = ClassificationScorerFactory()(metric=self._DEFAULT_METRIC)(y=y, y_pred=y_pred)        
        return score    


    def _predict(self, X):
        """Predicts sigmoid probabilities."""        
        z = self._linear_prediction(X) 
        y_pred = self._sigmoid(z).astype('float64').flatten()
        return y_pred

    def predict(self, X):
        """Predicts class label.
        
        Parameters
        ----------
        X : array-like of shape (m, n_features)

        Returns
        -------
        Vector of class label predictions
        """        
        prob = self._predict(X)      
        y_pred = np.round(prob).astype(int).flatten()
        return y_pred
