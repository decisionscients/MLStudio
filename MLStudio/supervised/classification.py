#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : MLStudio                                                          #
# Version : 0.1.0                                                             #
# File    : classification.py                                                 #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Wednesday, March 18th 2020, 4:34:57 am                      #
# Last Modified : Saturday, March 21st 2020, 5:47:08 pm                       #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Classes supporting binary and multinomial classification ."""
import numpy as np
from sklearn.base import RegressorMixin, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

from mlstudio.supervised.estimator.cost import Cost
from mlstudio.supervised.estimator.cost import BinaryClassificationCostFactory
from mlstudio.supervised.estimator.cost import MultinomialClassificationCostFactory
from mlstudio.supervised.estimator.scorers import ClassificationScorerFactory
from mlstudio.supervised.estimator.gradient import GradientDescent
from mlstudio.utils.data_manager import data_split, one_hot

# --------------------------------------------------------------------------- #
#                          LOGISTIC REGRESSION                                #
# --------------------------------------------------------------------------- #            
class LogisticRegression(GradientDescent, ClassifierMixin):
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
    _TASK = "Logistic Regression"

    def __init__(self, name=None, gradient_descent=True, 
                 learning_rate=0.01, batch_size=None, theta_init=None,  
                 epochs=1000, algorithm=LogisticRegression(),
                 optimizer=Standard(), regularizer=NullRegularizer(),
                 scorer=Accuracy(), early_stop=False, val_size=0.0, 
                 verbose=False, checkpoint=100, random_state=None):
        super(LogisticRegression, self).__init__(name=name,
                                         learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, 
                                         algorithm=algorithm,
                                         optimizer=optimizer,
                                         regularizer=regularizer,
                                         scorer=scorer,
                                         early_stop=early_stop, 
                                         val_size=val_size,                                          
                                         verbose=verbose, checkpoint=checkpoint, 
                                         random_state=random_state)   

        self.gradient_descent = gradient_descent

    def _prepare_data(self, X, y):
        """Creates the X design matrix and saves data as attributes."""
        self._X = self._X_val = self._y = self._y_val = None
        # Add a column of ones to train the intercept term
        self._X = np.array(X)
        self._X_design = np.insert(self._X, 0, 1.0, axis=1)  
        
        # Add classes_ attribute
        self._y = np.array(y)
        self.classes_, self._y = np.unique(self._y, return_inverse=True)        

        # Set aside val_size training observations for validation set 
        if self.val_size:
            self._X_design, self._X_val, self._y, self._y_val = \
                data_split(self._X_design, self._y, 
                test_size=self.val_size, random_state=self.random_state)


    def fit(self, X,y):
        """Adds check for classification targets."""
        check_classification_targets(y)
        super(LogisticRegression, self).fit(X,y)

    def predict(self, X):
        """Predicts class label.
        
        Parameters
        ----------
        X : array-like of shape (m, n_features)

        Returns
        -------
        Vector of class label predictions
        """        
        check_is_fitted(self)
        X = np.array(X)
        check_array(X)        
        decision = self.algorithm.predict(X, self._theta)
        y_pred = np.round(decision).astype(int)
        return y_pred


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
        if self.scorer:
            check_X_y(X,y)
            y_pred = self.predict(X)
            score = self.scorer(y=y, y_pred=y_pred)    
        else:
            raise Exception("No scorer class designated. Unable to compute score.")
        return score  
