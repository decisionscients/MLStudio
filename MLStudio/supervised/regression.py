#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ============================================================================ #
# Project : MLStudio                                                           #
# Version : 0.1.0                                                              #
# File    : regression.py                                                      #
# Python  : 3.8.2                                                              #
# ---------------------------------------------------------------------------- #
# Author  : John James                                                         #
# Company : DecisionScients                                                    #
# Email   : jjames@decisionscients.com                                         #
# URL     : https://github.com/decisionscients/MLStudio                        #
# ---------------------------------------------------------------------------- #
# Created       : Sunday, March 15th 2020, 9:04:06 pm                          #
# Last Modified : Sunday, March 15th 2020, 9:04:06 pm                          #
# Modified By   : John James (jjames@decisionscients.com)                      #
# ---------------------------------------------------------------------------- #
# License : BSD                                                                #
# Copyright (c) 2020 DecisionScients                                           #
# ============================================================================ #
"""Linear Regression, L1, L2 and ElasticNet Regression classes."""
from abc import abstractmethod
import numpy as np
from numpy.linalg import inv, svd
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

from mlstudio.supervised.estimator.regularizers import L1, L2, ElasticNet
from mlstudio.supervised.estimator.gradient import GradientDescent
from mlstudio.supervised.estimator.scorers import RegressionScorerFactory
from mlstudio.supervised.estimator.cost import RegressionCostFactory

import warnings

# --------------------------------------------------------------------------- #
#                          REGRESSION CLASS                                   #
# --------------------------------------------------------------------------- #

class Regression(GradientDescent):
    """Base class for all regression classes."""

    _DEFAULT_METRIC = 'mse'
    _TASK = "Regression"

    def __init__(self, learning_rate=0.01, batch_size=None, theta_init=None, 
                 epochs=1000, cost='quadratic', metric='mse', 
                 early_stop=False, val_size=0.0, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, random_state=None):
        super(Regression, self).__init__(learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, cost=cost, 
                                         metric=metric, 
                                         early_stop=early_stop, 
                                         val_size=val_size, 
                                         patience=patience, 
                                         precision=precision,
                                         verbose=verbose, checkpoint=checkpoint, 
                                         name=name, random_state=random_state)     

    def _get_cost_function(self):
        """Obtains the cost function associated with the cost parameter."""
        cost_function = RegressionCostFactory()(cost=self.cost)
        if not cost_function:
            msg = str(self.cost) + ' is not a supported regression cost function.'
            raise ValueError(msg)
        else:
            return cost_function

    def _get_scorer(self):
        """Obtains the scoring function associated with the metric parameter."""
        if self.metric:
            try:
                scorer = RegressionScorerFactory()(metric=self.metric)
            except ValueError:
                msg = str(self.metric) + ' is not a supported regression metric.'
                print(msg)
            return scorer
        
    def _predict(self, X):
        """Computes predictions during training with current weights."""        
        y_pred = self._linear_prediction(X)
        return y_pred.ravel()

    def predict(self, X):
        """Predicts output as a linear function of inputs and final parameters.

        The method computes predictions based upon final parameters; therefore,
        the model must have been trained.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Feature matrix for which predictions will be rendered.

        Returns
        -------
        array, shape(n_samples,)
            Returns the linear regression prediction.        
        """
        check_is_fitted(self)
        X = np.array(X)
        check_array(X)
        return self._predict(X)

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
            score = RegressionScorerFactory()(metric=self._DEFAULT_METRIC)(y=y, y_pred=y_pred)        
        return score

# --------------------------------------------------------------------------- #
#                            LINEAR REGRESSION                                #
# --------------------------------------------------------------------------- #

class LinearRegression(Regression):
    """Performs linear regression analytically or by gradient descent.
    
    Parameters
    ----------
    gradient_descent : bool (Default = True)
        If True, parameters are estimated using gradient descent. Otherwise,
        parameters are computed using ordinary least squares (OLS).

    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mse')
        Metrics used to evaluate regression scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mae':
            Mean absolute error
        'mape':
            Mean absolute percentage error
        'mse':
            Mean squared error
        'nmse':
            Negative mean squared error
        'rmse':
            Root mean squared error
        'nrmse':
            Negative root mean squared error
        'msle':
            Mean squared log error
        'rmsle':
            Root mean squared log error
        'medae':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.0
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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

    converged_ : boolean
        If True, the estimator has converged

    is_fitted_ : boolean
        If True, the estimator has been fitted and parameters estimated. 

    epochs_ : int
        Total number of epochs executed.

    history_ : History object
        Object containing training history data. 

    scorer_ : Scorer object.
        Class that calculates scores. There is a Scorer object for each
        metric supported.

    cost_function_ : Cost object.
        Class that computes regression and classification costs for gradient descent.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    regression.LassoRegression : Lasso Regression
    regression.RidgeRegression : Ridge Regression
    regression.ElasticNetRegression : ElasticNet Regression
    """    

    _DEFAULT_METRIC = 'mse'
    _TASK = "Linear Regression"
    
    def __init__(self, gradient_descent=True, learning_rate=0.01, batch_size=None, 
                 theta_init=None, epochs=1000, cost='quadratic', metric='mse', 
                 early_stop=False, val_size=0.0, patience=5, precision=0.001,
                 verbose=False, checkpoint=100, name=None, random_state=None):
        super(LinearRegression, self).__init__(learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, cost=cost, 
                                         metric=metric, 
                                         early_stop=early_stop, 
                                         val_size=val_size, 
                                         patience=patience, 
                                         precision=precision,
                                         verbose=verbose, checkpoint=checkpoint, 
                                         name=name, random_state=random_state)   

        self.gradient_descent=gradient_descent  
            
    @property
    def description(self):
        """Returns the estimator description."""
        if self.gradient_descent:
            description = str(self._TASK + ' with ' + self.algorithm)      
        else:
            description = str(self._TASK + ' by Ordinary Least Squares')
        return description

    def fit(self, X, y):
        """Fits the model to the data.

        If analytical is True, linear regression is performed via ordinary least
        squares approximation of the parameters.  Otherwise, model parameters
        are estimated using gradient descent.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        y : numpy array, shape (n_samples,)
            Target values 

        Returns
        -------
        self : returns instance of self._
        """
        if self.gradient_descent:
            super(LinearRegression, self).fit(X, y)

        else:
            log = {}
            log['X'] = X
            log['y'] = y

            self._begin_training(log)
                
            # Calculate coefficients using closed-form solution
            self._theta = inv(self._X_design.transpose().dot(self._X_design)).dot(self._X_design.transpose()).dot(self._y)

            self._end_training()
        return self


# --------------------------------------------------------------------------- #
#                         LASSO REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #


class LassoRegression(Regression):
    """Trains lasso regression models using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    alpha : Float, default=0.001
        Constant that multiplies the regularization term.

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mse')
        Metrics used to evaluate regression scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mae':
            Mean absolute error
        'mape':
            Mean absolute percentage error
        'mse':
            Mean squared error
        'nmse':
            Negative mean squared error
        'rmse':
            Root mean squared error
        'nrmse':
            Negative root mean squared error
        'msle':
            Mean squared log error
        'rmsle':
            Root mean squared log error
        'medae':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.0
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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

    epochs_ : int
        Total number of epochs executed.

    converged_ : boolean
        If True, the estimator has converged

    is_fitted_ : boolean
        If True, the estimator has been fitted and parameters estimated.        

    history_ : History object
        Object containing training history data. 

    scorer_ : Scorer object.
        Class that calculates scores. There is a Scorer object for each
        metric supported.

    cost_function_ : Cost object.
        Class that computes regression and classification costs for gradient descent.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    regression.LinearRegression : Linear Regression
    regression.RidgeRegression : Ridge Regression
    regression.ElasticNetRegression : ElasticNet Regression
    """    

    _DEFAULT_METRIC = 'mse'
    _TASK = "Linear Regression"

    def __init__(self, name=None, learning_rate=0.01, batch_size=None, 
                 theta_init=None,  epochs=1000, regularizer=L1(alpha=0.0001),
                 early_stop=False, patience=5,  precision=0.001, 
                 cost='quadratic', metric='mse',  val_size=0.0, 
                 verbose=False, checkpoint=100, random_state=None):
        super(LassoRegression, self).__init__(name=name,
                                         learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, 
                                         regularizer=regularizer,
                                         early_stop=early_stop, 
                                         patience=patience, 
                                         precision=precision,
                                         cost=cost, 
                                         metric=metric,                                          
                                         val_size=val_size,                                          
                                         verbose=verbose, checkpoint=checkpoint, 
                                         random_state=random_state)    

# --------------------------------------------------------------------------- #
#                         RIDGE REGRESSION CLASS                              #
# --------------------------------------------------------------------------- #
class RidgeRegression(Regression):
    """Trains ridge regression models using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    alpha : Float, default=0.0001
        Constant that multiplies the regularization term.

    epochs : int, optional (default=1000)
        The number of epochs to execute during training        

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mse')
        Metrics used to evaluate regression scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mae':
            Mean absolute error
        'mape':
            Mean absolute percentage error
        'mse':
            Mean squared error
        'nmse':
            Negative mean squared error
        'rmse':
            Root mean squared error
        'nrmse':
            Negative root mean squared error
        'msle':
            Mean squared log error
        'rmsle':
            Root mean squared log error
        'medae':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.0
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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

    epochs_ : int
        Total number of epochs executed.

    converged_ : boolean
        If True, the estimator has converged

    is_fitted_ : boolean
        If True, the estimator has been fitted and parameters estimated.        

    history_ : History object
        Object containing training history data. 

    scorer_ : Scorer object.
        Class that calculates scores. There is a Scorer object for each
        metric supported.

    cost_function_ : Cost object.
        Class that computes regression and classification costs for gradient descent.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    regression.LinearRegression : Linear Regression
    regression.LassoRegression : Lasso Regression
    regression.ElasticNetRegression : ElasticNet Regression
    """  

    _DEFAULT_METRIC = 'mse'
    _TASK = "Linear Regression"      

    def __init__(self, name=None, learning_rate=0.01, batch_size=None, 
                 theta_init=None,  epochs=1000, regularizer=L2(alpha=0.0001),
                 early_stop=False, patience=5,  precision=0.001, 
                 cost='quadratic', metric='mse',  val_size=0.0, 
                 verbose=False, checkpoint=100, random_state=None):
        super(RidgeRegression, self).__init__(name=name,
                                         learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, 
                                         regularizer=regularizer,
                                         early_stop=early_stop, 
                                         patience=patience, 
                                         precision=precision,
                                         cost=cost, 
                                         metric=metric,                                          
                                         val_size=val_size,                                          
                                         verbose=verbose, checkpoint=checkpoint, 
                                         random_state=random_state)    
                                  

# --------------------------------------------------------------------------- #
#                        ELASTICNET REGRESSION CLASS                          #
# --------------------------------------------------------------------------- #


class ElasticNetRegression(Regression):
    """Trains lasso regression models using Gradient Descent.
    
    Parameters
    ----------
    learning_rate : float or LearningRateSchedule instance, optional (default=0.01)
        Learning rate or learning rate schedule.

    batch_size : None or int, optional (default=None)
        The number of examples to include in a single batch.

    theta_init : None or array_like, optional (default=None)
        Initial values for the parameters :math:`\\theta`

    alpha : Float, default=0.0001
        Constant that multiplies the regularization term.

    ratio : Float, default=0.5
        The L1 ratio with 0 <= ratio <= 1. For ratio = 0 the penalty is an L2 
        penalty. For ratio = 1 it is an L1 penalty. For 0 < ratio < 1, the 
        penalty is a combination of L1 and L2. 

    epochs : int, optional (default=1000)
        The number of epochs to execute during training

    cost : str, (default='quadratic')
        The string name for the cost function

        'quadratic':
            Quadratic or Mean Squared Error (MSE) cost 
        'mae':
            Mean Absolute Error (MAE)
        'huber':
            Computes Huber cost

    metric : str, optional (default='mse')
        Metrics used to evaluate classification scores:

        'r2': 
            R2 - The coefficient of determination
        'var_explained': 
            Percentage of variance explained
        'mae':
            Mean absolute error
        'mape':
            Mean absolute percentage error
        'mse':
            Mean squared error
        'nmse':
            Negative mean squared error
        'rmse':
            Root mean squared error
        'nrmse':
            Negative root mean squared error
        'msle':
            Mean squared log error
        'rmsle':
            Root mean squared log error
        'medae':
            Median absolute error

    early_stop : Bool or EarlyStop subclass, optional (default=True)
        The early stopping algorithm to use during training.

    val_size : Float, default=0.0
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
        Coefficient of the features in X. 

    intercept_ : array-like, shape(1,) or (n_classes,) 
        Intercept (a.k.a. bias) added to the decision function. 

    epochs_ : int
        Total number of epochs executed.

    converged_ : boolean
        If True, the estimator has converged

    is_fitted_ : boolean
        If True, the estimator has been fitted and parameters estimated.        

    history_ : History object
        Object containing training history data. 

    scorer_ : Scorer object.
        Class that calculates scores. There is a Scorer object for each
        metric supported.

    cost_function_ : Cost object.
        Class that computes regression and classification costs for gradient descent.

    Methods
    -------
    fit(X,y) Fits the model to input X and output y
    predict(X) Renders predictions for input X using learned parameters
    score(X,y) Computes a score using metric designated in __init__.
    summary() Prints a summary of the model to sysout.  

    See Also
    --------
    regression.LinearRegression : Linear Regression
    regression.RidgeRegression : Ridge Regression
    regression.LassoRegression : Lasso Regression
    """    

    _DEFAULT_METRIC = 'mse'
    _TASK = "Linear Regression"

    def __init__(self, name=None, learning_rate=0.01, batch_size=None, 
                 theta_init=None,  epochs=1000, 
                 regularizer=ElasticNet(alpha=0.0001, ratio=0.15),
                 early_stop=False, patience=5,  precision=0.001, 
                 cost='quadratic', metric='mse',  val_size=0.0, 
                 verbose=False, checkpoint=100, random_state=None):
        super(LassoRegression, self).__init__(name=name,
                                         learning_rate=learning_rate, 
                                         batch_size=batch_size, 
                                         theta_init=theta_init, 
                                         epochs=epochs, 
                                         regularizer=regularizer,
                                         early_stop=early_stop, 
                                         patience=patience, 
                                         precision=precision,
                                         cost=cost, 
                                         metric=metric,                                          
                                         val_size=val_size,                                          
                                         verbose=verbose, checkpoint=checkpoint, 
                                         random_state=random_state)    
        
