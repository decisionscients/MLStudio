#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# =========================================================================== #
# Project : ML Studio                                                         #
# Version : 0.1.0                                                             #
# File    : validation.py                                                     #
# Python  : 3.8.2                                                             #
# --------------------------------------------------------------------------  #
# Author  : John James                                                        #
# Company : DecisionScients                                                   #
# Email   : jjames@decisionscients.com                                        #
# URL     : https://github.com/decisionscients/MLStudio                       #
# --------------------------------------------------------------------------  #
# Created       : Thursday, May 14th 2020, 1:33:31 pm                         #
# Last Modified : Thursday, May 14th 2020, 1:35:41 pm                         #
# Modified By   : John James (jjames@decisionscients.com)                     #
# --------------------------------------------------------------------------  #
# License : BSD                                                               #
# Copyright (c) 2020 DecisionScients                                          #
# =========================================================================== #
"""Functions used to validate the state, parameters or data of an estimator."""
import numpy as np

# --------------------------------------------------------------------------  #
def check_y(y):
    """Returns the number of outputs.

    This is to be used to obtain the number of classes for classification
    algorithms. 
    
    Parameters
    ----------
    y : nd_array
        The y values from the fit method.
    
    """
    y = np.array(y) 
    if y.ndim == 1:
        return 1
    else:
        return y.shape[1]

# --------------------------------------------------------------------------  #
def check_is_estimator(estimator):
    """Checks if the estimator is a valid Scikit-Learn estimator instance.

    This checks whether the estimator object is a valid instance 
    (not a class) and whether it has a fit method.

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    Returns
    -------
    None

    Raises
    ------
    TypeError if not a valid estimator

    """

    if isclass(estimator):
        raise TypeError("{} is a class, not an instance.".format(estimator))    

    if not hasattr(estimator, 'fit'):
        raise TypeError("%s is not an estimator instance." % (estimator))

# --------------------------------------------------------------------------  #
def check_is_fitted(estimator, attributes=None, *, msg=None):
    """Determines whether an estimator has been fitted.
    
    It checks if the estimator is fitted by verifying the presence of
    fitted attributes (ending with a trailing underscore). This utility 
    is to be used in estimator's predict methods.

    This function was adapted from the Scikit-Learn function of the 
    same name on May 14th, 2020.
    https://github.com/scikit-learn/scikit-learn/sklearn/utils/validation.py
        

    Parameters
    ----------
    estimator : estimator instance.
        estimator instance for which the check is performed.

    attributes : str, list or tuple of str, default=None
        Attribute name(s) given as string or a list/tuple of strings
        Eg.: ``["coef_", "estimator_", ...], "coef_"``
        If `None`, `estimator` is considered fitted if there exist an
        attribute that ends with a underscore and does not start with double
        underscore.

    msg : string
        The default error message is, "This %(name)s instance is not fitted
        yet. Call 'fit' with appropriate arguments before using this
        estimator."
        For custom messages if "%(name)s" is present in the message string,
        it is substituted for the estimator name.
        Eg. : "Estimator, %(name)s, must be fitted before sparsifying".
    
    Returns
    -------
    None
    
    Raises
    ------
    NotFittedError
        If the attributes are not found.
    """
    check_is_estimator(estimator)

    if msg is None:
        msg = ("This %(name)s instance is not fitted yet. Call 'fit' with "
               "appropriate arguments before using this estimator.")

    if attributes is not None:
        if not isinstance(attributes, (list, tuple)):
            attributes = [attributes]
        attrs = any([hasattr(estimator, attr) for attr in attributes])
    else:
        attrs = [v for v in vars(estimator)
                 if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError(msg % {'name': type(estimator).__name__})
