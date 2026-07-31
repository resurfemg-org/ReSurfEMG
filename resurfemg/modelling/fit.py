"""
Copyright 2026 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for checking the fit of linear models.
"""

import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error


def explained_variance(y_true, y_pred):
    """
    Calculate the explained variance score
    :param y_true: The true output values.
    :type y_true: ~numpy.ndarray
    :param y_pred: The predicted output values.
    :type y_pred: ~numpy.ndarray
    """
    # Propotion of variance explained
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)


def residual_standard_error(y_true, y_pred, p):
    """
    Calculate the residual standard error (RSE).
    :param y_true: The true output values.
    :type y_true: ~numpy.ndarray
    :param y_pred: The predicted output values.
    :type y_pred: ~numpy.ndarray
    :param p: The number of predictors in the model.
    :type p: int
    """
    # Standard deviation of residuals
    return np.sqrt(np.sum((y_true - y_pred) ** 2)/(len(y_true) - p))


def check_model_fit(
        mdl,
        x_input,
        y_true,
        verbose=True,
        include_constant_term=False):
    """
    Check the fit of a linear model by calculating the coefficient of
    determination (R^2), mean absolute error (MAE), and residual standard error
    (RSE).
    :param mdl: The fitted linear model.
    :type mdl: ~statsmodels.regression.linear_model.RegressionResultsWrapper
    :param x_input: The input features for the model.
    :type x_input: ~numpy.ndarray
    :param y_true: The true output values.
    :type y_true: ~numpy.ndarray
    """
    if include_constant_term:
        x_input = sm.add_constant(x_input, has_constant='add')
    y_pred = mdl.predict(x_input)
    res = y_pred - y_true

    r2 = np.corrcoef(y_true, y_pred)[0, 1]**2
    cod = explained_variance(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    p = x_input.shape[1] if len(x_input.shape) > 1 else 1
    rse = residual_standard_error(y_true, y_pred, p)

    if verbose:
        print(f"CoD (R^2) (prop of variance explained): {cod*100:.2f} %")
        print(f"Mean absolute error: {mae:.2f}")
        print(f"RSE (STD of residuals): {rse:.2f}")

    return r2, cod, rse, res
