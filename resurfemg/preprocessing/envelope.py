"""Envelope calculation module.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions extract the envelopes from EMG arrays.
"""

import numpy as np
import pandas as pd
from scipy import stats


def full_rolling_rms(emg_clean: np.ndarray, window_length: int) -> np.ndarray:
    """Compute RMS envelope on clean EMG (full).

    This function computes a root mean squared envelope over an
    array cleaned (filtered and ECG eliminated) EMG.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarrayx
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_rms: The root-mean-squared EMG sample data
    :rtype emg_rms: ~numpy.ndarray[float]
    """
    emg_clean_sqr = pd.Series(np.power(emg_clean, 2))
    return np.sqrt(
        emg_clean_sqr.rolling(window=window_length, min_periods=1, center=True).mean()
    )


def naive_rolling_rms(emg_clean: np.ndarray, window_length: int) -> np.ndarray:
    """Compute RMS envelope on clean EMG (naive).

    This function computes a root mean squared envelope over an
    array `emg_clean`.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_rms: : The root-mean-squared EMG sample data
    :rtype emg_rms: ~numpy.ndarray[float]
    """
    x_c = np.cumsum(abs(emg_clean) ** 2)
    return np.sqrt((x_c[window_length:] - x_c[:-window_length]) / window_length)


def full_rolling_arv(emg_clean: np.ndarray, window_length: int) -> np.ndarray:
    """Compute ARV envelope on clean EMG.

    This function computes an average rectified value envelope over an
    array `emg_clean`.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray
    :param window_length: Length of the sample use as window for function
    :type window_length: int

    :returns emg_arv: The average rectified value EMG sample data
    :rtype emg_arv: ~numpy.ndarray[float]
    """
    emg_clean_abs = pd.Series(np.abs(emg_clean))
    return np.asarray(
        emg_clean_abs.rolling(window=window_length, min_periods=1, center=True)
        .mean()
        .values
    )


def rolling_rms_ci(
    emg_clean: np.ndarray, window_length: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate RMS confidence interval.

    This function estimates the confidence interval for each window of the RMS.
    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray[float]
    :param window_length: Length of the sample used as window for function
    :type window_length: int
    :param alpha: Significance level for the confidence interval
    :type alpha: float

    :returns lower_ci: Lower bound of the confidence interval
    :rtype lower_ci: ~numpy.ndarray[float]
    :returns upper_ci: Upper bound of the confidence interval
    :rtype upper_ci: ~numpy.ndarray[float]
    """
    emg_clean_sqr = pd.Series(np.power(emg_clean, 2))
    emg_ms = (
        emg_clean_sqr.rolling(window=window_length, min_periods=1, center=True)
        .mean()
        .values
    )
    emg_sem = (
        emg_clean_sqr.rolling(window=window_length, min_periods=1, center=True)
        .sem()
        .values
    )
    # Calculate the confidence interval
    confidence_level = 1 - alpha
    df = window_length - 1
    ci = stats.t.interval(confidence_level, df, emg_ms, emg_sem)

    lower_ci = np.sqrt(ci[0])
    upper_ci = np.sqrt(ci[1])

    return lower_ci, upper_ci


def rolling_arv_ci(
    emg_clean: np.ndarray, window_length: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """This function estimates the confidence interval for each window.

    ---------------------------------------------------------------------------
    :param emg_clean: Samples from the EMG
    :type emg_clean: ~numpy.ndarray[float]
    :param window_length: Length of the sample used as window for function
    :type window_length: int
    :param alpha: Significance level for the confidence interval
    :type alpha: float

    :returns lower_ci: Lower bound of the confidence interval
    :rtype lower_ci: ~numpy.ndarray[float]
    :returns upper_ci: Upper bound of the confidence interval
    :rtype upper_ci: ~numpy.ndarray[float]
    """
    emg_clean_abs = pd.Series(np.abs(emg_clean))
    emg_arv = (
        emg_clean_abs.rolling(window=window_length, min_periods=1, center=True)
        .mean()
        .values
    )
    emg_sem = (
        pd.Series(emg_clean)
        .rolling(window=window_length, min_periods=1, center=True)
        .sem()
        .values
    )

    # Calculate the confidence interval
    confidence_level = 1 - alpha
    df = window_length - 1
    ci = stats.t.interval(confidence_level, df, emg_arv, emg_sem)

    lower_ci = ci[0]
    upper_ci = ci[1]

    return lower_ci, upper_ci
