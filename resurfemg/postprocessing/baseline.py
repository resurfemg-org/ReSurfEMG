"""
Copyright 2024 University of Twente Licensed under the Apache License, version
 2.0. See LICENSE for details.

This file contains functions to calculate moving baselines from a filtered
EMG envelope.
"""
import numpy as np
import pandas as pd
from resurfemg.helper_functions.math_operations import derivative


def moving_baseline(
    signal_env,
    window_s,
    step_s,
    set_percentile=33,
    omit_nan=False,
):
    """
    This function calculates a moving baseline from a envelope data in
    accordance with Graßhoff et al. (2021)
    ---------------------------------------------------------------------------
    :param emg_env: envelope signal
    :type emg_env: ~numpy.ndarray
    :param window_s: window length in samples
    :type window_s: int
    :param step_s: number of consecutive samples with the same baseline value
    :type step_s: int
    :param: set_percentile
    :type: numpy percentile

    :returns rolling_baseline: The moving baseline for the signal envelope
    :rtype rolling_baseline: numpy.ndarray
    """
    rolling_baseline = np.zeros((len(signal_env), ))

    for idx in range(0, len(signal_env), step_s):
        start_i = max([0, idx-int(window_s/2)])
        end_i = min([len(signal_env), idx + int(window_s/2)])
        if omit_nan:
            baseline_value_y = np.nanpercentile(
                signal_env[start_i:end_i], set_percentile)
        else:
            baseline_value_y = np.percentile(
                signal_env[start_i:end_i], set_percentile)

        for i in range(idx, min([idx + step_s, len(signal_env)])):
            rolling_baseline[i] = baseline_value_y
    return rolling_baseline


def slopesum_baseline(
    signal_env,
    window_s,
    step_s,
    fs,
    set_percentile=33,
    augm_percentile=25,
    ma_window=None,
    perc_window=None,
):
    """
    This function calculates the augmented version of the moving baseline over
    a signal envelope, using a slope sum.
    ---------------------------------------------------------------------------
    :param signal_env: envelope signal
    :type signal_env: ~numpy.ndarray
    :param window_s: window length in seconds
    :type window_s: int
    :param step_s: number of consecutive samples with the same baseline value
    :type step_s: int
    :param emg_fs: emg sampling rating
    :type emg_fs: int
    :param set_percentile
    :type set_percentile: float (0-100)
    :param ma_window: moving average window in samples for average dy/dt
    :type ma_window: int
    :param perc_window: number of consecutive samples with the same
    baseline value
    :type perc_window: int

    :returns _slopesum_baseline: The slopesum baseline for the signal envelope
    :rtype: numpy.ndarray
    :returns y_baseline_mean: The running mean baseline of the baseline
    :rtype: numpy.ndarray
    :returns y_baseline_std: The running standard deviation of the baseline
    :rtype: numpy.ndarray
    :returns y_baseline_series: The running baseline series
    :rtype: pandas.Series
    """
    if ma_window is None:

        ma_window = fs//2

    if perc_window is None:
        perc_window = fs

    # 1. call the Graßhoff version function for moving baseline
    rolling_baseline = moving_baseline(
        signal_env,
        window_s,
        step_s,
        set_percentile)

    # 2. Calculate the augmented moving baseline for the signal_env data
    # 2.a. Rolling standard deviation and mean over provided window length
    y_baseline_series = pd.Series(rolling_baseline)
    y_baseline_std = y_baseline_series.rolling(window_s,
                                               min_periods=1,
                                               center=True).std().values
    y_baseline_mean = y_baseline_series.rolling(window_s,
                                                min_periods=1,
                                                center=True).mean().values

    # 2.b. Augmented signal: signal_env + abs([dsignal_env/dt]_smoothed)
    dy_dt = derivative(signal_env - rolling_baseline, fs, ma_window)
    y_aug = signal_env[:-1] + np.abs(dy_dt)

    # 2.c. Run the moving median filter over the augmented signal to obtain
    #       the baseline
    _slopesum_baseline = np.zeros((len(signal_env), ))

    for idx in range(0, len(signal_env), perc_window):
        start_i = max([0, idx-int(window_s)])
        end_i = min([len(signal_env)-1, idx + int(window_s)])

        baseline_value_y = np.nanpercentile(
            y_aug[start_i:end_i], augm_percentile)
        for i in range(idx, min([idx+int(perc_window), len(signal_env)-1])):
            _slopesum_baseline[i] = 1.2 * baseline_value_y

    _slopesum_baseline[i+1] = _slopesum_baseline[i]
    return (_slopesum_baseline, y_baseline_mean,
            y_baseline_std, y_baseline_series)
