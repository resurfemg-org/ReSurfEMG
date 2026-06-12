"""This file contains functions to extract features from preprocessed EMG arrays.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.integrate import trapezoid

from resurfemg.helper_functions.math_operations import running_smoother


def time_to_peak(
    emg_env: np.ndarray,
    start_idxs: list[int],
    end_idxs: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the absolute and relative time to peak.

    :param emg_env: an single lead EMG envelope
    :type emg_env: np.array
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~[int]
    :param end_idxs: list of individual peak end indices
    :type end_idxs: ~[int]

    :returns abs_times: absolute time-to_peak
    :rtype abs_times: numpy.ndarray[int]
    :returns percent_times: relative time-to_peak
    :rtype percent_times: numpy.ndarray[float]
    """
    start_idxs_np = np.array(start_idxs)
    end_idxs_np = np.array(end_idxs)
    abs_times = np.zeros(start_idxs_np.shape)
    percent_times = np.zeros(start_idxs_np.shape)
    for idx, (start_idx, end_idx) in enumerate(zip(start_idxs_np, end_idxs_np)):
        breath_arc = emg_env[start_idx:end_idx]
        smoothed_breath = running_smoother(breath_arc)
        abs_times[idx] = smoothed_breath.argmax()
        percent_times[idx] = abs_times[idx] / len(breath_arc)

    return abs_times, percent_times


def pseudo_slope(
    emg_env: np.ndarray,
    start_idxs: list[int],
    end_idxs: list[int],
    smoothing: bool = True,
) -> np.ndarray:
    """Calculates the pseudo-slope of the take-off angle of the EMG signal.

    This is a function to get the shape/slope of the take-off angle of the
    EMG signal. The slope is returned in units/samples (in abs values), not
    true slope. The true slope will depend on sampling rate and pre-
    processing. Therefore, only within sample comparison is recommended.
    ---------------------------------------------------------------------------
    :param emg_env: an single lead EMG envelope
    :type emg_env: np.array
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~[int]
    :param end_idxs: list of individual peak end indices
    :type end_idxs: ~[int]
    :param smoothing: smoothing which can or can not run before calculations
    :type smoothing: bool

    :returns pseudoslope: initial slope of the peak
    :rtype pseudoslope: np.ndarray[float]
    """
    start_idxs_np = np.array(start_idxs)
    end_idxs_np = np.array(end_idxs)
    pseudoslopes = np.zeros(start_idxs_np.shape)
    for idx, (start_idx, end_idx) in enumerate(zip(start_idxs_np, end_idxs_np)):
        breath_arc = emg_env[start_idx:end_idx]
        pos_arc = abs(breath_arc)
        if smoothing:
            smoothed_breath = running_smoother(pos_arc)
            abs_time = smoothed_breath.argmax()
        else:
            abs_time = pos_arc.argmax()
        abs_height = pos_arc[abs_time]
        pseudoslopes[idx] = abs_height / abs_time
    return pseudoslopes


def amplitude(
    signal: np.ndarray,
    peak_idxs: np.ndarray,
    baseline: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Calculate the peak height relative to the baseline.

    Calculate the peak height of signal and the baseline for the windows
    at the peak_idxs relative to the baseline. If no baseline is provided, the
    peak height relative to zero is determined.
    ---------------------------------------------------------------------------
    :param signal: signal to determine the peak heights in
    :type signal: ~numpy.ndarray[float]
    :param peak_idxs: list of individual peak start indices
    :type peak_idxs: ~np.ndarray[int]
    :param baseline: running baseline of the signal
    :type baseline: ~numpy.ndarray[float]

    :returns amplitudes: list of peak amplitudes
    :rtype amplitudes: ~np.ndarray[float]
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)
    return np.array(signal[peak_idxs] - baseline[peak_idxs])


def time_product(
    signal: np.ndarray,
    fs: int,
    start_idxs: list[int] | np.ndarray,
    end_idxs: list[int] | np.ndarray,
    baseline: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Calculate the time product between the signal and the baseline.

    Calculate the time product between the signal and the baseline for the
    windows defined by the start_idx and end_idx sample pairs.
    ---------------------------------------------------------------------------
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray[float]
    :param fs: sampling frequency
    :type fs: ~int
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~list[int]
    :param end_idxs: list of individual peak end indices
    :type end_idxs: ~list[int]
    :param baseline: running Baseline of the signal
    :type baseline: ~numpy.ndarray[float]

    :returns time_products: the calculated time products
    :rtype time_products: numpy.ndarray[float]
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)

    time_products = np.zeros(np.asarray(start_idxs).shape)
    for idx, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs)):
        y_delta = signal[start_idx : end_idx + 1] - baseline[start_idx : end_idx + 1]
        if not np.all(np.sign(y_delta[1:]) >= 0) and not np.all(
            np.sign(y_delta[1:]) <= 0
        ):
            warnings.warn(
                "Warning: Curve for peak idx"
                + str(idx)
                + " not entirely above or below baseline. The "
                + "calculated integrals will cancel out."
            )

        time_products[idx] = np.abs(trapezoid(y_delta, dx=1 / fs))

    return time_products


def area_under_baseline(
    signal: np.ndarray,
    fs: int,
    peak_idxs: list[int] | np.ndarray,
    start_idxs: list[int] | np.ndarray,
    end_idxs: list[int] | np.ndarray,
    aub_window_s: int,
    baseline: np.ndarray | tuple[np.ndarray, np.ndarray],
    ref_signal: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the area under the baseline.

    Calculate the time product between the baseline and the nadir of the
    reference signal in the aub_window_s for the windows defined by the
    start_idx and end_idx sample pairs.
    ---------------------------------------------------------------------------
    :param signal: signal to calculate the time product over
    :type signal: ~numpy.ndarray[float]
    :param fs: sampling frequency
    :type fs: ~int
    :param peak_idxs: list of individual peak indices
    :type peak_idxs: ~list[int]
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~list[int]
    :param end_idxs: list of individual peak end indices
    :type end_idxs: ~list[int]
    :param aub_window_s: number of samples before and after peak_idxs to look
    for the nadir
    :type aub_window_s: ~int
    :param baseline: running baseline of the signal
    :type baseline: ~numpy.ndarray[float]
    :param ref_signal: signal in which the nadir is searched
    :type ref_signal: ~numpy.ndarray[float]

    :returns aubs: the calculated areas under the baseline
    :rtype: numpy.ndarray[float]
    :returns y_ref: the reference signal
    :rtype: numpy.ndarray[float]
    """
    if ref_signal is None:
        ref_signal = signal

    aubs = np.zeros(np.asarray(peak_idxs).shape)
    y_refs = np.zeros(np.asarray(peak_idxs).shape)
    for idx, (start_idx, peak_idx, end_idx) in enumerate(
        zip(start_idxs, peak_idxs, end_idxs)
    ):
        y_delta_curve = (
            signal[start_idx : end_idx + 1] - baseline[start_idx : end_idx + 1]
        )
        ref_start_idx = max([0, peak_idx - aub_window_s])
        ref_end_idx = min([len(signal) - 1, peak_idx + aub_window_s])
        if not np.all(np.sign(y_delta_curve[1:]) >= 0) and not np.all(
            np.sign(y_delta_curve[1:]) <= 0
        ):
            warnings.warn(
                "Warning: Curve for peak idx"
                + str(idx)
                + " not entirely above or below baseline. The "
                + "calculated integrals will cancel out."
            )

        if np.median(np.sign(y_delta_curve[1:]) >= 0):
            # Positively deflected signal: Baseline below peak
            y_ref = min(ref_signal[ref_start_idx:ref_end_idx])
            y_delta = baseline[start_idx : end_idx + 1] - y_ref
        else:
            # Negatively deflected signal: Baseline above peak
            y_ref = max(ref_signal[ref_start_idx:ref_end_idx])
            y_delta = y_ref - baseline[start_idx : end_idx + 1]

        aubs[idx] = np.abs(trapezoid(y_delta, dx=1 / fs))
        y_refs[idx] = y_ref

    return aubs, y_refs


def respiratory_rate(
    breath_idxs: np.ndarray,
    fs: int,
    outlier_percentile: float = 33,
    outlier_factor: float = 3,
) -> tuple[float, np.ndarray]:
    """Estimate respiratory rate based on breath indices.

    Estimate respiratory rate based from breath indices. Breath-by-breath
    respiratory rate larger than the outlier_percentile * outlier_factor are
    excluded.
    ---------------------------------------------------------------------------
    :param breath_idxs: breath indices
    :type breath_idxs: ~numpy.ndarray[int]
    :param fs: sampling frequency
    :type fs: ~int
    :param outlier_percentile: Respiratory rate outlier percentile
    :type outlier_percentile: ~float
    :param outlier_percentile: Respiratory rate outlier factor
    :type outlier_percentile: ~float

    :returns rr_median: median respiratory rate
    :rtype: ~float
    :returns rr_b2b: breath-to-breath respiratory rate
    :rtype rr_b2b: numpy.ndarray[~float]

    """
    breath_interval = np.array(breath_idxs[1:]) - np.array(breath_idxs[:-1])
    rr_b2b = 60 * fs / breath_interval
    outlier_threshold = outlier_factor * np.percentile(rr_b2b, outlier_percentile)
    rr_b2b[rr_b2b > outlier_threshold] = np.nan
    rr_median = float(np.nanmedian(rr_b2b))

    return rr_median, rr_b2b
