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

    Args:
        emg_env (numpy.ndarray): A single lead EMG envelope.
        start_idxs (list[int]): List of individual peak start indices.
        end_idxs (list[int]): List of individual peak end indices.

    Returns:
        tuple:
            - numpy.ndarray: Absolute time-to-peak.
            - numpy.ndarray: Relative time-to-peak.
    """
    start_idxs_np = np.array(start_idxs)
    end_idxs_np = np.array(end_idxs)
    abs_times = np.zeros(start_idxs_np.shape)
    percent_times = np.zeros(start_idxs_np.shape)
    for idx, (start_idx, end_idx) in enumerate(zip(start_idxs_np, end_idxs_np, strict=False)):
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

    Args:
        emg_env (numpy.ndarray): A single lead EMG envelope.
        start_idxs (list[int]): List of individual peak start indices.
        end_idxs (list[int]): List of individual peak end indices.
        smoothing (bool): Whether to apply smoothing before calculations.

    Returns:
        numpy.ndarray: Initial slope of each peak.
    """
    start_idxs_np = np.array(start_idxs)
    end_idxs_np = np.array(end_idxs)
    pseudoslopes = np.zeros(start_idxs_np.shape)
    for idx, (start_idx, end_idx) in enumerate(zip(start_idxs_np, end_idxs_np, strict=False)):
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
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate the peak height relative to the baseline.

    Calculate the peak height of signal and the baseline for the windows
    at the peak_idxs relative to the baseline. If no baseline is provided, the
    peak height relative to zero is determined.

    Args:
        signal (numpy.ndarray): Signal to determine the peak heights in.
        peak_idxs (numpy.ndarray): List of individual peak indices.
        baseline (numpy.ndarray, optional): Running baseline of the signal.

    Returns:
        numpy.ndarray: List of peak amplitudes.
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)
    return np.array(signal[peak_idxs] - baseline[peak_idxs])


def time_product(
    signal: np.ndarray,
    fs: int,
    start_idxs: list[int] | np.ndarray,
    end_idxs: list[int] | np.ndarray,
    baseline: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate the time product between the signal and the baseline.

    Calculate the time product between the signal and the baseline for the
    windows defined by the start_idx and end_idx sample pairs.

    Args:
        signal (numpy.ndarray): Signal to calculate the time product over.
        fs (int): Sampling frequency.
        start_idxs (list[int]): List of individual peak start indices.
        end_idxs (list[int]): List of individual peak end indices.
        baseline (numpy.ndarray, optional): Running baseline of the signal.

    Returns:
        numpy.ndarray: The calculated time products.
    """
    if baseline is None:
        baseline = np.zeros(signal.shape)

    time_products = np.zeros(np.asarray(start_idxs).shape)
    for idx, (start_idx, end_idx) in enumerate(zip(start_idxs, end_idxs, strict=False)):
        y_delta = signal[start_idx : end_idx + 1] - baseline[start_idx : end_idx + 1]
        if not np.all(np.sign(y_delta[1:]) >= 0) and not np.all(np.sign(y_delta[1:]) <= 0):
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
    baseline: np.ndarray,
    ref_signal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the area under the baseline.

    Calculate the time product between the baseline and the nadir of the
    reference signal in the aub_window_s for the windows defined by the
    start_idx and end_idx sample pairs.

    Args:
        signal (numpy.ndarray): Signal to calculate the time product over.
        fs (int): Sampling frequency.
        peak_idxs (list[int]): List of individual peak indices.
        start_idxs (list[int]): List of individual peak start indices.
        end_idxs (list[int]): List of individual peak end indices.
        aub_window_s (int): Number of samples before and after peak_idxs to look
            for the nadir.
        baseline (numpy.ndarray): Running baseline of the signal.
        ref_signal (numpy.ndarray, optional): Signal in which the nadir is searched.

    Returns:
        tuple:
            - numpy.ndarray: The calculated areas under the baseline.
            - numpy.ndarray: The reference signal nadir values.
    """
    if ref_signal is None:
        ref_signal = signal

    aubs = np.zeros(np.asarray(peak_idxs).shape)
    y_refs = np.zeros(np.asarray(peak_idxs).shape)
    for idx, (start_idx, peak_idx, end_idx) in enumerate(zip(start_idxs, peak_idxs, end_idxs, strict=False)):
        y_delta_curve = signal[start_idx : end_idx + 1] - baseline[start_idx : end_idx + 1]
        ref_start_idx = max([0, peak_idx - aub_window_s])
        ref_end_idx = min([len(signal) - 1, peak_idx + aub_window_s])
        if not np.all(np.sign(y_delta_curve[1:]) >= 0) and not np.all(np.sign(y_delta_curve[1:]) <= 0):
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

    Args:
        breath_idxs (numpy.ndarray): Breath indices.
        fs (int): Sampling frequency.
        outlier_percentile (float): Respiratory rate outlier percentile.
        outlier_factor (float): Respiratory rate outlier factor.

    Returns:
        tuple:
            - float: Median respiratory rate.
            - numpy.ndarray: Breath-to-breath respiratory rate.
    """
    breath_interval = np.array(breath_idxs[1:]) - np.array(breath_idxs[:-1])
    rr_b2b = 60 * fs / breath_interval
    outlier_threshold = outlier_factor * np.percentile(rr_b2b, outlier_percentile)
    rr_b2b[rr_b2b > outlier_threshold] = np.nan
    rr_median = float(np.nanmedian(rr_b2b))

    return rr_median, rr_b2b
