"""Evaluation of signal quality and peak quality for EMG signals.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to determine peak and signal quality from
preprocessed EMG arrays.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import curve_fit

import resurfemg.helper_functions.math_operations as mo
import resurfemg.postprocessing.features as feat

logger = logging.getLogger(__name__)


def snr_pseudo(
    src_signal: np.ndarray,
    peaks: list[int] | np.ndarray,
    baseline: np.ndarray,
    fs: int,
) -> np.ndarray:
    """Approximate the signal-to-noise ratio (SNR) of the signal.

    Approximate the signal-to-noise ratio (SNR) of the signal based
    on the peak height relative to the baseline.
    Args:
        src_signal (numpy.ndarray): Signal to evaluate.
        peaks (list[int]): List of individual peak indices.
        baseline (numpy.ndarray): Baseline signal to evaluate SNR to.
        fs (int): Sampling rate.

    Returns:
        numpy.ndarray: The SNR per peak.
    """
    peak_heights = np.zeros((len(peaks),))
    noise_heights = np.zeros((len(peaks),))

    for peak_nr, idx in enumerate(peaks):
        peak_heights[peak_nr] = src_signal[idx]
        start_idx = max([0, idx - fs])
        end_i = min([len(src_signal), idx + fs])
        noise_heights[peak_nr] = np.median(baseline[start_idx:end_i])

    return np.divide(peak_heights, noise_heights)


def pocc_quality(
    p_vent_signal: np.ndarray,
    pocc_peaks: np.ndarray,
    pocc_ends: np.ndarray,
    ptp_occs: np.ndarray,
    dp_up_10_threshold: float = 0.0,
    dp_up_90_threshold: float = 2.0,
    dp_up_90_norm_threshold: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the quality of occlusion pressure (Pocc) peaks.

    Evaluation of occlusion pressure (Pocc) quality, in accordance with Warnaar
    et al. (2024). Poccs are labelled invalid if too many negative deflections
    happen in the upslope (first decile < 0), or if the upslope is to steep
    (high absolute or relative 9th decile), indicating occlusion release before
    the patient's inspiriratory effort has ended.
    Args:
        p_vent_signal (numpy.ndarray): Airway pressure signal.
        pocc_peaks (numpy.ndarray): List of individual peak indices.
        pocc_ends (numpy.ndarray): List of individual peak end indices.
        ptp_occs (numpy.ndarray): List of pressure-time products for each occlusion.
        dp_up_10_threshold (float): Minimum first decile of upslope after the
            (negative) occlusion pressure peak.
        dp_up_90_threshold (float): Maximum 9th decile of upslope after the
            (negative) occlusion pressure peak.
        dp_up_90_norm_threshold (float): Maximum normalised 9th decile of upslope
            after the (negative) occlusion pressure peak.

    Returns:
        tuple:
            - numpy.ndarray: Boolean list of valid Pocc peaks.
            - numpy.ndarray: Matrix of the calculated criteria.
    """
    dp_up_10 = np.zeros((len(pocc_peaks),))
    dp_up_90 = np.zeros((len(pocc_peaks),))
    dp_up_90_norm = np.zeros((len(pocc_peaks),))
    for idx, pocc_peak in enumerate(pocc_peaks):
        end_i = pocc_ends[idx]
        dp = p_vent_signal[pocc_peak + 1 : end_i] - p_vent_signal[pocc_peak : end_i - 1]
        dp_up_10[idx] = np.percentile(dp, 10)
        dp_up_90[idx] = np.percentile(dp, 90)
        dp_up_90_norm[idx] = dp_up_90[idx] / np.sqrt(ptp_occs[idx])

    criteria_matrix = np.array([dp_up_10, dp_up_90, dp_up_90_norm])
    criteria_bool_matrix = np.array(
        [
            dp_up_10 <= dp_up_10_threshold,
            dp_up_90 > dp_up_90_threshold,
            dp_up_90_norm > dp_up_90_norm_threshold,
        ]
    )
    valid_poccs = ~np.any(criteria_bool_matrix, axis=0)
    return valid_poccs, criteria_matrix


def interpeak_dist(
    ecg_peak_idxs: np.ndarray, emg_peak_idxs: np.ndarray, threshold: float = 1.1
) -> bool:
    """Calculate the interpeak distance ratio of ECG and EMG peaks.

    Calculate the median interpeak distances for ECG and EMG and check if their
    ratio is above the given threshold, i.e. if cardiac frequency is higher
    than respiratory frequency (True)
    Args:
        ecg_peak_idxs (numpy.ndarray): Indices of ECG peaks.
        emg_peak_idxs (numpy.ndarray): Indices of EMG peaks.
        threshold (float): The threshold value to compare against. Default is 1.1.

    Returns:
        bool: Boolean value indicating if the interpeak distance is valid.
    """
    # Calculate median interpeak distance for ECG
    t_delta_ecg_med = np.median(
        np.array(ecg_peak_idxs[1:]) - np.array(ecg_peak_idxs[:-1])
    )
    # # Calculate median interpeak distance for EMG
    t_delta_emg_med = np.median(
        np.array(emg_peak_idxs[1:]) - np.array(emg_peak_idxs[:-1])
    )
    # Check if each median interpeak distance is above the threshold
    t_delta_relative = t_delta_emg_med / t_delta_ecg_med

    return bool(t_delta_relative >= threshold)


def percentage_under_baseline(
    signal: np.ndarray,
    fs: int,
    peak_idxs: list[int] | np.ndarray,
    start_idxs: list[int] | np.ndarray,
    end_idxs: list[int] | np.ndarray,
    baseline: np.ndarray,
    aub_window_s: int | None = None,
    ref_signal: np.ndarray | None = None,
    aub_threshold: float = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the percentage of the area under the baseline for each peak.

    Calculate the percentage area under the baseline, in accordance with
    Warnaar et al. (2024).
    Args:
        signal (numpy.ndarray): Signal in which the peaks are detected.
        fs (int): Sampling frequency.
        peak_idxs (list[int]): List of individual peak indices.
        start_idxs (list[int]): List of individual peak start indices.
        end_idxs (list[int]): List of individual peak end indices.
        baseline (numpy.ndarray): Running baseline of the signal.
        aub_window_s (int, optional): Number of samples before and after peak_idxs
            to look for the nadir.
        ref_signal (numpy.ndarray, optional): Signal in which the nadir is searched.
        aub_threshold (float): Maximum AUB error percentage for a peak.

    Returns:
        tuple:
            - numpy.ndarray: Boolean list of valid time products.
            - numpy.ndarray: List of calculated AUB percentages.
            - numpy.ndarray: Reference signal nadir values.
    """
    if aub_window_s is None:
        aub_window_s = 5 * fs

    if ref_signal is None:
        ref_signal = signal

    time_products = feat.time_product(
        signal,
        fs,
        start_idxs,
        end_idxs,
        baseline,
    )
    aubs, y_refs = feat.area_under_baseline(
        signal,
        fs,
        peak_idxs,
        start_idxs,
        end_idxs,
        aub_window_s,
        baseline,
        ref_signal=signal,
    )

    percentages_aub = aubs / (time_products + aubs) * 100
    valid_timeproducts = percentages_aub <= aub_threshold

    return valid_timeproducts, percentages_aub, y_refs


def detect_local_high_aub(
    aubs: np.ndarray, threshold_percentile: float = 75.0, threshold_factor: float = 4.0
) -> np.ndarray:
    """Detect local upward deflections in the area under the baseline.

    Args:
        aubs (numpy.ndarray): List of area under the baseline values. See
            resurfemg.postprocessing.features.area_under_baseline.
        threshold_percentile (float): Percentile for detecting high baseline.
        threshold_factor (float): Multiplication factor for threshold_percentile.

    Returns:
        numpy.ndarray: Boolean list of AUB values under threshold.
    """
    threshold = threshold_factor * np.percentile(aubs, threshold_percentile)
    return aubs < threshold


def detect_extreme_time_products(
    time_products: np.ndarray,
    upper_percentile: float = 95.0,
    upper_factor: float = 10.0,
    lower_percentile: float = 5.0,
    lower_factor: float = 0.1,
) -> np.ndarray:
    """Detect extreme (high or low) time product values.

    Args:
        time_products (numpy.ndarray): List of time product values.
        upper_percentile (float): Percentile for detecting high time products.
        upper_factor (float): Multiplication factor for upper_percentile.
        lower_percentile (float): Percentile for detecting low time products.
        lower_factor (float): Multiplication factor for lower_percentile.

    Returns:
        numpy.ndarray: Boolean list of time product values within bounds.
    """
    upper_threshold = upper_factor * np.percentile(time_products, upper_percentile)
    lower_threshold = lower_factor * np.percentile(time_products, lower_percentile)
    return np.all(
        np.array([time_products < upper_threshold, time_products > lower_threshold]),
        axis=0,
    )


def detect_non_consecutive_manoeuvres(
    ventilator_breath_idxs: np.ndarray, manoeuvres_idxs: np.ndarray
) -> np.ndarray:
    """Detect non-consecutive manoeuvres.

    Detect manoeuvres (for example Pocc) with no supported breaths
    in between. Input are the ventilator breaths, to be detected with the
    function post_processing.event_detecton.detect_supported_breaths
    If no supported breaths are detected in between two manoeuvres,
    valid_manoeuvres is 'True'.
    Note: fs of both signals should be equal.
    Args:
        ventilator_breath_idxs (numpy.ndarray): List of supported breath indices.
        manoeuvres_idxs (numpy.ndarray): List of manoeuvres indices.

    Returns:
        numpy.ndarray: Boolean list of valid manoeuvres.
    """
    consecutive_manoeuvres = np.zeros(len(manoeuvres_idxs), dtype=bool)
    for idx, _ in enumerate(manoeuvres_idxs):
        if idx > 0:
            # Check for supported breaths in between two Poccs
            intermediate_breaths = np.equal(
                (manoeuvres_idxs[idx - 1] < ventilator_breath_idxs),
                (ventilator_breath_idxs < manoeuvres_idxs[idx]),
            )

            # If no supported breaths are detected in between, a
            # 'double dip' is detected
            intermediate_breath_count = np.sum(intermediate_breaths)
            if intermediate_breath_count > 0:
                consecutive_manoeuvres[idx] = False
            else:
                consecutive_manoeuvres[idx] = True
        else:
            consecutive_manoeuvres[idx] = False

    return np.logical_not(consecutive_manoeuvres)


def evaluate_bell_curve_error(
    peak_idxs: np.ndarray,
    start_idxs: np.ndarray,
    end_idxs: np.ndarray,
    signal: np.ndarray,
    fs: int,
    time_products: np.ndarray,
    bell_window_s: int | None = None,
    bell_threshold: float = 40,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the bell-curve error of signal peaks.

    Calculate the bell-curve error of signal peaks, in accordance with Warnaar
    et al. (2024).
    Args:
        peak_idxs (numpy.ndarray): List of peak indices.
        start_idxs (numpy.ndarray): List of onset indices.
        end_idxs (numpy.ndarray): List of offset indices.
        signal (numpy.ndarray): Filtered signal.
        fs (int): Sampling rate.
        time_products (numpy.ndarray): List of area under the curves per peak.
        bell_window_s (int, optional): Number of samples before and after peak_idxs
            to look for the nadir.
        bell_threshold (float): Maximum bell error percentage for a valid peak.

    Returns:
        tuple:
            - numpy.ndarray: Boolean list of valid peaks.
            - numpy.ndarray: Calculated bell errors.
            - numpy.ndarray: Calculated bell errors in percentage.
            - numpy.ndarray: Minimum value of the baseline.
            - numpy.ndarray: Fitted bell curve parameters.
    """
    if bell_window_s is None:
        bell_window_s = fs * 5
    t = np.array([i / fs for i in range(len(signal))])

    bell_error = np.zeros((len(peak_idxs),))
    percentage_bell_error = np.zeros((len(peak_idxs),))
    fitted_parameters = np.zeros((len(peak_idxs), 3))
    y_min = np.zeros((len(peak_idxs),))
    for idx, (peak_idx, start_idx, end_i, tp) in enumerate(
        zip(peak_idxs, start_idxs, end_idxs, time_products)
    ):
        baseline_start_idx = max(0, peak_idx - bell_window_s)
        baseline_end_i = min(len(signal) - 1, peak_idx + bell_window_s)
        y_min[idx] = np.min(signal[baseline_start_idx:baseline_end_i])

        plus_idx = (
            3 - (end_i - start_idx) if end_i - start_idx < 3 else 0  # noqa: PLR2004
        )

        x_data = t[start_idx : end_i + 1 + plus_idx]
        y_data = signal[start_idx : end_i + 1 + plus_idx] - y_min[idx]

        if (
            np.any(np.isnan(x_data))
            or np.any(np.isnan(y_data))
            or np.any(np.isinf(x_data))
            or np.any(np.isinf(y_data))
        ):
            msg = f"NaNs or Infs detected in data for peak index {idx}. Skipping this peak."
            logger.info(msg)
            bell_error[idx] = np.nan
            continue

        try:
            popt, *_ = curve_fit(
                mo.bell_curve,
                x_data,
                y_data,
                bounds=(
                    [0.0, t[peak_idx] - 0.5, 0.0],
                    [np.inf, t[peak_idx] + 0.5, np.inf],
                ),
            )
        except RuntimeError as e:
            msg = f"Curve fitting failed for peak index {idx} with error: {e}"
            logger.exception(msg)
            bell_error[idx] = np.nan
            popt = np.array([np.nan, np.nan, np.nan])
            continue

        bell_error[idx] = trapezoid(
            np.abs(
                signal[start_idx : end_i + 1]
                - (mo.bell_curve(t[start_idx : end_i + 1], *popt) + y_min[idx])
            ),
            dx=1 / fs,
        )
        percentage_bell_error[idx] = bell_error[idx] / tp * 100
        fitted_parameters[idx, :] = popt

    valid_peak = percentage_bell_error <= bell_threshold

    return (valid_peak, bell_error, percentage_bell_error, y_min, fitted_parameters)


def evaluate_event_timing(
    t_events_1: np.ndarray,
    t_events_2: np.ndarray,
    delta_min: float = 0,
    delta_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the timing of two sets of events.

    Evaluate whether the timing of the events in `t_events_1` preceeds the
    events in `t_events_2` minimally by `delta_min` and maximally by
    `delta_max`. `t_events_1` and `t_events_2` should be the same length.
    Args:
        t_events_1 (numpy.ndarray): Timing of the events that should happen first.
        t_events_2 (numpy.ndarray): Timing of the events that should happen second.
        delta_min (float): The minimum time event 1 should precede event 2.
        delta_max (float, optional): The maximum time event 1 should precede event 2.

    Returns:
        tuple:
            - numpy.ndarray: Boolean list of correct timing.
            - numpy.ndarray: List of delta times between the events.
    """
    delta_time = np.array(t_events_2) - np.array(t_events_1)
    min_crit = delta_time >= delta_min
    if delta_max is not None:
        max_crit = delta_time <= delta_max
        correct_timing = np.all(np.array([min_crit, max_crit]), axis=0)
    else:
        correct_timing = min_crit
    return correct_timing, delta_time


def evaluate_respiratory_rates(
    emg_breath_idxs: np.ndarray, t_emg: float, rr_vent: float, min_fraction: float = 0.1
) -> tuple[float, bool]:
    """Evaluate the respiratory rate of detected EMG breaths relative to the ventilatory respiratory rate.

    This function evaluates fraction of detected EMG breaths relative to the
    ventilatory respiratory rate.
    Args:
        emg_breath_idxs (numpy.ndarray): EMG breath indices.
        t_emg (float): Recording time in seconds.
        rr_vent (float): Ventilatory respiratory rate (breath/min).
        min_fraction (float): Required minimum detected fraction of EMG breaths.

    Returns:
        tuple:
            - float: Fraction of detected EMG breaths.
            - bool: Boolean indicating if the fraction is above the minimum.
    """
    detected_fraction = float(len(emg_breath_idxs) / (rr_vent * t_emg / 60))
    criterium_met = detected_fraction >= min_fraction
    return (detected_fraction, criterium_met)
