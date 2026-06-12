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
    ---------------------------------------------------------------------------
    :param signal: Signal to evaluate
    :type signal: ~numpy.ndarray[float]
    :param peaks: list of individual peak indices
    :type gate_peaks: ~list[int]
    :param baseline: Baseline signal to evaluate SNR to.
    :type baseline: ~numpy.ndarray[float]
    :param fs: sampling rate
    :type fs: int

    :returns snr_peaks: the SNR per peak
    :rtype snr_peaks: numpy.ndarray[float]
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
    ---------------------------------------------------------------------------
    :param signal: Airway pressure signal
    :type signal: ~numpy.ndarray
    :param pocc_peaks: list of individual peak indices
    :type pocc_peaks: ~list
    :param pocc_ends: list of individual peak end indices
    :type pocc_ends: ~list
    :param dp_up_10_threshold: Minimum first decile of upslope after the
    (negative) occlusion pressure peak
    :type dp_up_10_threshold: ~float
    :param dp_up_90_threshold: Maximum 9th decile of upslope after the
    (negative) occlusion pressure peak
    :type dp_up_90_threshold: ~float
    :param dp_up_90_norm_threshold: Maximum 9th decile of upslope after the
    (negative) occlusion pressure peak
    :type dp_up_90_norm_threshold: ~float

    :returns valid_poccs: boolean list of valid Pocc peaks
    :rtype valid_poccs: numpy.ndarray[bool]
    :returns criteria_matrix: matrix of the calculated criteria
    :rtype criteria_matrix: numpy.ndarray
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
    ---------------------------------------------------------------------------
    :param ecg_peak_idxs: Indices of ECG peaks
    :type ecg_peak_idxs: ~numpy.ndarray
    :param emg_peak_idxs: Indices of EMG peaks
    :type emg_peak_idxs: ~numpy.ndarray
    :param threshold: The threshold value to compare against. Default is 1.1
    :type threshold: ~float

    :returns valid_interpeak: Boolean value if the interpeak distance is valid
    :rtype valid_interpeak: bool
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
    ---------------------------------------------------------------------------
    :param signal: signal in which the peaks are detected
    :type signal: ~numpy.ndarray
    :param fs: sampling frequency
    :type fs: ~int
    :param peak_idxs: list of individual peak indices
    :type peak_idxs: ~list
    :param start_idxs: list of individual peak start indices
    :type start_idxs: ~list
    :param end_idxs: list of individual peak end indices
    :type end_idxs: ~list
    :param baseline: running baseline of the signal
    :type baseline: ~numpy.ndarray
    :param aub_window_s: number of samples before and after peak_idxs to look
    for the nadir
    :type aub_window_s: ~int
    :param ref_signal: signal in which the nadir is searched
    :type ref_signal: ~numpy.ndarray
    :param aub_threshold: maximum aub error percentage for a peak
    :type aub_threshold: ~float

    :returns valid_timeproducts: boolean list of valid time products
    :rtype valid_timeproducts: numpy.ndarray[bool]
    :returns percentages_aub: list of calculated aub percentages
    :rtype percentages_aub: numpy.ndarray[float]
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

    param aubs: List of area under the baseline values. See
    resurfemg.postprocessing.features.area_under_baseline
    ---------------------------------------------------------------------------
    :features.area_under_baseline
    :type aubs: ~numpy.ndarray[~float]
    :param threshold_percentile: percentile for detecting high baseline
    :type threshold_percentile: ~float
    :param threshold_factor: multiplication factor for threshold_percentile
    :type threshold_factor: ~float

    :returns valid_aubs: Boolean list of aub values under threshold
    :rtype valid_aubs: numpy.ndarray[bool]
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

    See postprocessing.features.time_product
    ---------------------------------------------------------------------------
    :param time_products: List of time_productsvalues.
    :type time_products: ~numpy.ndarray[~float]
    :param upper_percentile: percentile for detecting high time products
    :type upper_percentile: ~float
    :param upper_factor: multiplication factor for upper_percentile
    :type upper_factor: ~float
    :param lower_percentile: percentile for detecting low time products
    :type lower_percentile: ~float
    :param lower_factor: multiplication factor for lower_percentile
    :type lower_factor: ~float

    :returns valid_etps: Boolean list of time product values within bounds
    :rtype valid_etps: numpy.ndarray[bool]
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
    ---------------------------------------------------------------------------
    :param ventilator_breath_idxs: list of supported breath indices
    :type ventilator_breath_idxs: ~numpy.ndarray
    :param manoeuvres_idxs : list of manoeuvres indices
    :type manoeuvres_idxs: ~numpy.ndarray

    :returns valid_manoeuvres: Boolean list of valid manoeuvres
    :rtype valid_manoeuvres: numpy.ndarray[bool]
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
    ---------------------------------------------------------------------------
    :param signal: filtered signal
    :type signal: ~numpy.ndarray
    :param peak_idxs: list of peak indices
    :type peak_idxs: ~numpy.ndarray
    :param start_idxs: list of onsets indices
    :type start_idxs: ~numpy.ndarray
    :param end_idxs: list of offsets indices
    :type end_idxs: ~numpy.ndarray
    :param fs: sampling rate
    :type fs: int
    :param time_products: list of area under the curves per peak
    :type time_products: ~numpy.ndarray
    :param bell_window_s: number of samples before and after peak_idxs to look
    for the nadir
    :type bell_window_s: ~int
    :param bell_threshold: maximum bell error percentage for a valid peak
    :type bell_threshold: ~float

    :returns valid_peak: boolean list of valid peaks
    :rtype valid_peak: numpy.ndarray[bool]
    :returns percentage_bell_error: calculated bell errors in percentage
    :rtype: numpy.ndarray[float]
    :returns bell_error: calculated bell errors
    :rtype: numpy.ndarray[float]
    :returns y_min: minimum value of the baseline
    :rtype: numpy.ndarray[float]
    :returns fitted_parameters: fitted bell curve parameters
    :rtype: numpy.ndarray[float]
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
    ---------------------------------------------------------------------------
    :param t_events_1: Timing of the events that should happen first
    :type t_events_1: ~numpy.ndarray[float]
    :param t_events_2: Timing of the events that should happen second
    :type t_events_2: ~numpy.ndarray[float]
    :param delta_min: The delta time event 1 should at least preceed event 2.
    :type delta_min: ~float
    :param delta_max: The delta time event 1 should maximally preceed event 2.
    :type delta_max: ~float

    :returns correct_timing: Boolean list of correct timing
    :rtype correct_timing: numpy.ndarray[bool]
    :returns delta_time: List of delta times between the events
    :rtype delta_time: numpy.ndarray[float]
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
    ---------------------------------------------------------------------------
    :param emg_breath_idxs: EMG breath indices
    :type emg_breath_idxs: ~numpy.ndarray
    :param t_emg: Recording time in seconds
    :type t_emg: ~float
    :param vent_rr: ventilatory respiratory rate (breath/min)
    :type vent_rr: ~float
    :param min_fraction: Required minimum detected fraction of EMG breaths
    :type min_fraction: ~numpy.ndarray

    :returns detected_fraction: Fraction of detected EMG breaths
    :rtype detected_fraction: float
    :returns criterium_met: Boolean if the fraction is above the minimum
    :rtype criterium_met: bool
    """
    detected_fraction = float(len(emg_breath_idxs) / (rr_vent * t_emg / 60))
    criterium_met = detected_fraction >= min_fraction
    return (detected_fraction, criterium_met)
