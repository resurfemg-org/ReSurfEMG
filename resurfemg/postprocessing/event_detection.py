"""Peak extraction and on- and offset detection functions.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to extract detect peak, on- and offset samples.
"""

from __future__ import annotations

import numpy as np
import scipy
import scipy.signal

from resurfemg.helper_functions.math_operations import derivative


def find_occluded_breaths(
    p_vent: np.ndarray,
    fs: int,
    peep: float,
    start_idx: int = 0,
    end_idx: int | None = None,
    prominence_factor: float = 0.8,
    min_width_s: int | None = None,
    distance_s: int | None = None,
) -> np.ndarray:
    """Find ventilatory PEEP.

    Find end-expiratory occlusion manoeuvres (Pocc) in ventilator pressure
    timeseries data. start_idx and end_idx specify the samples to look into.
    The prominence_factor, min_width_s, and distance_s specify the minimal
    peak prominence relative to the PEEP level, peak width in samples, and
    distance to other peaks.

    Args:
        p_vent (numpy.ndarray): Ventilator pressure signal.
        fs (int): Sampling rate.
        peep (float): Positive end-expiratory pressure.
        start_idx (int): Start index to start looking for Pocc manoeuvres.
        end_idx (int, optional): End index to stop looking for Pocc manoeuvres.
        prominence_factor (float): Multiplier in setting the minimum peak prominence.
        min_width_s (int, optional): Minimum peak width in samples.
        distance_s (int, optional): Minimum interpeak distance in samples.

    Returns:
        numpy.ndarray: List of Pocc peak indices.
    """
    if end_idx is None:
        end_idx = len(p_vent) - 1

    if min_width_s is None:
        if fs is None:
            msg = "Minimmal peak min_width_s and ventilator sampling rate are not defined."
            raise ValueError(msg)
        min_width_s = int(0.1 * fs)

    if distance_s is None:
        if fs is None:
            msg = "Minimmal peak distance and ventilator sampling rate are not defined."
            raise ValueError(msg)
        distance_s = int(0.5 * fs)

    prominence = prominence_factor * np.abs(peep - min(p_vent))
    height = prominence - peep

    peak_idxs, _ = scipy.signal.find_peaks(
        -p_vent[start_idx:end_idx],
        height=height,
        prominence=prominence,
        width=min_width_s,
        distance=distance_s,
    )
    return peak_idxs


def onoffpeak_baseline_crossing(
    signal_env: np.ndarray, baseline: np.ndarray, peak_idxs: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate breath peaks.

    This function calculates the peaks of each breath using the
    slopesum baseline of envelope data.

    Args:
        signal_env (numpy.ndarray): Envelope signal.
        baseline (numpy.ndarray): Baseline signal of EMG data for baseline detection.
        peak_idxs (numpy.ndarray): List of peak indices for which to find on- and offset.

    Returns:
        tuple:
            - numpy.ndarray: List of start indices of the peaks.
            - numpy.ndarray: List of end indices of the peaks.
            - numpy.ndarray: List of boolean values for valid starts.
            - numpy.ndarray: List of boolean values for valid ends.
            - numpy.ndarray: List of boolean values for valid peaks.
    """
    # Detect the sEAdi on- and offsets
    baseline_crossings_idx = np.nonzero(np.diff(np.sign(signal_env - baseline)) != 0)[0]

    peak_start_idxs = np.zeros((len(peak_idxs),), dtype=int)
    peak_end_idxs = np.zeros((len(peak_idxs),), dtype=int)
    valid_starts_bools = np.array([True for _ in range(len(peak_idxs))])
    valid_ends_bools = np.array([True for _ in range(len(peak_idxs))])
    for peak_nr, peak_idx in enumerate(peak_idxs):
        delta_samples = peak_idx - baseline_crossings_idx[baseline_crossings_idx < peak_idx]
        if len(delta_samples) < 1:
            peak_start_idxs[peak_nr] = 0
            peak_end_idxs[peak_nr] = baseline_crossings_idx[baseline_crossings_idx > peak_idx][0]
        else:
            a = np.argmin(delta_samples)

            peak_start_idxs[peak_nr] = int(baseline_crossings_idx[a])
            if a < len(baseline_crossings_idx) - 1:
                peak_end_idxs[peak_nr] = int(baseline_crossings_idx[a + 1])
            else:
                peak_end_idxs[peak_nr] = len(signal_env) - 1

        # Evaluate start validity
        if (peak_nr > 0) and (peak_start_idxs[peak_nr] > peak_idx):
            valid_starts_bools[peak_nr] = False

        # Evaluate end validity
        if (peak_nr < (len(peak_idxs) - 2)) and (valid_ends_bools[peak_nr] > peak_idxs[peak_nr + 1]):
            valid_ends_bools[peak_nr] = False

        if (
            peak_nr > 0
            and peak_start_idxs[peak_nr] <= peak_end_idxs[peak_nr - 1]
            and valid_starts_bools[peak_nr]
            and valid_ends_bools[peak_nr - 1]
        ):
            invalid_current_start = (
                peak_idx - peak_start_idxs[peak_nr] > peak_end_idxs[peak_nr - 1] - peak_idxs[peak_nr - 1]
            )

            valid_starts_bools[peak_nr] = not invalid_current_start
            valid_ends_bools[peak_nr - 1] = invalid_current_start

    valid_peaks = np.array(
        [valid_detections[0] and valid_detections[1] for valid_detections in zip(valid_starts_bools, valid_ends_bools)],
        dtype=bool,
    )

    return (
        peak_start_idxs,
        peak_end_idxs,
        valid_starts_bools,
        valid_ends_bools,
        valid_peaks,
    )


def onoffpeak_slope_extrapolation(
    signal_env: np.ndarray, fs: int, peak_idxs: np.ndarray, slope_window_s: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the on- and offsets of peaks using slope extrapolation.

    This function calculates the peak on- and offsets of a signal by extra-
    polating the maximum slopes in de slope_window_s to the zero crossings.
    The validity arrays provide feedback on the validity of the detected on-
    and offsets, aiming to prevent onsets after peak indices, offsets before
    peak indices, and overlapping peaks.

    Args:
        signal_env (numpy.ndarray): Signal to identify on- and offsets in.
        fs (int): Sampling rate.
        peak_idxs (numpy.ndarray): List of peak indices for which to find on- and offset.
        slope_window_s (int): How many samples on each side to use for detecting the
            local maximum slope.

    Returns:
        tuple:
            - numpy.ndarray: List of start indices of the peaks.
            - numpy.ndarray: List of end indices of the peaks.
            - numpy.ndarray: List of boolean values for valid starts.
            - numpy.ndarray: List of boolean values for valid ends.
            - numpy.ndarray: List of boolean values for valid peaks.
    """
    dsignal_dt = derivative(signal_env, fs)

    max_upslope_idxs = scipy.signal.argrelextrema(dsignal_dt, np.greater, order=slope_window_s)[0]
    max_downslope_idxs = scipy.signal.argrelextrema(dsignal_dt, np.less, order=slope_window_s)[0]

    peak_start_idxs = np.zeros((len(peak_idxs),), dtype=int)
    peak_end_idxs = np.zeros((len(peak_idxs),), dtype=int)
    valid_starts_bools = np.array([True for _ in range(len(peak_idxs))])
    valid_ends_bools = np.array([True for _ in range(len(peak_idxs))])
    prev_downslope = 0
    new_upslope = 0
    for peak_nr, peak_idx in enumerate(peak_idxs):
        if len(max_upslope_idxs[max_upslope_idxs < peak_idx]) < 1:
            start_idx = 0
        else:
            max_upslope_idx = int(max_upslope_idxs[max_upslope_idxs < peak_idx][-1])

            new_upslope = dsignal_dt[max_upslope_idx]
            y_val = signal_env[max_upslope_idx]
            dy_dt_val = dsignal_dt[max_upslope_idx]
            upslope_idx_ds = np.array(y_val * fs // (dy_dt_val), dtype=int).astype(np.int64)

            start_idx = max([0, max_upslope_idx - upslope_idx_ds])

        peak_start_idxs[peak_nr] = start_idx

        if len(max_downslope_idxs[max_downslope_idxs > peak_idx]) < 1:
            end_idx = len(signal_env) - 1
        else:
            max_downslope_idx = int(max_downslope_idxs[max_downslope_idxs > peak_idx][0])

            if peak_nr > 0:
                prev_downslope = dsignal_dt[max_downslope_idx]

            y_val = signal_env[max_downslope_idx]
            dy_dt_val = dsignal_dt[max_downslope_idx]
            downslope_idx_ds = np.array(y_val * fs // (dy_dt_val), dtype=int).astype(np.int64)

            end_idx = min([len(signal_env) - 1, max_downslope_idx - downslope_idx_ds])

        peak_end_idxs[peak_nr] = end_idx

        # Evaluate start validity
        if start_idx > peak_idx:
            valid_starts_bools[peak_nr] = False

        # Evaluate end validity
        if end_idx < peak_idx:
            valid_ends_bools[peak_nr] = False

        if (peak_nr < (len(peak_idxs) - 2)) and (end_idx > peak_idxs[peak_nr + 1]):
            valid_ends_bools[peak_nr] = False

        # Evaluate conflicts
        if peak_nr > 0 and start_idx < peak_end_idxs[peak_nr - 1] and valid_ends_bools[peak_nr - 1]:
            invalidate_previous_end = new_upslope > -prev_downslope
            valid_ends_bools[peak_nr - 1] = not invalidate_previous_end
            valid_starts_bools[peak_nr] = invalidate_previous_end

    valid_peaks = np.array(
        [valid_detections[0] and valid_detections[1] for valid_detections in zip(valid_starts_bools, valid_ends_bools)],
        dtype=bool,
    )

    return (
        peak_start_idxs,
        peak_end_idxs,
        valid_starts_bools,
        valid_ends_bools,
        valid_peaks,
    )


def detect_ventilator_breath(
    v_vent: np.ndarray,
    start_idx: int,
    end_idx: int,
    width_s: int,
    threshold: float | None = None,
    prominence: float | None = None,
    threshold_new: float | None = None,
    prominence_new: float | None = None,
) -> list[int]:
    """Identify ventilator breaths in ventilator volume signal.

    Identify the breaths from the ventilator signal and return an array
    of ventilator peak breath indices, in two steps of peak detection.
    Input of threshold and prominence values is optional.

    Args:
        v_vent (numpy.ndarray): Ventilator volume signal.
        start_idx (int): Start sample of the window in which to be searched.
        end_idx (int): End sample of the window in which to be searched.
        width_s (int): Required width of peak in samples.
        threshold (float, optional): Required threshold of peaks, vertical threshold to
            neighbouring samples.
        prominence (float, optional): Required prominence of peaks.
        threshold_new (float, optional): Updated threshold for second peak detection pass.
        prominence_new (float, optional): Updated prominence for second peak detection pass.

    Returns:
        list[int]: List of ventilator breath peak indices.
    """
    v_t_slice = v_vent[int(start_idx) : int(end_idx)]
    if threshold is None:
        threshold = float(0.25 * np.percentile(v_t_slice, 90))
    if prominence is None:
        prominence = float(0.10 * np.percentile(v_t_slice, 90))

    resp_eff, _ = scipy.signal.find_peaks(v_t_slice, height=threshold, prominence=prominence, width=width_s)

    if threshold_new is None:
        threshold_new = float(0.5 * np.percentile(v_t_slice[resp_eff], 90))
    if prominence_new is None:
        prominence_new = float(0.5 * np.percentile(v_t_slice, 90))

    ventilator_breath_idxs, _ = scipy.signal.find_peaks(
        v_t_slice, height=threshold_new, prominence=prominence_new, width=width_s
    )

    return ventilator_breath_idxs


def detect_emg_breaths(
    emg_env: np.ndarray,
    emg_baseline: np.ndarray | None = None,
    threshold: float = 0,
    prominence_factor: float = 0.5,
    min_peak_width_s: int = 1,
) -> list[int]:
    """Identify breaths in EMG envelope.

    Identify the electrophysiological breaths from the EMG envelope and return
    an array breath peak indices. Input of baseline threshold, peak prominence
    factor, and minimal peak width are optional.

    Args:
        emg_env (numpy.ndarray): 1D EMG envelope signal.
        emg_baseline (numpy.ndarray, optional): EMG baseline. If none provided,
            0 baseline is used.
        threshold (float): Required threshold of peaks, vertical threshold to
            neighbouring samples.
        prominence_factor (float): Required prominence of peaks, relative to the
            75th - 50th percentile of the emg_env above the baseline.
        min_peak_width_s (int): Required width of peak in samples.

    Returns:
        list[int]: List of EMG breath peak indices.
    """
    if emg_baseline is None:
        emg_baseline = np.zeros(emg_env.shape)

    emg_env_delta = emg_env - emg_baseline
    prominence = prominence_factor * (np.nanpercentile(emg_env_delta, 75) + np.nanpercentile(emg_env_delta, 50))
    peak_idxs, _ = scipy.signal.find_peaks(emg_env, height=threshold, prominence=prominence, width=min_peak_width_s)

    return peak_idxs


def find_linked_peaks(
    signal_1_t_peaks: np.ndarray | list[float],
    signal_2_t_peaks: np.ndarray | list[float],
) -> np.ndarray:
    """Find linked peaks between two signals.

    Find the indices of the peaks in signal 2 closest to the time of the
    peaks in signal 1.

    Args:
        signal_1_t_peaks (numpy.ndarray): List of timing of peaks in signal 1.
        signal_2_t_peaks (numpy.ndarray): List of timing of peaks in signal 2.

    Returns:
        numpy.ndarray: Peak indices of signal 2 closest to the peaks in signal 1.
    """
    if not isinstance(signal_1_t_peaks, np.ndarray):
        signal_1_t_peaks = np.array(signal_1_t_peaks)
    if not isinstance(signal_2_t_peaks, np.ndarray):
        signal_2_t_peaks = np.array(signal_2_t_peaks)
    peaks_idxs_signal_1_in_2 = np.zeros(signal_1_t_peaks.shape, dtype=int)
    for idx, signal_1_t_peak in enumerate(signal_1_t_peaks):
        peaks_idxs_signal_1_in_2[idx] = np.argmin(np.abs(signal_2_t_peaks - signal_1_t_peak))

    return peaks_idxs_signal_1_in_2
