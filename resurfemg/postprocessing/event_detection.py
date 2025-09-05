"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to extract detect peak, on- and offset samples.
"""
import numpy as np
import scipy
import scipy.signal

from resurfemg.helper_functions.math_operations import derivative


def find_occluded_breaths(
    p_vent,
    fs,
    peep,
    start_idx=0,
    end_idx=None,
    prominence_factor=0.8,
    min_width_s=None,
    distance_s=None,
):
    """
    Find end-expiratory occlusion manoeuvres (Pocc) in ventilator pressure
    timeseries data. start_idx and end_idx specify the samples to look into.
    The prominence_factor, min_width_s, and distance_s specify the minimal
    peak prominence relative to the PEEP level, peak width in samples, and
    distance to other peaks.
    ---------------------------------------------------------------------------
    :param p_vent: ventilator pressure signal
    :type p_vent: ~nd.array
    :param fs: sampling rate
    :type fs: int
    :param peep: positive end-expiratory pressure
    :type peep: ~float
    :param start_idx: start index to start looking for Pocc manoeuvres
    :type start_idx: int
    :param end_idx: end index to start looking for Pocc manoeuvres
    :type end_idx: int
    :param prominence_factor: multiplier in setting the minimum peak prominence
    :type prominence_factor: ~float
    :param min_width_s: minimum peak width in samples
    :type min_width_s: int
    :param distance_s: mininum interpeak distance in samples
    :type distance_s: int

    :returns peak_idxs: list of Pocc peak indices
    :rtpe peak_idxs: ~nd.array[int]
    """
    if end_idx is None:
        end_idx = len(p_vent) - 1

    if min_width_s is None:
        if fs is None:
            raise ValueError('Minimmal peak min_width_s and ventilator'
                             + ' sampling rate are not defined.')
        min_width_s = int(0.1 * fs)

    if distance_s is None:
        if fs is None:
            raise ValueError('Minimmal peak distance and ventilator sampling'
                             + ' rate are not defined.')
        distance_s = int(0.5 * fs)

    prominence = prominence_factor * np.abs(peep - min(p_vent))
    height = prominence - peep

    peak_idxs, _ = scipy.signal.find_peaks(
        -p_vent[start_idx:end_idx],
        height=height,
        prominence=prominence,
        width=min_width_s,
        distance=distance_s
    )
    return peak_idxs


def onoffpeak_baseline_crossing(
    signal_env,
    baseline,
    peak_idxs
):
    """This function calculates the peaks of each breath using the
    slopesum baseline of envelope data.
    ---------------------------------------------------------------------------
    :param signal_env: envelope signal
    :type signal_env: ~numpy.ndarray
    :param baseline: baseline signal of EMG data for baseline detection
    :type baseline: ~numpy.ndarray
    :param peak_idxs: list of peak indices for which to find on- and offset
    :type peak_idxs: ~numpy.ndarray

    :returns peak_start_idxs: list of start indices of the peaks
    :rtype peak_start_idxs: ~numpy.ndarray[int]
    :returns peak_end_idxs: list of end indices of the peaks
    :rtype peak_end_idxs: ~numpy.ndarray[int]
    :returns valid_starts_bools: list of boolean values for valid starts
    :rtype valid_starts_bools: ~numpy.ndarray[bool]
    :returns valid_ends_bools: list of boolean values for valid ends
    :rtype valid_ends_bools: ~numpy.ndarray[bool]
    :returns valid_peaks: list of boolean values for valid peaks
    :rtype valid_peaks: ~numpy.ndarray[bool]
    """

    # Detect the sEAdi on- and offsets
    baseline_crossings_idx = np.nonzero(
        np.diff(np.sign(signal_env - baseline)) != 0)[0]

    peak_start_idxs = np.zeros((len(peak_idxs),), dtype=int)
    peak_end_idxs = np.zeros((len(peak_idxs),), dtype=int)
    valid_starts_bools = np.array([True for _ in range(len(peak_idxs))])
    valid_ends_bools = np.array([True for _ in range(len(peak_idxs))])
    for peak_nr, peak_idx in enumerate(peak_idxs):
        delta_samples = peak_idx - baseline_crossings_idx[
            baseline_crossings_idx < peak_idx]
        if len(delta_samples) < 1:
            peak_start_idxs[peak_nr] = 0
            peak_end_idxs[peak_nr] = baseline_crossings_idx[
                baseline_crossings_idx > peak_idx][0]
        else:
            a = np.argmin(delta_samples)

            peak_start_idxs[peak_nr] = int(baseline_crossings_idx[a])
            if a < len(baseline_crossings_idx) - 1:
                peak_end_idxs[peak_nr] = int(baseline_crossings_idx[a+1])
            else:
                peak_end_idxs[peak_nr] = len(signal_env) - 1

        # Evaluate start validity
        if (peak_nr > 0) and (peak_start_idxs[peak_nr] > peak_idxs[peak_nr]):
            valid_starts_bools[peak_nr] = False

        # Evaluate end validity
        if ((peak_nr < (len(peak_idxs)-2))
                and (valid_ends_bools[peak_nr] > peak_idxs[peak_nr+1])):
            valid_ends_bools[peak_nr] = False

        # Evaluate conflicts
        if ((peak_nr > 0)
                and (peak_start_idxs[peak_nr] <= peak_end_idxs[peak_nr-1])):
            if valid_starts_bools[peak_nr] is False:
                # The current start is already labelled as incorrect
                pass
            elif valid_ends_bools[peak_nr-1] is False:
                # The previous end is already labelled as incorrect
                pass
            elif ((peak_idx - peak_start_idxs[peak_nr])
                    > (peak_end_idxs[peak_nr-1] - peak_idxs[peak_nr-1])):
                # New start is further apart from peak idx than previous end
                # New start is probably invalid
                valid_starts_bools[peak_nr] = False
            else:
                # Previous end is further apart from peak idx than new start
                # Previous end is probably invalid
                valid_ends_bools[peak_nr-1] = False

    valid_peaks = [valid_detections[0] and valid_detections[1]
                   for valid_detections
                   in zip(valid_starts_bools, valid_ends_bools)]

    return (peak_start_idxs, peak_end_idxs,
            valid_starts_bools, valid_ends_bools, valid_peaks)


def onoffpeak_slope_extrapolation(
    signal_env,
    fs,
    peak_idxs,
    slope_window_s
):
    """This function calculates the peak on- and offsets of a signal by extra-
    polating the maximum slopes in de slope_window_s to the zero crossings.
    The validity arrays provide feedback on the validity of the detected on-
    and offsets, aiming to prevent onsets after peak indices, offsets before
    peak indices, and overlapping peaks.
    ---------------------------------------------------------------------------
    :param signal_env: signal to identify on- and offsets in
    :type signal_env: ~numpy.ndarray
    :param fs: sampling rate
    :type fs: int
    :param peak_idxs: list of peak indices for which to find on- and offset
    :type peak_idxs: ~numpy.ndarray
    :slope_window_s: how many samples on each side to use for the comparison
    to consider for detecting the local maximum slope
    :type fs: int

    :returns peak_start_idxs: list of start indices of the peaks
    :rtype peak_start_idxs: ~numpy.ndarray[int]
    :returns peak_end_idxs: list of end indices of the peaks
    :rtype peak_end_idxs: ~numpy.ndarray[int]
    :returns valid_starts_bools: list of boolean values for valid starts
    :rtype valid_starts_bools: ~numpy.ndarray[bool]
    :returns valid_ends_bools: list of boolean values for valid ends
    :rtype valid_ends_bools: ~numpy.ndarray[bool]
    :returns valid_peaks: list of boolean values for valid peaks
    :rtype valid_peaks: ~numpy.ndarray[bool]
    """

    dsignal_dt = derivative(signal_env, fs)

    max_upslope_idxs = scipy.signal.argrelextrema(
        dsignal_dt, np.greater, order=slope_window_s)[0]
    max_downslope_idxs = scipy.signal.argrelextrema(
        dsignal_dt, np.less, order=slope_window_s)[0]

    peak_start_idxs = np.zeros((len(peak_idxs),), dtype=int)
    peak_end_idxs = np.zeros((len(peak_idxs),), dtype=int)
    valid_starts_bools = np.array([True for _ in range(len(peak_idxs))])
    valid_ends_bools = np.array([True for _ in range(len(peak_idxs))])
    prev_downslope = 0
    for peak_nr, peak_idx in enumerate(peak_idxs):
        if len(max_upslope_idxs[max_upslope_idxs < peak_idx]) < 1:
            start_idx = 0
        else:
            max_upslope_idx = int(
                max_upslope_idxs[max_upslope_idxs < peak_idx][-1])

            new_upslope = dsignal_dt[max_upslope_idx]
            y_val = signal_env[max_upslope_idx]
            dy_dt_val = dsignal_dt[max_upslope_idx]
            upslope_idx_ds = np.array(
                y_val * fs // (dy_dt_val), dtype=int).astype(np.int64)

            start_idx = max([0, max_upslope_idx - upslope_idx_ds])

        peak_start_idxs[peak_nr] = start_idx

        if len(max_downslope_idxs[max_downslope_idxs > peak_idx]) < 1:
            end_idx = len(signal_env) - 1
        else:
            if peak_nr > 0:
                prev_downslope = dsignal_dt[max_downslope_idx]

            max_downslope_idx = int(
                max_downslope_idxs[max_downslope_idxs > peak_idx][0])\

            y_val = signal_env[max_downslope_idx]
            dy_dt_val = dsignal_dt[max_downslope_idx]
            downslope_idx_ds = np.array(
                y_val * fs // (dy_dt_val), dtype=int).astype(np.int64)

            end_idx = min([len(signal_env) - 1,
                           max_downslope_idx - downslope_idx_ds])

        peak_end_idxs[peak_nr] = end_idx

        # Evaluate start validity
        if start_idx > peak_idx:
            valid_starts_bools[peak_nr] = False

        # Evaluate end validity
        if end_idx < peak_idx:
            valid_ends_bools[peak_nr] = False

        if (peak_nr < (len(peak_idxs)-2)) and (end_idx > peak_idxs[peak_nr+1]):
            valid_ends_bools[peak_nr] = False

        # Evaluate conflicts
        if (peak_nr > 0) and (start_idx < peak_end_idxs[peak_nr-1]):
            if valid_ends_bools[peak_nr-1] is False:
                # The previous end is already labelled as incorrect
                pass
            elif new_upslope > -prev_downslope:
                # New upslope is steeper than previous downslope
                # Previous downslope is probably invalid
                valid_ends_bools[peak_nr-1] = False
            else:
                # Previous downslope is steeper than new upslope
                # New upslope is probably invalid
                valid_starts_bools[peak_nr] = False

    valid_peaks = [valid_detections[0] and valid_detections[1]
                   for valid_detections
                   in zip(valid_starts_bools, valid_ends_bools)]

    return (peak_start_idxs, peak_end_idxs,
            valid_starts_bools, valid_ends_bools, valid_peaks)


def detect_ventilator_breath(
    signal,
    start_idx,
    end_idx,
    width_s,
    threshold=None,
    prominence=None,
    threshold_new=None,
    prominence_new=None
):
    """Identify the breaths from the ventilator signal and return an array
    of ventilator peak breath indices, in two steps of peak detection.
    Input of threshold and prominence values is optional.
    ---------------------------------------------------------------------------
    :param signal: Ventilator signal
    :type signal: ~numpy.ndarray
    :param start_idx: start sample of the window in which to be searched
    :type start_idx: ~int
    :param end_idx: end sample of the window in which to be searched
    :type end_idx: ~int
    :param width_s: required width of peak in samples
    :type width_s:
    :param threshold: required threshold of peaks, vertical threshold to
    neighbouring samples
    :type threshold: ~int
    :param prominence: required prominence of peaks
    :type prominence: ~int
    :param threshold_new:
    :type threshold_new: ~int
    :param prominence_new:
    :type prominence_new: ~int

    :returns ventilator_breath_idxs: list of ventilator breath peak indices
    :rtype ventilator_breath_idxs: list[int]
    """

    v_t_slice = signal[int(start_idx):int(end_idx)]
    if threshold is None:
        threshold = 0.25 * np.percentile(v_t_slice, 90)
    if prominence is None:
        prominence = 0.10 * np.percentile(v_t_slice, 90)

    resp_eff, _ = scipy.signal.find_peaks(
        v_t_slice, height=threshold, prominence=prominence, width=width_s)

    if threshold_new is None:
        threshold_new = 0.5 * np.percentile(v_t_slice[resp_eff], 90)
    if prominence_new is None:
        prominence_new = 0.5 * np.percentile(v_t_slice, 90)

    ventilator_breath_idxs, _ = scipy.signal.find_peaks(
        v_t_slice,
        height=threshold_new,
        prominence=prominence_new,
        width=width_s
    )

    return ventilator_breath_idxs


def detect_emg_breaths(
    emg_env,
    emg_baseline=None,
    threshold=0,
    prominence_factor=0.5,
    min_peak_width_s=1,
):
    """
    Identify the electrophysiological breaths from the EMG envelope and return
    an array breath peak indices. Input of baseline threshold, peak prominence
    factor, and minimal peak width are optional.
    ---------------------------------------------------------------------------
    :param emg_env: 1D EMG envelope signal
    :type emg_env: ~numpy.ndarray
    :param emg_baseline: EMG baseline. If none provided, 0 baseline is used.
    :type emg_baseline: ~numpy.ndarray
    :param threshold: required threshold of peaks, vertical threshold to
    neighbouring samples
    :type threshold: ~float
    :param prominence_factor: required prominence of peaks, relative to the
    75th - 50th percentile of the emg_env above the baseline
    :type prominence_factor: ~float
    :param min_peak_width_s: required width of peak in samples
    :type min_peak_width_s: ~int

    :returns peak_idxs: list of EMG breath peak indices
    :rtype peak_idxs: list[int]
    """
    if emg_baseline is None:
        emg_baseline = np.zeros(emg_env.shape)

    emg_env_delta = emg_env - emg_baseline
    prominence = prominence_factor * \
        (np.nanpercentile(emg_env_delta, 75)
         + np.nanpercentile(emg_env_delta, 50))
    peak_idxs, _ = scipy.signal.find_peaks(
        emg_env,
        height=threshold,
        prominence=prominence,
        width=min_peak_width_s)

    return peak_idxs


def find_linked_peaks(
    signal_1_t_peaks,
    signal_2_t_peaks,
):
    """
    Find the indices of the peaks in signal 2 closest to the time of the
    peaks in signal 1
    :param signal_1_t_peaks: list of timing of peaks in signal 1
    :type signal_1_t_peaks: ~numpy.ndarray
    :param signal_2_t_peaks: list of timing of peaks in signal 2
    :type signal_2_t_peaks: ~numpy.ndarray

    :returns peaks_idxs_signal_1_in_2: Peak indices of signal 2 closest to the
        peaks in signal 1
    :rtype peaks_idxs_signal_1_in_2: ~numpy.ndarray[int]
    """
    if not isinstance(signal_1_t_peaks, np.ndarray):
        signal_1_t_peaks = np.array(signal_1_t_peaks)
    if not isinstance(signal_2_t_peaks, np.ndarray):
        signal_2_t_peaks = np.array(signal_2_t_peaks)
    peaks_idxs_signal_1_in_2 = np.zeros(signal_1_t_peaks.shape, dtype=int)
    for idx, signal_1_t_peak in enumerate(signal_1_t_peaks):
        peaks_idxs_signal_1_in_2[idx] = np.argmin(
            np.abs(signal_2_t_peaks - signal_1_t_peak)
        )

    return peaks_idxs_signal_1_in_2
