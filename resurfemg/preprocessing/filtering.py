"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to filter EMG arrays.
"""

import numpy as np
from scipy import signal


def emg_bandpass_butter(
    emg_raw,
    high_pass,
    low_pass,
    fs_emg,
    order=3,
):
    """Bandpass filter for EMG signal.
    ---------------------------------------------------------------------------
    :param emg_raw: The raw EMG signal
    :type emg_raw: ~numpy.ndarray
    :param high_pass: High pass cut-off frequency `frequenceisabove`
    :type high_pass: ~float
    :param low_pass: Low pass cut-off frequency `frequenciesbelow`
    :type low_pass: ~float
    :param fs_emg: Sampling frequency
    :type fs_emg: int
    :param order: The filter order
    :type order: int

    :returns emg_filt: The bandpass filtered EMG data
    :rtype emg_filt: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
        [high_pass, low_pass],
        "bandpass",
        fs=fs_emg,
        output="sos",
    )
    # sos (output parameter)is second order section  -> "stabilizes" ?
    emg_filt = signal.sosfiltfilt(sos, emg_raw)
    return emg_filt


def emg_lowpass_butter(
    emg_raw,
    low_pass,
    fs_emg,
    order=3,
):
    """Lowpass filter for EMG signal.
    ---------------------------------------------------------------------------
    :param emg_raw: The raw EMG signal
    :type emg_raw: ~numpy.ndarray
    :param low_pass: Low pass cut-off frequency `frequenciesbelow`
    :type low_pass: ~float
    :param fs_emg: Sampling frequency
    :type fs_emg: int
    :param order: The filter order
    :type order: int

    :returns emg_filt: The lowpass filtered EMG data
    :rtype: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
        low_pass,
        "lowpass",
        fs=fs_emg,
        output="sos",
    )
    emg_filt = signal.sosfiltfilt(sos, emg_raw)
    return emg_filt


def emg_highpass_butter(
    emg_raw,
    high_pass,
    fs_emg,
    order=3,
):
    """Highpass filter for EMG signal.
    ---------------------------------------------------------------------------
    :param emg_raw: The raw EMG signal
    :type emg_raw: ~numpy.ndarray
    :param high_pass: High pass cut-off frequency `frequenceisabove`
    :type high_pass: ~float
    :param fs_emg: Sampling frequency
    :type fs_emg: int
    :param order: The filter order
    :type order: int

    :returns emg_filt: The highpass filtered EMG data
    :rtype emg_filt: ~numpy.ndarray
    """
    sos = signal.butter(
        order,
        high_pass,
        "highpass",
        fs=fs_emg,
        output="sos",
    )
    emg_filt = signal.sosfiltfilt(sos, emg_raw)
    return emg_filt


def notch_filter(emg_raw, f_notch, fs_emg, q):
    """Filter to take out a specific frequency band.
    ---------------------------------------------------------------------------
    :param emg_raw: Percentage variation tolerance to allow without cutting
    :type emg_raw: int
    :param f_notch: The frequency to remove from the signal
    :type f_notch: float
    :param fs_emg: Sampling frequency
    :type fs_emg: int
    :param q: quality factor of notch filter, Q = f_notch/band_width of band-
        stop, see scipy.signal.iirnotch
    :type q: float

    :returns emg_filt: The notch filtered EMG data
    :rtype emg_filt: ~numpy.ndarray
    """
    b_notch, a_notch = signal.iirnotch(f_notch, q, fs_emg)

    output_signal = signal.filtfilt(b_notch, a_notch, emg_raw)
    return output_signal


def compute_power_loss(
    signal_original,
    fs_original,
    signal_processed,
    fs_processed,
    n_segment=None,
    percent_overlap=25,
):
    """Compute the percentage of power loss after the processing.
    ---------------------------------------------------------------------------
    :param signal_original: Original signal
    :type  signal_original: ~numpy.ndarray
    :param fs_original: Sampling frequency of orginal signal
    :type original_signal_sampling_frequency: int
    :param signal_processed: Array.
    :type  signal_processed: ~numpy.ndarray
    :param fs_processed: Sampling frequency of orginal signal
    :type fs_processed: int
    :param n_segment: Pwelch window width
    :type n_segment: int
    :param percent_overlap: Pwelch window overlap percentage
    :type percent_overlap: float

    :returns power_loss: Percentage of power loss
    :rtype power_loss: float
    """
    if n_segment is None:
        n_segment = fs_original // 2

    noverlap = int(percent_overlap / 100 * fs_original)

    # power spectrum density of the original and
    # processed signals using Welch method
    pxx_den_orig = signal.welch(  # as per Lu et al. 2009
        signal_original,
        fs_original,
        nperseg=n_segment,
        noverlap=noverlap,
    )
    pxx_den_processed = signal.welch(
        signal_processed,
        fs_processed,
        nperseg=n_segment,
        noverlap=noverlap,
    )
    # compute the percentage of power loss
    power_loss = 100 * (1 - (np.sum(pxx_den_orig) / np.sum(pxx_den_processed)))

    return power_loss
