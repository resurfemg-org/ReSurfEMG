"""This file contains functions to filter EMG arrays.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def emg_bandpass_butter(
    emg_raw: np.ndarray | tuple[np.ndarray, np.ndarray],
    high_pass: float,
    low_pass: float,
    fs_emg: int,
    order: int = 3,
) -> np.ndarray:
    """Bandpass filter for EMG signal.

    Args:
        emg_raw (numpy.ndarray): The raw EMG signal.
        high_pass (float): High pass cut-off frequency.
        low_pass (float): Low pass cut-off frequency.
        fs_emg (int): Sampling frequency.
        order (int): The filter order.

    Returns:
        numpy.ndarray: The bandpass filtered EMG data.
    """
    sos = signal.butter(
        order,
        [high_pass, low_pass],
        "bandpass",
        fs=fs_emg,
        output="sos",
    )
    # sos (output parameter)is second order section  -> "stabilizes" ?
    return signal.sosfiltfilt(sos, emg_raw)


def emg_lowpass_butter(
    emg_raw: np.ndarray | tuple[np.ndarray, np.ndarray],
    low_pass: float,
    fs_emg: int,
    order: int = 3,
) -> np.ndarray:
    """Lowpass filter for EMG signal.

    Args:
        emg_raw (numpy.ndarray): The raw EMG signal.
        low_pass (float): Low pass cut-off frequency.
        fs_emg (int): Sampling frequency.
        order (int): The filter order.

    Returns:
        numpy.ndarray: The lowpass filtered EMG data.
    """
    sos = signal.butter(
        order,
        low_pass,
        "lowpass",
        fs=fs_emg,
        output="sos",
    )
    return signal.sosfiltfilt(sos, emg_raw)


def emg_highpass_butter(
    emg_raw: np.ndarray | tuple[np.ndarray, np.ndarray],
    high_pass: float,
    fs_emg: int,
    order: int = 3,
) -> np.ndarray:
    """Highpass filter for EMG signal.

    Args:
        emg_raw (numpy.ndarray): The raw EMG signal.
        high_pass (float): High pass cut-off frequency.
        fs_emg (int): Sampling frequency.
        order (int): The filter order.

    Returns:
        numpy.ndarray: The highpass filtered EMG data.
    """
    sos = signal.butter(
        order,
        high_pass,
        "highpass",
        fs=fs_emg,
        output="sos",
    )
    return signal.sosfiltfilt(sos, emg_raw)


def notch_filter(
    emg_raw: np.ndarray | tuple[np.ndarray, np.ndarray],
    f_notch: float,
    fs_emg: int,
    q: float,
) -> np.ndarray:
    """Filter to take out a specific frequency band.

    Args:
        emg_raw (numpy.ndarray): The raw EMG signal.
        f_notch (float): The frequency to remove from the signal.
        fs_emg (int): Sampling frequency.
        q (float): Quality factor of notch filter, Q = f_notch/band_width of
            band-stop, see scipy.signal.iirnotch.

    Returns:
        numpy.ndarray: The notch filtered EMG data.
    """
    b_notch, a_notch = signal.iirnotch(f_notch, q, fs_emg)

    return signal.filtfilt(b_notch, a_notch, emg_raw)


def compute_power_loss(
    signal_original: np.ndarray | tuple[np.ndarray, np.ndarray],
    fs_original: int,
    signal_processed: np.ndarray | tuple[np.ndarray, np.ndarray],
    fs_processed: int,
    n_segment: int | None = None,
    percent_overlap: float = 25,
) -> float:
    """Compute the percentage of power loss after the processing.

    Args:
        signal_original (numpy.ndarray): Original signal.
        fs_original (int): Sampling frequency of original signal.
        signal_processed (numpy.ndarray): Processed signal.
        fs_processed (int): Sampling frequency of processed signal.
        n_segment (int, optional): Pwelch window width.
        percent_overlap (float): Pwelch window overlap percentage.

    Returns:
        float: Percentage of power loss.
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
    return 100 * (1 - (np.sum(pxx_den_orig) / np.sum(pxx_den_processed)))
