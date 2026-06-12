"""This file contains functions to perform default signal processing procedures.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

import resurfemg.helper_functions.visualization as vis
import resurfemg.preprocessing.ecg_removal as ecg_rm
import resurfemg.preprocessing.envelope as evl
import resurfemg.preprocessing.filtering as filt

if TYPE_CHECKING:
    import numpy as np


def quick_look(
    emg_raw: np.ndarray,
    fs_emg: int,
    plot_raw: bool = False,
    plot_clean: bool = True,
    plot_env: bool = True,
    plot_power_spectrum: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Method for quick inspection of EMG data based on high-pass filtering @80Hz.

    :param emg_raw: raw single channel EMG data
    :type emg_raw: ~numpy.ndarray
    :param fs_emg: sampling frequency
    :type fs_emg: ~int
    :param plot_raw: Plot the raw signal
    :type plot_raw: bool
    :param plot_clean: Plot the filtered signal
    :type plot_clean: bool
    :param plot_env: Plot the envelope of the signal
    :type plot_env: bool
    :param plot_power_spectrum: Plot the powerspectrum of the raw signal
    :type plot_power_spectrum: bool

    :returns emg_filt: The bandpass filtered EMG data
    :rtype emg_filt: ~numpy.ndarray
    :returns emg_env: The envelope of the EMG data
    :rtype emg_env: ~numpy.ndarray
    """
    emg_filt = filt.emg_bandpass_butter(
        emg_raw=emg_raw,
        high_pass=80,
        low_pass=min([fs_emg // 2, 500]),
        fs_emg=fs_emg,
    )
    emg_env = evl.full_rolling_arv(emg_filt, fs_emg // 2)
    t_emg = [i / fs_emg for i in range(len(emg_raw))]
    if any([plot_raw, plot_clean, plot_env]):
        _, axis_t = plt.subplots()
        if plot_raw:
            axis_t.plot(t_emg, emg_raw, color="tab:cyan")
        if plot_clean:
            axis_t.plot(t_emg, emg_filt, color="tab:blue")
        if plot_env:
            axis_t.plot(t_emg, emg_env, color="tab:red")
        axis_t.grid(True)
        plt.show()
    if plot_power_spectrum:
        _, axis_f = plt.subplots()
        vis.show_power_spectrum(emg_raw, fs_emg, t_emg[-1], signal_unit="uV")
        axis_f.grid(True)
    return emg_filt, emg_env


def ecg_removal_gating(
    emg_raw: np.ndarray,
    ecg_peaks_idxs: np.ndarray,
    gate_width_samples: int,
    method: int = 3,
    ecg_shift: int | None = None,
) -> np.ndarray:
    """Eliminate the ECG peaks from the emg_raw signal.

    :param emg_raw: 1 dimensional emg signal to gate
    :type emg_raw: ~numpy.ndarray
    :param ecg_peaks_idxs: List of ECG peak sample numbers to gate.
    :type ecg_peaks_idxs: ~numpy.ndarray
    :param gate_width_samples: Number of samples to gate
    :type gate_width_samples: int
    :param fs: Sampling rate of emg_raw
    :type fs: int
    :param method: gating method. See the ecg_removal.gating function.
    :type method: int
    :param ecg_shift: Shift gate windows relative to detected peaks in samples.
    :type ecg_shift: int

    :returns emg_gated: The gated EMG signal
    :rtype emg_gated: numpy.ndarray
    """
    if len(emg_raw.shape) > 1:
        msg = "emg_raw should be a 1-D array"
        raise ValueError(msg)

    if ecg_shift is None:
        ecg_shift = 0

    gate_peaks_idxs = ecg_peaks_idxs + ecg_shift

    # Gate ECG and EMG signal
    # Fill methods: 0: Zeros, 1: Interpolate start-end, 2: Average prior data
    # 3: Moving average
    return ecg_rm.gating(
        emg_raw, gate_peaks_idxs, gate_width=gate_width_samples, method=method
    )
