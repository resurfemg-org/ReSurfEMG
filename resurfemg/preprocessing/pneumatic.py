"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for processing pneumatic data.
"""

import numpy as np
from scipy.signal import medfilt


def zero_cros_flow(flow, flow_threshold=0.3):
    """
    Function to find the onset of inspiratory,i.e., positive, flow in the
    flow signal. Flow onsets are defined as the points where the flow signal
    crosses the zero line to become positive and the flow signal reaches
    above the flow_threshold during the breath.
    :param flow: The flow signal.
    :type flow: ~numpy.ndarray
    :param flow_threshold: The threshold for the flow signal.
    :type flow_threshold: float

    :return zc_idxs: zero crossings indices
    :rtype: tuple
    :return zc_candi: zero crossings candidates indices
    :rtype: tuple

    """
    zc_candi = np.argwhere(np.diff(np.sign(flow))).flatten() + 1
    mask_positive = np.zeros_like(zc_candi, dtype=bool)
    for i, (_zc, _zc_next) in enumerate(zip(zc_candi[:-1], zc_candi[1:])):
        # Check if the maximum flow in the range is above the threshold and if
        # the flow signal is positive at the next sample.
        _flow_seg = flow[_zc:_zc_next]
        if np.max(_flow_seg) > flow_threshold and flow[_zc + 1] > 0:
            mask_positive[i] = True

    zc_idxs = zc_candi[mask_positive]
    return zc_idxs, zc_candi


def volume_computation(t, flow, fs, zc_idxs, method):
    """
    Function to compute the volume signal from the flow signal. The volume
    signal is computed by integrating the flow signal. The volume signal is
    then baseline corrected using the end-expiratory samples. The end-
    expiratory samples are selected around the zero crossings of the flow
    signal. The baseline correction method can be set to "Last point", "Last
    points" or "Mask". The "Last point" option uses the zero crossing only.
    "Last points" uses the samples up to 0.1 seconds before the zero crossing.
    "Mask" selects the samples where the flow signal is below 0.01 L/s and the
    raw volume signal is below 0.1 L.
    :param t: The time vector.
    :type t: ~numpy.ndarray
    :param flow: The flow signal.
    :type flow: ~numpy.ndarray
    :param fs: The sampling frequency.
    :type fs: float
    :param zc_idxs: The zero crossing indices.
    :type zc_idxs: ~numpy.ndarray
    :param method: The baseline correction method.
    :type method: str

    :return volume: The volume signal.
    :rtype: ~numpy.ndarray
    :return vol_baseline: The baseline of the volume signal.
    :rtype: ~numpy.ndarray
    :return volume_raw: The raw volume signal.
    :rtype: ~numpy.ndarray
    :return end_exp_idxs: End-expiratory samples used for baseline correction.
    :rtype: ~numpy.ndarray
    """
    volume_raw = np.cumsum(flow) / fs
    volume_raw = np.insert(volume_raw, 0, 0)[:len(t)]

    match method:
        case "Last point":
            end_exp_idxs = np.array((zc_idxs), dtype=int)
            vol_zc = medfilt(volume_raw[end_exp_idxs], kernel_size=3)

        case "Last points":
            end_exp_idxs = []
            zc_start_idxs = np.clip(
                zc_idxs - int(0.1 * fs) - 1, 0, len(flow)).astype(int)
            zc_end_idxs = zc_idxs
            delta = np.zeros(volume_raw.shape, dtype=np.int64)
            np.add.at(delta, zc_start_idxs, 1)  # Increment + 1 at start idxs
            np.add.at(delta, zc_end_idxs, -1)   # Decrement - 1 at end idxs
            end_exp_idxs = np.where(np.cumsum(delta) > 0)[0]

            vol_zc = medfilt(
                volume_raw[end_exp_idxs], kernel_size=int(0.1 * fs) - 1
            )
        case "Mask":
            # Select the samples where the flow signal is below 0.01 L/s and
            # the raw volume signal is below 0.1 L.
            end_exp_mask = (flow < 0) & (flow > -0.01) & (volume_raw < 0.1)
            end_exp_idxs = np.argwhere(end_exp_mask).flatten()
            vol_zc = medfilt(volume_raw[end_exp_idxs], kernel_size=5)

    vol_baseline = np.interp(t, t[end_exp_idxs], vol_zc)
    volume = volume_raw - vol_baseline

    return volume, vol_baseline, volume_raw, end_exp_idxs
