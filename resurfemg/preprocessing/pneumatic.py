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

    :return zc:  zero crossings indices
    :rtype: tuple
    :return zc_candidate:  zero crossings candidates indices
    :rtype: tuple

    """
    zc_candidate = np.where(np.diff(np.sign(flow)))[0]
    mask_positive = np.zeros_like(zc_candidate, dtype=bool)
    for i in range(len(zc_candidate)-1):
        # Check if the flow signal is above the threshold in the range between
        # the zero crossings.
        if zc_candidate[i+1] < len(flow):
            post_range = flow[zc_candidate[i]:zc_candidate[i+1]+1]
        else:
            post_range = flow[zc_candidate[i]:-1]
        # Check if the maximum flow in the range is above the threshold and if
        # the flow signal is positive at the next sample.
        max_flow = np.max(post_range)
        sample_next = zc_candidate[i]+1
        if max_flow > flow_threshold and flow[sample_next] > 0:
            mask_positive[i] = True

    zc = zc_candidate[mask_positive]
    return zc, zc_candidate


def volume_computation(t, flow, fs, zc, method):
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
    :param zc: The zero crossings.
    :type zc: ~numpy.ndarray
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
            end_exp_idxs = np.array((zc), dtype=int)
            vol_zc = medfilt(volume_raw[end_exp_idxs], kernel_size=3)

        case "Last points":
            end_exp_idxs = []
            for i, zc in enumerate(zc):
                zc_begin = max(0, zc - int(0.1 * fs))
                zc_array = np.arange(zc_begin, zc + 1)
                end_exp_idxs.append(zc_array)

            end_exp_idxs = [
                item for sublist in end_exp_idxs for item in sublist]
            end_exp_idxs = np.array((end_exp_idxs), dtype=int).flatten()

            vol_zc = medfilt(
                volume_raw[end_exp_idxs], kernel_size=int(0.1 * fs) - 1
            )
        case "Mask":
            # Select the samples where the flow signal is below 0.01 L/s and
            # the raw volume signal is below 0.1 L.
            end_exp_mask = (flow < 0) & (flow > -0.01) & (volume_raw < 0.1)
            end_exp_idxs = np.arange(0, len(flow))[end_exp_mask]
            vol_zc = medfilt(
                volume_raw[end_exp_idxs], kernel_size=5
            )

    vol_baseline = np.interp(t, t[end_exp_idxs], vol_zc)
    volume = volume_raw - vol_baseline

    return volume, vol_baseline, volume_raw, end_exp_idxs
