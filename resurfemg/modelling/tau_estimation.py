"""
Copyright 2026 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for estimating the time constant (tau) of the
respiratory system.
"""

import logging
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from resurfemg.preprocessing.pneumatic import zero_cros_flow


logger = logging.getLogger(__name__)


def tau_mask(p_aw, flow, volume, peep, fs, zc_idxs=None, **kwargs):
    """
    Function to create a mask for the breaths based on various criteria. The
    criteria include the difference between Paw and PEEP, the flow signal, and
    the quality of the breaths. The function returns a boolean mask indicating
    the samples that meet the criteria. Optional keyword arguments can be
    provided to customize the criteria: theta_paw_peep, theta_flow,
    flow_threshold (zero-crossing), min_duration, min_tv, max_v0, and min_vol.
    :param p_aw: The airway pressure signal.
    :type p_aw: ~numpy.ndarray
    :param flow: The flow signal.
    :type flow: ~numpy.ndarray
    :param volume: The volume signal.
    :type volume: ~numpy.ndarray
    :param peep: The PEEP signal.
    :type peep: ~numpy.ndarray
    :param fs: The sampling rate of the signals.
    :type fs: int
    :param zc_idxs: The indices of the zero-crossings of the flow signal.
    :type zc_idxs: ~numpy.ndarray
    :param kwargs: Additional keyword arguments for the criteria.
    :type kwargs: dict

    :return mask: A boolean mask indicating the samples that meet the criteria.
    :rtype: ~numpy.ndarray
    """
    def _mask_paw_min_peep(p_aw, peep, theta_paw_peep=0.5):
        """
        Select samples where the difference between Paw and PEEP is less than
        the threshold theta_paw_peep.
        :param p_aw: The airway pressure signal.
        :type p_aw: ~numpy.ndarray
        :param peep: The PEEP signal.
        :type peep: ~numpy.ndarray
        :param theta_paw_peep: difference threshold for between Paw and PEEP.
        :type theta_paw_peep: float

        :return mask: A boolean mask indicating where the difference between
        Paw and PEEP is less than the threshold.
        :rtype: ~numpy.ndarray
        """
        return np.abs(p_aw - peep) < theta_paw_peep

    def _mask_expiration(flow, theta_flow=0.0):
        """
        Select samples where the flow signal is negative (i.e., during
        expiration).
        :param flow: The flow signal.
        :type flow: ~numpy.ndarray
        :param theta_flow: The threshold for the flow signal.
        :type theta_flow: float

        :return mask_inspiration, mask_expiration: Boolean masks indicating
        the inspiration and expiration phases.
        :rtype: ~numpy.ndarray, ~numpy.ndarray
        """
        mask_inspiration = flow >= theta_flow
        mask_expiration = flow < theta_flow
        return mask_inspiration, mask_expiration

    def _quality_mask(
            breath_df,
            min_duration=1.5,
            min_tv=0.2,
            max_v0=0.1,
            min_vol=-0.05):
        """
        Create a quality mask for the breaths based on various criteria.
        :param breath_df: A DataFrame containing the breath identifiers and
        associated signals.
        :type breath_df: ~pandas.DataFrame
        :param min_duration: The minimum duration of a breath in seconds.
        :type min_duration: float
        :param min_tv: The minimum tidal volume of a breath in liters.
        :type min_tv: float
        :param max_v0: The maximum volume at the end of a breath.
        :type max_v0: float
        :param min_vol: The minimum volume during a breath.
        :type min_vol: float
        """
        mask = np.zeros(len(breath_df), dtype=bool)
        breath_ids = np.unique(breath_df.breath_id.dropna())
        for _, breath_id in enumerate(breath_ids):
            mask_breath = breath_df.breath_id == breath_id
            vol_segment = breath_df.volume[mask_breath].to_numpy()
            sample_segment = breath_df.sample_idx[mask_breath].to_numpy()

            time_bool = (sample_segment[-1] - sample_segment[0]) > min_duration
            vol_max_bool = max(vol_segment) > min_tv
            v0_bool = vol_segment[-1] < max_v0
            min_vol_bool = min(vol_segment) > min_vol
            if time_bool and vol_max_bool and v0_bool and min_vol_bool:
                mask[mask_breath] = True
        return mask

    mask_paw_peep = _mask_paw_min_peep(
        p_aw, peep, theta_paw_peep=kwargs.get('theta_paw_peep', 0.5))

    mask_insp, mask_exp = _mask_expiration(
        flow, theta_flow=kwargs.get('theta_flow', 0.0))

    if zc_idxs is None:
        zc_idxs, _ = zero_cros_flow(
            flow, flow_threshold=kwargs.get('flow_threshold', 0.2))
    breath_df = _breath_id(flow, zc_idxs, volume)
    mask_quality = _quality_mask(
            breath_df,
            min_duration=kwargs.get('min_duration', 1.5),
            min_tv=kwargs.get('min_tv', 0.2),
            max_v0=kwargs.get('max_v0', 0.1),
            min_vol=kwargs.get('min_vol', -0.05)
        )

    mask = mask_paw_peep & mask_exp & mask_quality
    df_submask = pd.DataFrame({
        'paw_peep': mask_paw_peep,
        'inspiration': mask_insp,
        'expiration': mask_exp,
        'quality': mask_quality,
    })
    return breath_df, mask, df_submask


def _breath_id(flow, zc_idxs, volume):
    """
    Assign a unique identifier to each breath.
    :param flow: The flow signal.
    :type flow: ~numpy.ndarray
    :param zc_idxs: The indices of the zero-crossings of the flow signal.
    :type zc_idxs: ~numpy.ndarray

    :return df_breaths: A DataFrame containing the breath identifiers and
    associated signals.
    :rtype: ~pandas.DataFrame
    """
    # Initialize arrays
    sample_array = np.arange(len(flow))
    breath_array = np.full(flow.shape, np.nan)
    flow_output = np.full(flow.shape, np.nan)
    volume_output = np.full(flow.shape, np.nan)
    # Calculate breath indices
    breath_delta = np.zeros(flow.shape, dtype=int)
    np.add.at(breath_delta, zc_idxs, 1)  # Increment + 1 at start indices
    breath_idxs = np.cumsum(breath_delta) - 1   # Breath IDs start from 0
    # Detect expiration segments
    delta = np.zeros(flow.shape, dtype=np.int64)
    np.add.at(delta, zc_idxs[:-1], 1)    # Increment + 1 at start indices
    np.add.at(delta, zc_idxs[1:] + 1, -1)    # Decrement - 1 at end indices
    exp_idxs = np.argwhere(np.cumsum(delta) > 0)
    # Assign breath IDs and flow/volume values for expiration segments
    breath_array[exp_idxs] = breath_idxs[exp_idxs]
    flow_output[exp_idxs] = flow[exp_idxs]
    volume_output[exp_idxs] = volume[exp_idxs]
    breath_df = pd.DataFrame({
        'breath_id': pd.Series(breath_array),
        'flow': pd.Series(flow_output),
        'volume': pd.Series(volume_output),
        'sample_idx': pd.Series(sample_array)
        })

    return breath_df


def tau_switch_smf(df, c=4.685, verbose=False):
    """
    Estimate the time constant (tau) using a robust linear model with Tukey's
    biweight function. The model is fitted to the end-expiratory slope between
    volume and flow data. The breath ID is included as a categorical variable
    to account for breath-specific offsets.
    :param df: A DataFrame containing the flow, volume, and breath identifier.
    :type df: ~pandas.DataFrame
    :param c: The tuning constant for Tukey's biweight function.
    :type c: float
    :param verbose: If True, print the model summary and estimated tau.
    :type verbose: bool

    :return tau, mdl: The estimated time constant and the fitted model.
    :rtype: float, ~statsmodels.regression.linear_model.RLMResults
    """
    mdl = smf.rlm(
        'volume ~ flow + C(breath_id)',
        data=df,
        M=sm.robust.norms.TukeyBiweight(c=c)
    ).fit()
    tau = -1 * mdl.params.iloc[-1]
    if verbose:
        logger.info(f"Estimated Tau is: {tau}")
        logger.info(mdl.summary())
    return tau, mdl
