"""
Copyright 2026 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for estimating the time constant (tau) of the
respiratory system from ventilator data.
"""
import logging

import numpy as np
import statsmodels.api as sm
from resurfemg.postprocessing.baseline import moving_baseline


logger = logging.getLogger(__name__)


def v_rs_estimation(
        volume,
        flow,
        tau,
        expiration_mask,
        window_s,
        step_s,
        set_percentile=33,
        omit_nan=True
        ):
    """
    Estimate the Vrs signal for the integrated equation of motion. See the
    paper by Graßhoff et al. (2023) for details on the method.
    :param volume: The volume signal.
    :type volume: ~numpy.ndarray
    :param flow: The flow signal.
    :type flow: ~numpy.ndarray
    :param tau: The time constant (tau) of the respiratory system.
    :type tau: float
    :param expiration_mask: A boolean mask indicating the samples that are
    during expiration.
    :type expiration_mask: ~numpy.ndarray
    :param window_s: The window size in samples for the moving baseline
    :type window_s: float
    :param step_s: The step size in samples for the moving baseline
    :type step_s: float
    :param set_percentile: The percentile to use for the moving baseline
    :type set_percentile: float
    :return v_rs: The estimated Vrs signal.
    :rtype: ~numpy.ndarray
    """
    v_rs_hat = - volume - tau * flow
    v_rs_hat_est = np.where(expiration_mask, v_rs_hat, np.nan)
    v0_hat = moving_baseline(
        v_rs_hat_est,
        window_s,
        step_s,
        set_percentile=set_percentile,
        omit_nan=omit_nan,
    )
    v_rs = volume + tau * flow + v0_hat
    return v_rs


def compliance_estimation(df_ieqm, t=1.345, keys=None, verbose=False):
    """
    Estimate the compliance and non-mechanical component of the integrated
    equation of motion using robust linear regression.
    :param df_ieqm: The integrated equation of motion data.
    :type df_ieqm: ~pandas.DataFrame
    :param t: The tuning parameter for the Huber loss function.
    :type t: float
    :param keys: key map for the PTP, VTP, and ETP.
    :type keys: dict
    :param verbose: Whether to print the summary of the model.
    :type verbose: bool
    :return e_est: The estimated elastance
    :rtype: np.float
    :return nmc_est: The estimated neuro-mechanical coupling
    :rtype: np.float
    :return mdl_C: The estimated model.
    :rtype: statsmodels.robust.robust_linear_model.RLMResults
    """
    if keys is None:
        keys = {'PTP': 'PTPaw-PTPpeep', 'VTP': 'VTPrs', 'ETP': 'ETPmus'}
    x_breath = np.column_stack(
        (df_ieqm[keys['PTP']], df_ieqm[keys['ETP']]))
    rlm_model_breath = sm.RLM(
        df_ieqm[keys['VTP']],
        x_breath, M=sm.robust.norms.HuberT(t=t)
    ).fit(conv="coefs", cov="H3")
    beta_0, beta_1 = rlm_model_breath.params
    c_est = beta_0 * 1000
    nmc_est = beta_1 / beta_0
    mdl_c = rlm_model_breath
    if verbose:
        logger.info(rlm_model_breath.summary())
        logger.info("Compliance is: %s mL/cmH2O", c_est)
        logger.info("NMC is: %s cmH2O/uV", nmc_est)
    return c_est, nmc_est, mdl_c


def p_mus_estimation(p_aw, peep, v_rs, c):
    """
    Estimate the Pmus signal using the integrated equation of motion:
    Pmus,est = 1/C * Vrs - (Paw - PEEP)
    :param p_aw: The airway pressure signal.
    :type p_aw: ~numpy.ndarray
    :param peep: The positive end-expiratory pressure (PEEP) signal.
    :type peep: ~numpy.ndarray
    :param v_rs: The estimated Vrs signal.
    :type v_rs: ~numpy.ndarray
    :param c: The compliance of the respiratory system in mL/cmH2O.
    :type c: float
    :return p_mus_pred: The estimated Pmus signal.
    :rtype: ~numpy.ndarray
    """
    return 1000 * v_rs / c - (p_aw - peep)
