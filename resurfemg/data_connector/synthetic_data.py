"""Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to create synthetic data with several methods.
"""

import numpy as np
import pandas as pd
from scipy import signal

from resurfemg.preprocessing import filtering as filt


def respiratory_pattern_generator(
    t_end=7 * 60,
    fs=2048,
    rr=22,
    ie_ratio=1 / 2,
    t_p_occs=None,
):
    """This function simulates an on/off respiratory muscle activiation pattern
    for generating a synthetic EMG.
    ---------------------------------------------------------------------------
    :param t_end: end time
    :type t_end: float
    :param fs: Sampling rate
    :type fs: int
    :param rr: Respiratory rate (/min)
    :type rr: float
    :param ie_ratio: Ratio between inspiratory and expiratory time
    :type ie_ratio: float
    :param t_p_occs: Timing of occlusions (s)
    :type t_p_occs: float

    :returns respiratory_pattern: The simulated on/off respiratory muscle
        pattern.
    :rtype respiratory_pattern: np.array[float]
    """
    ie_fraction = ie_ratio / (ie_ratio + 1)
    if t_p_occs is None:
        t_occs = np.array([])
    else:
        t_occs = np.floor(np.array(t_p_occs) * rr / 60) * 60 / rr

    for _, t_occ in enumerate(t_occs):
        if t_end < (t_occ + 60 / rr):
            print(
                "t="
                + str(t_occ)
                + ":t_occ"
                + "should be at least a full respiratory cycle from t_end."
            )
    # time axis
    t_emg = np.array([i / fs for i in range(int(t_end * fs))])

    # reference signal pattern generator
    respiratory_pattern = (
        signal.square(t_emg * rr / 60 * 2 * np.pi + 0.5, ie_fraction) + 1
    ) / 2
    for _, t_occ in enumerate(t_occs):
        i_occ = int(t_occ * fs)
        blocker = np.arange(int(fs * 60 / rr) + 1) / fs * rr / 60 * 2 * np.pi
        squared_wave = (signal.square(blocker, ie_fraction) + 1) / 2
        respiratory_pattern[i_occ : i_occ + int(fs * 60 / rr) + 1] = (
            squared_wave
        )
    return respiratory_pattern


def simulate_muscle_dynamics(
    block_pattern,
    fs=2048,
    tau_mus_up=0.3,
    tau_mus_down=0.3,
):
    """This function simulates an respiratory muscle activation dynamics for
    generating a synthetic EMG.
    ---------------------------------------------------------------------------
    :param block_pattern: Simulated on/off respiratory muscle pattern.
    :type block_pattern: ~np.array[float]
    :param fs: Sampling rate
    :type fs: int
    :param tau_mus_up: Muscle contraction time constant (s)
    :type tau_mus_up: float
    :param tau_mus_down: Muscle relaxation time constant (s)
    :type tau_mus_down: float

    :returns muscle_activation: The simulated muscle activation pattern.
    :rtype muscle_activation: np.array[float]
    """
    # simulate up- and downslope dynamics of EMG
    muscle_activation = np.zeros((len(block_pattern),))
    for i in range(1, len(block_pattern)):
        pat = muscle_activation[i - 1]
        if (block_pattern[i - 1] - pat) > 0:
            muscle_activation[i] = pat + (
                (block_pattern[i - 1] - pat) / (tau_mus_up * fs)
            )
        else:
            muscle_activation[i] = pat + (
                (block_pattern[i - 1] - pat) / (tau_mus_down * fs)
            )
    return muscle_activation


def simulate_ventilator_data(p_mus, fs_vent=100, t_occ_bool=None, **kwargs):
    """This function simulates ventilator data with occlusion manouevers based on
    the provided `p_mus` and adds noise to the signal.
    ---------------------------------------------------------------------------
    :param p_mus: Respiratory muscle pressure
    :type p_mus: numpy.ndarray[float]
    :param fs_vent: Ventilator sampling rate
    :type fs_vent: int
    :param t_occ_bool: Boolean array. Is true when a Pocc manoeuver is done
    :type t_occ_bool: numpy.ndarray[bool]

    :returns y_vent: The synthetic ventilator pressure, flow and volume
    :rtype y_vent: np.array[float]
    """

    def evaluate_ventilator_status(
        idx,
        y_vent,
        vent_settings,
        vent_status,
    ):
        """Define the ventilator status (active support, sensitive for trigger)
        based on the ventilator settings and ventilator flow.
        -----------------------------------------------------------------------
        :param idx: The index to evaluate the ventilator status
        :type idx: int
        :param y_vent: The ventilator pressure, flow and volume
        :type y_vent: numpy.ndarray
        :param vent_settings: The ventilator settings
        :type vent_settings: dict
        :param vent_status: The current ventilator status
        :type vent_status: dict

        :returns vent_status: The new ventilator status
        :rtype vent_status: dict
        """
        if (vent_status["sensitive"] is True) and (
            60 * y_vent[1, idx] > vent_settings["flow_trigger"]
        ):
            vent_status["active"] = True
            vent_status["sensitive"] = False

        if vent_status["active"] and y_vent[1, idx] > vent_status["F_max"]:
            vent_status["p_set"] = vent_settings["dp"]
            vent_status["F_max"] = y_vent[1, idx]
        elif (
            y_vent[1, idx] < vent_settings["flow_cycle"] * vent_status["F_max"]
        ) and vent_status["active"]:
            vent_status["active"] = False
            vent_status["p_set"] = 0
        return vent_status

    lung_mechanics = {
        "c": 0.050,
        "r": 5,
    }
    vent_settings = {
        "dp": 5,
        "peep": 5,
        "flow_cycle": 0.25,  # Fraction F_max
        "flow_trigger": 2,  # L/min
        "tau_dp_up": 10,
        "tau_dp_down": 5,
    }
    for key, value in kwargs.items():
        if key in lung_mechanics:
            lung_mechanics[key] = value
        elif key in vent_settings:
            vent_settings[key] = value
        else:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    if t_occ_bool is None:
        t_occ_bool = np.zeros(p_mus.shape, dtype=bool)

    # Simulate up- and downslope dynamics of airway pressure
    p_noise_ma = (
        0
        * pd.Series(np.random.normal(0, 2, size=(len(p_mus),)))
        .rolling(fs_vent, min_periods=1, center=True)
        .mean()
        .values
    )

    vent_status = {
        "p_set": 0,
        "active": False,
        "sensitive": True,
        "F_max": 0,
    }
    p_dp = -p_mus
    y_vent = np.zeros((3, len(p_mus)))
    for i in range(1, len(p_mus)):
        vent_status = evaluate_ventilator_status(
            idx=i - 1,
            y_vent=y_vent,
            vent_settings=vent_settings,
            vent_status=vent_status,
        )
        dp_step = vent_status["p_set"] - (p_dp[i - 1] + p_noise_ma[i - 1])
        if t_occ_bool[i]:
            vent_status["active"] = False
            vent_status["sensitive"] = False
            vent_status["p_set"] = 0
            vent_status["F_max"] = 0

            # Occlusion pressure results into negative airway pressure:
            dp_step = (
                -np.mean(p_mus[i - int(1 * fs_vent / 2) : int(i - 1)])
                - p_dp[i - 1]
            )
            p_dp[i] = p_dp[i - 1] + dp_step / (vent_settings["tau_dp_up"])
            # During occlusion manoeuvre: flow and volume are zero
            y_vent[1:2, i] = 0
        else:
            if (vent_status["p_set"] - p_dp[i - 1]) > 0:
                p_dp[i] = p_dp[i - 1] + dp_step / vent_settings["tau_dp_up"]
            else:
                p_dp[i] = p_dp[i - 1] + dp_step / vent_settings["tau_dp_down"]
            # Calculate flows and volumes from equation of motion:
            y_vent[1, i] = (
                (p_dp[i - 1] + p_mus[i - 1])
                - y_vent[2, i - 1] / lung_mechanics["c"]
            ) / lung_mechanics["r"]
            y_vent[2, i] = y_vent[2, i - 1] + y_vent[1, i] * 1 / fs_vent
            if (vent_status["sensitive"] is False) and (y_vent[1, i] < 0):
                vent_status["sensitive"] = True
                vent_status["F_max"] = 0
    y_vent[0, :] = vent_settings["peep"] + p_dp

    return y_vent


def simulate_emg(
    muscle_activation, fs_emg=2048, emg_amp=5, drift_amp=100, noise_amp=2
):
    """This function simulates an surface respiratory emg based on the provided
    `muscle_activation` and adds noise and drift to the signal. No ecg
    component is included, but can be added later.
    ---------------------------------------------------------------------------
    :param muscle_activation: The muscle activation pattern
    :type muscle_activation: float
    :param emg_amp: Approximate EMG-RMS amplitude (uV)
    :type emg_amp: float
    :param drift_amp: Approximate drift RMS amplitude (uV)
    :type drift_amp: float
    :param noise_amp: Approximate baseline noise RMS amplitude (uV)
    :type noise_amp: float

    :returns emg_raw: The raw synthetic EMG without the ECG added.
    :rtype: np.array[float]
    """
    n_samp = len(muscle_activation)
    # make respiratory EMG component
    part_emg = muscle_activation * np.random.normal(0, 2, size=(n_samp,))

    # make noise and drift components
    part_noise = np.random.normal(0, 2 * noise_amp, size=(n_samp,))
    part_drift = np.zeros((n_samp,))

    f_high = 0.05
    white_noise = np.random.normal(
        0, drift_amp, size=(n_samp + int(1 / f_high) * fs_emg,)
    )
    part_drift_tmp = filt.emg_lowpass_butter(
        white_noise, f_high, fs_emg, order=3
    )
    part_drift = part_drift_tmp[int(1 / f_high) * fs_emg :] / f_high

    # mix channels, could be remixed with an ecg
    emg_raw = emg_amp * part_emg + part_drift + part_noise
    return emg_raw
