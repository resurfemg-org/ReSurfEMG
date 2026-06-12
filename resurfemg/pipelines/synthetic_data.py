"""This file contains functions to generate mixed (signal and noise) synthetic data.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

import logging
import math
from pathlib import Path

import neurokit2 as nk
import numpy as np

import resurfemg.data_connector.synthetic_data as synth

logger = logging.getLogger(__name__)


def simulate_raw_emg(t_end: int, fs_emg: int, emg_amp: float = 5, rr: float = 22, **kwargs) -> np.ndarray:
    """Generate realistic synthetic respiratory EMG data remixed with ECG.

    Args:
        t_end (int): Length of synthetic EMG tracing in seconds.
        fs_emg (int): Sampling rate.
        emg_amp (float): EMG amplitude.
        rr (float): Respiratory rate (/min).
        **kwargs: Optional arguments — ie_ratio, tau_mus_up, tau_mus_down,
            t_p_occs, drift_amp, noise_amp, ecg_acceleration, ecg_amplitude. See
            data_connector.synthetic_data respiratory_pattern_generator,
            simulate_muscle_dynamics, and simulate_emg functions for specifics.

    Returns:
        numpy.ndarray: The realistic synthetic EMG.
    """
    sim_parameters = {
        "ie_ratio": 1 / 2,  # ratio btw insp + expir phase
        "tau_mus_up": 0.3,
        "tau_mus_down": 0.3,
        "t_p_occs": [],
        "drift_amp": 100,
        "noise_amp": 2,
        "heart_rate": 80,
        "ecg_acceleration": 1.5,
        "ecg_amplitude": 200,
    }
    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    respiratory_pattern = synth.respiratory_pattern_generator(
        t_end=t_end,
        fs=fs_emg,
        rr=rr,
        ie_ratio=sim_parameters["ie_ratio"],
        t_p_occs=sim_parameters["t_p_occs"],
    )
    muscle_activation = synth.simulate_muscle_dynamics(
        block_pattern=respiratory_pattern,
        fs=fs_emg,
        tau_mus_up=sim_parameters["tau_mus_up"],
        tau_mus_down=sim_parameters["tau_mus_down"],
    )
    emg_sim = synth.simulate_emg(
        muscle_activation=muscle_activation,
        fs_emg=fs_emg,
        emg_amp=emg_amp,
        drift_amp=sim_parameters["drift_amp"],
        noise_amp=sim_parameters["noise_amp"],
    )
    sim_hr = sim_parameters["heart_rate"] / sim_parameters["ecg_acceleration"]
    fs_ecg = int(fs_emg * sim_parameters["ecg_acceleration"])
    ecg_t_end = math.ceil(t_end / sim_parameters["ecg_acceleration"])
    ecg_sim = nk.ecg_simulate(
        duration=ecg_t_end,
        sampling_rate=fs_ecg,
        heart_rate=sim_hr,
    )
    ecg_sim = ecg_sim[: len(emg_sim)]
    return sim_parameters["ecg_amplitude"] * ecg_sim + emg_sim


def synthetic_emg_cli(n_emg: int, output_directory: str, **kwargs) -> None:
    """Generate syntetic respiratory EMG data remixed with ECG using the cli.

    Generate realistic, single lead, synthetic respiratory EMG data remixed
    with ECG through command line using the cli.

    Args:
        n_emg (int): Number of EMGs to simulate.
        output_directory (str): File directory where synthetic EMG will be saved.
        **kwargs: Optional arguments — t_end, fs_emg, emg_amp, rr, ie_ratio,
            tau_mus_up, tau_mus_down, t_p_occs, drift_amp, noise_amp, heart_rate,
            ecg_acceleration, ecg_amplitude. See data_connector.synthetic_data
            respiratory_pattern_generator, simulate_muscle_dynamics, and
            simulate_emg functions for specifics.

    Returns:
        None
    """
    sim_parameters = {
        "t_end": 7 * 60,
        "fs_emg": 2048,  # hertz
        "rr": 22,  # respiratory rate /min
        "ie_ratio": 1 / 2,  # ratio btw insp + expir phase
        "tau_mus_up": 0.3,
        "tau_mus_down": 0.3,
        "t_p_occs": [],
        "drift_amp": 100,
        "noise_amp": 2,
        "heart_rate": 80,
        "ecg_acceleration": 1.5,
        "ecg_amplitude": 200,
    }
    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    for i in range(n_emg):
        emg_raw = simulate_raw_emg(**sim_parameters)
        out_fname = Path(output_directory) / f"emg_{i}"
        if not Path(output_directory).exists():
            Path(output_directory).mkdir(parents=True, exist_ok=True)
        np.save(out_fname, emg_raw)
        logger.info("File(s) saved to %s.", output_directory)


def simulate_ventilator_data(
    t_end: int,
    fs_vent: int,
    p_mus_amp: float = 5,
    rr: float = 22,
    dp: float = 5,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate realistic synthetic ventilator tracings.

    Args:
        t_end (int): Length of synthetic ventilator tracings in seconds.
        fs_vent (int): Sampling rate.
        p_mus_amp (float): Respiratory muscle pressure amplitude (positive).
        rr (float): Respiratory rate.
        dp (float): Driving pressure.
        **kwargs: Optional arguments — ie_ratio, tau_mus_up, tau_mus_down,
            t_p_occs, c, r, peep, flow_cycle, flow_trigger, tau_dp_up,
            tau_dp_down. See data_connector.synthetic_data
            respiratory_pattern_generator, simulate_muscle_dynamics,
            and simulate_ventilator_data functions for specifics.

    Returns:
        tuple:
            - numpy.ndarray: The realistic synthetic ventilator data.
            - numpy.ndarray: The respiratory muscle pressure.
    """
    sim_parameters = {
        "ie_ratio": 1 / 2,  # ratio btw insp + expir phase
        "tau_mus_up": 0.3,
        "tau_mus_down": 0.3,
        "t_p_occs": [],
        "c": 0.050,
        "r": 5,
        "peep": 5,
        "flow_cycle": 0.25,  # Fraction F_max
        "flow_trigger": 2,  # L/min
        "tau_dp_up": 10,
        "tau_dp_down": 5,
    }

    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    respiratory_pattern = synth.respiratory_pattern_generator(
        t_end=t_end,
        fs=fs_vent,
        rr=rr,
        ie_ratio=sim_parameters["ie_ratio"],
        t_p_occs=sim_parameters["t_p_occs"],
    )
    p_mus = p_mus_amp * synth.simulate_muscle_dynamics(
        block_pattern=respiratory_pattern,
        fs=fs_vent,
        tau_mus_up=sim_parameters["tau_mus_up"],
        tau_mus_down=sim_parameters["tau_mus_down"],
    )
    t_occ_bool = np.zeros(p_mus.shape, dtype=bool)
    for t_occ in sim_parameters["t_p_occs"]:
        t_occ_bool[int((t_occ - 1) * fs_vent) : int((t_occ + 1 / rr * 60) * fs_vent)] = True
    lung_mechanics = {
        "c": 0.050,
        "r": 5,
    }
    vent_settings = {
        "dp": dp,
        "peep": 5,
        "flow_cycle": 0.25,  # Fraction F_max
        "flow_trigger": 2,  # L/min
        "tau_dp_up": 10,
        "tau_dp_down": 5,
    }
    for key, value in sim_parameters.items():
        if key in lung_mechanics:
            lung_mechanics[key] = value
        elif key in vent_settings:
            vent_settings[key] = value

    y_vent = synth.simulate_ventilator_data(
        **{
            "p_mus": p_mus,
            "dp": dp,
            "fs_vent": fs_vent,
            "t_occ_bool": t_occ_bool,
            **lung_mechanics,
            **vent_settings,
        }
    )
    return y_vent, p_mus


def synthetic_ventilator_data_cli(n_datasets: int, output_directory: str, **kwargs) -> None:
    """Generate realistic synthetic ventilator data through cli.

    Generate realistic synthetic ventilator data through
    command line using the cli.

    Args:
        n_datasets (int): Number of datasets to simulate.
        output_directory (str): File directory where synthetic data will be saved.
        **kwargs: Optional arguments — t_end, fs_vent, p_mus_amp, rr, dp,
            ie_ratio, tau_mus_up, tau_mus_down, t_p_occs, c, r, peep, flow_cycle,
            flow_trigger, tau_dp_up, tau_dp_down. See data_connector.synthetic_data
            respiratory_pattern_generator, simulate_muscle_dynamics, and
            simulate_ventilator_data functions for specifics.

    Returns:
        None
    """
    sim_parameters = {
        "t_end": 7 * 60,
        "fs_vent": 100,
        "p_mus_amp": 5,
        "rr": 22,
        "dp": 5,
        "ie_ratio": 1 / 2,  # ratio btw insp + expir phase
        "tau_mus_up": 0.3,
        "tau_mus_down": 0.3,
        "t_p_occs": [],
        "c": 0.050,
        "r": 5,
        "peep": 5,
        "flow_cycle": 0.25,  # Fraction F_max
        "flow_trigger": 2,  # L/min
        "tau_dp_up": 10,
        "tau_dp_down": 5,
    }
    for key, value in kwargs.items():
        if key in sim_parameters:
            sim_parameters[key] = value
        else:
            msg = f"kwarg `{key}` not available."
            raise UserWarning(msg)

    for i in range(n_datasets):
        y_vent, p_mus = simulate_ventilator_data(**sim_parameters)
        y_sig = np.vstack((y_vent, p_mus))
        out_fname = Path(output_directory) / ("vent_" + str(i))
        if not Path(output_directory).exists():
            Path(output_directory).mkdir(parents=True, exist_ok=True)
        np.save(out_fname, y_sig)
