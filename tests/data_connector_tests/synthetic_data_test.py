"""Sanity tests for the synthetic data functions."""

from typing import ClassVar
from unittest import TestCase

import numpy as np

import resurfemg.data_connector.synthetic_data as synth


class TestSyntheticEmgData(TestCase):
    """Test the synthetic EMG data generation functions."""

    t_end = 30
    fs_emg = 2048
    rr = 22
    emg_amp = 5
    sim_parameters: ClassVar[dict] = {
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

    def test_respiratory_pattern_generator(self):
        """Test the respiratory pattern generator."""
        assert self.respiratory_pattern.shape[0] == self.fs_emg * self.t_end

    def test_simulate_muscle_dynamics(self):
        """Test the muscle dynamics simulation."""
        assert self.muscle_activation.shape[0] == self.fs_emg * self.t_end

    def test_simulate_emg(self):
        """Test the EMG simulation."""
        assert self.emg_sim.shape[0] == self.fs_emg * self.t_end


class TestSyntheticVentilatorData(TestCase):
    """Test the synthetic ventilator data generation functions."""

    t_end = 30
    fs_vent = 100
    rr = 22
    p_mus_amp = 5
    dp = 5
    sim_parameters: ClassVar[dict] = {
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
        t_occ_bool[
            int((t_occ - 1) * fs_vent) : int((t_occ + 1 / rr * 60) * fs_vent)
        ] = True
    lung_mechanics: ClassVar[dict] = {
        "c": 0.050,
        "r": 5,
    }
    vent_settings: ClassVar[dict] = {
        "dp": dp,
        "peep": 5,
        "flow_cycle": 0.25,  # Fraction F_max
        "flow_trigger": 2,  # L/min
        "tau_dp_up": 10,
        "tau_dp_down": 5,
    }

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

    def test_respiratory_pattern_generator(self):
        """Test the respiratory pattern generator."""
        assert self.respiratory_pattern.shape[0] == self.fs_vent * self.t_end

    def test_simulate_muscle_dynamics(self):
        """Test the muscle dynamics simulation."""
        assert self.p_mus.shape[0] == self.fs_vent * self.t_end

    def test_simulate_ventilator_data(self):
        """Test the ventilator data simulation."""
        assert self.y_vent.shape[0] == 3
        assert self.y_vent.shape[1] == self.fs_vent * self.t_end
