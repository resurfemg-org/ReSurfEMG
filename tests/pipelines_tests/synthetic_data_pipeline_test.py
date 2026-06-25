"""Sanity tests for the synthetic data pipelines."""  # noqa: INP001

import unittest
from unittest import TestCase

from resurfemg.pipelines import synthetic_data as synth


class TestEmgPipeline(TestCase):
    """Test the EMG pipeline."""

    def test_simulate_raw_emg(self):
        """Test the simulate_raw_emg function."""
        sim_parameters = {
            "t_end": 30,
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

        y_emg = synth.simulate_raw_emg(**sim_parameters)
        assert len(y_emg) == 30 * 2048


class TestVentilatorPipeline(TestCase):
    """Test the ventilator pipeline."""

    def test_simulate_ventilator_data(self):
        """Test the simulate_ventilator_data function."""
        # Simulate ventilator data
        y_vent, p_mus = synth.simulate_ventilator_data(
            t_end=30,
            fs_vent=100,
            p_mus_amp=5,
        )
        assert y_vent.shape[0] == 3
        assert y_vent.shape[1] == 30 * 100
        assert p_mus.shape[0] == 30 * 100


if __name__ == "__main__":
    unittest.main()
