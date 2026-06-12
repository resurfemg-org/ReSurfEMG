"""Sanity tests for the postprocessing.baseline submodule of the resurfemg library."""  # noqa: INP001

import unittest
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import trapezoid

import resurfemg.postprocessing.baseline as bl
import resurfemg.postprocessing.event_detection as evt
import resurfemg.postprocessing.features as feat
from resurfemg.preprocessing import envelope as evl

sample_emg = Path(__name__).resolve().parents[2] / "test_data" / "emg_data_synth_quiet_breathing.Poly5"

rng = np.random.default_rng()

# Dummy EMG signal
fs_emg = 2048
t_emg = np.array([s_t / fs_emg for s_t in range(10 * fs_emg)])
y_sin = np.cos((0.5 * t_emg - 0.5) * 2 * np.pi)
y_sin[y_sin < 0] = 0
y_rand = rng.normal(0, 1, size=len(y_sin))
y_rand_baseline = rng.normal(0, 1, size=len(y_sin)) / 10
y_t_emg = y_sin * y_rand + y_rand_baseline

y_env_emg = evl.full_rolling_rms(y_t_emg, fs_emg // 5)
y_emg_baseline = bl.moving_baseline(y_env_emg, 5 * fs_emg, fs_emg // 2)

peaks_env, _ = scipy.signal.find_peaks(y_env_emg, prominence=0.1)
emg_start_idxs, emg_end_idxs, *_ = evt.onoffpeak_baseline_crossing(y_env_emg, y_emg_baseline, peaks_env)
etps = feat.time_product(signal=y_env_emg, fs=fs_emg, start_idxs=emg_start_idxs, end_idxs=emg_end_idxs)

# Dummy Pocc signal
fs_vent = 100
s_vent = np.array(list(range(10 * fs_vent)))
t_vent = (s_vent + 1) / fs_vent
rr = 12
t_r = 60 / rr
f_r = 1 / t_r
y_sin = np.sin((f_r * t_vent) * 2 * np.pi)
y_sin[y_sin > 0] = 0
y_t_p_vent = 5 * y_sin

pocc_peaks_valid, _ = scipy.signal.find_peaks(-y_t_p_vent, prominence=0.1)
pocc_starts = s_vent[((t_vent + t_r / 2) % t_r == 0)]
pocc_ends = s_vent[(t_vent % t_r == 0)]

PTP_occs = np.zeros(pocc_peaks_valid.shape)
for _idx, _ in enumerate(pocc_peaks_valid):
    PTP_occs[_idx] = trapezoid(-y_t_p_vent[pocc_starts[_idx] : pocc_ends[_idx]], dx=1 / fs_vent)


class TestBaseline(unittest.TestCase):
    """Sanity tests for the postprocessing.baseline submodule of the resurfemg library."""

    fs = 1000
    t = np.arange(0, 10, 1 / fs)
    slow_component = np.sin(2 * np.pi * 0.1 * t)
    fast_component = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    breathing_signal = np.abs(slow_component + fast_component)

    def test_movingbaseline(self):
        """Test that the moving baseline returns an array of the same length as the input signal."""
        sinusbase = bl.moving_baseline(self.breathing_signal, self.fs, self.fs // 5, 33)
        assert len(self.breathing_signal) == len(sinusbase)

    def test_slopesum(self):
        """Test that the slope sum baseline returns an array of the same length as the input signal."""
        sinusbase, _, _, _ = bl.slopesum_baseline(
            self.breathing_signal,
            self.fs,
            self.fs // 5,
            self.fs,
            set_percentile=33,
            augm_percentile=25,
        )

        assert len(self.breathing_signal) == len(sinusbase)
