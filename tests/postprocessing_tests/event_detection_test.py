"""
sanity tests for the postprocessing.event_detection submodule of the resurfemg
library
"""
import unittest
import numpy as np
import scipy
from scipy.integrate import trapezoid

import resurfemg.preprocessing.envelope as evl
import resurfemg.postprocessing.baseline as bl
import resurfemg.postprocessing.features as feat
import resurfemg.postprocessing.event_detection as evt

# Dummy EMG signal
fs_emg = 2048
t_emg = np.array([s_t/fs_emg for s_t in range(10*fs_emg)])
y_sin = np.cos((0.5* t_emg - 0.5)* 2 * np.pi)
y_sin[y_sin < 0] = 0
y_rand = np.random.normal(0, 1, size=len(y_sin))
y_rand_baseline = np.random.normal(0, 1, size=len(y_sin)) / 10
y_t_emg = y_sin * y_rand + y_rand_baseline

y_env_emg = evl.full_rolling_rms(y_t_emg, fs_emg // 5)
y_emg_baseline = bl.moving_baseline(y_env_emg, 5*fs_emg, fs_emg//2)

peaks_env, _ = scipy.signal.find_peaks(y_env_emg, prominence=0.1)
emg_start_idxs, emg_end_idxs, *_ = evt.onoffpeak_baseline_crossing(
             y_env_emg, y_emg_baseline, peaks_env)
etps = feat.time_product(
    signal=y_env_emg,
    fs=fs_emg,
    start_idxs=emg_start_idxs,
    end_idxs=emg_end_idxs)

# Dummy Pocc signal
fs_vent = 100
s_vent = np.array([(s_t) for s_t in range(10*fs_vent)])
t_vent = (s_vent + 1)/fs_vent
rr = 12
t_r = 60/rr
f_r = 1/t_r
y_sin = np.sin((f_r* t_vent)* 2 * np.pi)
y_sin[y_sin > 0] = 0
y_t_p_vent = 5 * y_sin

pocc_peaks_valid, _ = scipy.signal.find_peaks(-y_t_p_vent, prominence=0.1)
pocc_starts = s_vent[((t_vent+t_r/2)%t_r == 0)]
pocc_ends = s_vent[(t_vent%t_r == 0)]

PTP_occs = np.zeros(pocc_peaks_valid.shape)
for _idx, _ in enumerate(pocc_peaks_valid):
    PTP_occs[_idx] = trapezoid(
        -y_t_p_vent[pocc_starts[_idx]:pocc_ends[_idx]],
        dx=1/fs_vent
    )

class TestEventDetection(unittest.TestCase):
    fs = 1000
    t = np.arange(0, 10, 1/fs)
    slow_component = np.sin(2 * np.pi * 0.1 * t)
    fast_component = 0.5 * np.sin(2 * np.pi * 0.5 * t)
    breathing_signal = np.abs(slow_component + fast_component)

    baseline = 0.5 * np.ones((len(breathing_signal), ))
    treshold = 0
    width = fs // 2
    prominence = 0.5 * (np.nanpercentile(breathing_signal - baseline, 75)
                        + np.nanpercentile(breathing_signal - baseline, 50))

    peak_idxs, _ = scipy.signal.find_peaks(
        breathing_signal,
        height=treshold,
        prominence=prominence,
        width=width)

    def test_baseline_crossing_starts(self):
        peak_start_idxs, _, _, _, _ = evt.onoffpeak_baseline_crossing(
             self.breathing_signal, self.baseline, self.peak_idxs)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_start_idxs),
            )

    def test_baseline_crossing_ends(self):
        _, peak_end_idxs, _, _, _ = evt.onoffpeak_baseline_crossing(
             self.breathing_signal, self.baseline, self.peak_idxs)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_end_idxs),
            )

    def test_slope_extrapolate_starts(self):
        peak_start_idxs, _, _, _, _ = evt.onoffpeak_slope_extrapolation(
             self.breathing_signal, self.fs, self.peak_idxs, self.fs//4)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_start_idxs),
            )

    def test_slope_extrapolate_ends(self):
        _, peak_end_idxs, _, _, _ = evt.onoffpeak_slope_extrapolation(
             self.breathing_signal, self.fs, self.peak_idxs, self.fs//4)

        self.assertEqual(
            len(self.peak_idxs),
            len(peak_end_idxs),
            )

    def test_detect_ventilator_breath(self):
        ventilator_breath_idxs = evt.detect_ventilator_breath(
            signal=self.breathing_signal,
            start_idx=1,
            end_idx=10000,
            width_s=1
            )
        self.assertEqual(
            len(ventilator_breath_idxs),
            2,
            )

class TestPoccDetection(unittest.TestCase):
    def test_baseline_crossing_starts(self):
        peak_idxs_detected = evt.find_occluded_breaths(
            p_vent=y_t_p_vent,
            peep=0,
            fs=fs_vent,
        )

        np.testing.assert_array_equal(
            peak_idxs_detected,
            pocc_peaks_valid,
            )

class TestFindLinkedPeaks(unittest.TestCase):
    def test_find_linked_peaks(self):
        t_1 = [10.0, 15.0, 20.0]
        t_2 = [x + 0.2 for x in range(30)]
        linked_peaks = evt.find_linked_peaks(t_1, t_2)
        np.testing.assert_array_equal(
            linked_peaks,
            np.array([10, 15, 20])
        )

class TestDetectEmgBreaths(unittest.TestCase):
    def test_detect_emg_breaths(self):
        detected_peaks = evt.detect_emg_breaths(
            y_env_emg, y_emg_baseline)
        np.testing.assert_array_equal(
            detected_peaks,
            peaks_env
        )

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
