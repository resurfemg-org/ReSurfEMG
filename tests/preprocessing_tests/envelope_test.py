"""sanity tests for the preprocessing.envelope functions."""

import unittest

import numpy as np
from scipy.signal import find_peaks

from resurfemg.preprocessing import envelope as evl


class TestRmsMethods(unittest.TestCase):
    fs_emg = 2048
    t_emg = np.array(range(3 * fs_emg)) / fs_emg
    x_sin = np.sin(t_emg * 2 * np.pi)
    x_sin[x_sin < 0] = 0
    x_rand = np.random.normal(0, 1, size=len(x_sin))
    x_t = x_sin * x_rand
    peak_idxs_source, _ = find_peaks(x_sin, prominence=0.1)

    def test_full_rolling_rms_length(self):
        x_rms = evl.full_rolling_rms(self.x_t, self.fs_emg // 5)
        self.assertEqual(
            (len(self.x_t)),
            len(x_rms),
        )

    def test_full_rolling_rms_time_shift(self):
        x_rms = evl.full_rolling_rms(self.x_t, self.fs_emg // 5)
        peaks_rms, _ = find_peaks(x_rms, prominence=0.1)
        peak_errors = np.abs(
            self.t_emg[peaks_rms] - self.t_emg[self.peak_idxs_source]
        )

        self.assertFalse(np.any(peak_errors > 0.05))


class TestArvMethods(unittest.TestCase):
    fs_emg = 2048
    t_emg = np.array(range(3 * fs_emg)) / fs_emg
    x_sin = np.sin(t_emg * 2 * np.pi)
    x_sin[x_sin < 0] = 0
    x_rand = np.random.normal(0, 1, size=len(x_sin))
    x_t = x_sin * x_rand
    peak_idxs_source, _ = find_peaks(x_sin, prominence=0.1)

    def test_full_rolling_arv_length(self):
        x_arv = evl.full_rolling_arv(self.x_t, self.fs_emg // 5)
        self.assertEqual(
            (len(self.x_t)),
            len(x_arv),
        )

    def test_full_rolling_arv_time_shift(self):
        x_arv = evl.full_rolling_arv(self.x_t, self.fs_emg // 5)
        peaks_arv, _ = find_peaks(x_arv, prominence=0.1)
        peak_errors = np.abs(
            self.t_emg[peaks_arv] - self.t_emg[self.peak_idxs_source]
        )

        self.assertFalse(np.any(peak_errors > 0.05))

    class TestRmsCIMethods(unittest.TestCase):
        fs_emg = 2048
        t_emg = np.array(range(3 * fs_emg)) / fs_emg
        x_sin = np.sin(t_emg * 2 * np.pi)
        x_sin[x_sin < 0] = 0
        x_rand = np.random.normal(0, 1, size=len(x_sin))
        x_t = x_sin * x_rand
        peak_idxs_source, _ = find_peaks(x_sin, prominence=0.1)

        def test_rolling_rms_ci_length(self):
            lower_ci, upper_ci = evl.rolling_rms_ci(self.x_t, self.fs_emg // 5)
            self.assertEqual(len(self.x_t), len(lower_ci))
            self.assertEqual(len(self.x_t), len(upper_ci))

        def test_rolling_rms_ci_values(self):
            lower_ci, upper_ci = evl.rolling_rms_ci(self.x_t, self.fs_emg // 5)
            self.assertTrue(np.all(lower_ci <= upper_ci))

    class TestArvCIMethods(unittest.TestCase):
        fs_emg = 2048
        t_emg = np.array(range(3 * fs_emg)) / fs_emg
        x_sin = np.sin(t_emg * 2 * np.pi)
        x_sin[x_sin < 0] = 0
        x_rand = np.random.normal(0, 1, size=len(x_sin))
        x_t = x_sin * x_rand
        peak_idxs_source, _ = find_peaks(x_sin, prominence=0.1)

        def test_rolling_arv_ci_length(self):
            lower_ci, upper_ci = evl.rolling_arv_ci(self.x_t, self.fs_emg // 5)
            self.assertEqual(len(self.x_t), len(lower_ci))
            self.assertEqual(len(self.x_t), len(upper_ci))

        def test_rolling_arv_ci_values(self):
            lower_ci, upper_ci = evl.rolling_arv_ci(self.x_t, self.fs_emg // 5)
            self.assertTrue(np.all(lower_ci <= upper_ci))


if __name__ == "__main__":
    unittest.main()
