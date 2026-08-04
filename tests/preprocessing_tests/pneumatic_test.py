"""
Sanity tests for the preprocessing.pneumatic module of the ReSurfEMG library.
"""
import unittest
import numpy as np
from scipy import signal

from resurfemg.preprocessing import pneumatic as pneu


class TestZeroCrossingFlow(unittest.TestCase):
    def test_zero_crossing_flow(self):
        square_wave = signal.square(2 * np.pi * 1 * np.linspace(0, 5, 501))
        zc, zc_candidate = pneu.zero_cros_flow(square_wave, flow_threshold=0.5)
        self.assertTrue(len(zc) > 0)
        zc_expected = np.arange(100, 401, 100)
        np.testing.assert_array_equal(zc, zc_expected)
        zc_candidate_expected = np.arange(50, 501, 50)
        np.testing.assert_array_equal(zc_candidate, zc_candidate_expected)

class TestVolumeComputation(unittest.TestCase):
    def setUp(self):
        self.fs = 100
        self.t = np.arange(0, 5, 1/self.fs)
        self.square_wave = signal.square(2 * np.pi * self.t)
        self.zc_idxs, self.zc_candidate = pneu.zero_cros_flow(
            self.square_wave, flow_threshold=0.5)

    def test_volume_computation_last_point(self):
        volume, _, volume_raw, ee_idxs = pneu.volume_computation(
            t=self.t,
            flow=self.square_wave,
            fs=self.fs,
            zc_idxs=self.zc_idxs,
            method="Last point")
        np.testing.assert_array_equal(ee_idxs, self.zc_idxs)
        # End-expiratory is determined by flow zero crossings
        # Volume should be zero at the zero crossings
        v_ee = volume_raw[ee_idxs]
        np.testing.assert_array_equal(v_ee, np.zeros(len(v_ee)))
        # Tinsp = 0.5 s a flow of 1 L/s should give a volume of 0.5 L
        self.assertEqual(max(volume_raw), 0.50)
        # The actual volume should be 0.5 with a exact zero baseline.
        self.assertEqual(max(volume), 0.50)

    def test_volume_computation_last_points(self):
        volume, _, volume_raw, ee_idxs = pneu.volume_computation(
            t=self.t,
            flow=self.square_wave,
            fs=self.fs,
            zc_idxs=self.zc_idxs,
            method="Last points")
        # End-expiratory is determined by flow zero crossings and the 0.1 s
        # before the zero crossing
        ee_idxs_pred = np.array([], dtype=int)
        ee_vol_pred = np.array([], dtype=float)
        for zc_idx in self.zc_idxs:
            zc_begin = max(0, zc_idx - int(0.1 * self.fs))
            zc_array = np.arange(zc_begin - 1, zc_idx)
            ee_idxs_pred = np.concatenate((ee_idxs_pred, zc_array))
            vol_pred = np.arange(11.0, 0.0, -1.0) / 100.0  # 0.11 to 0.00 L
            ee_vol_pred = np.concatenate((ee_vol_pred, vol_pred))
        # The end-expiratory indices are the 0.1 s before the zero crossing,
        np.testing.assert_array_equal(ee_idxs, ee_idxs_pred)
        # The volume at the end-expiratory indices should be decreasing from 
        # 0.11 to 0.00 L in the 0.1 s before the zero crossing.
        np.testing.assert_array_equal(volume_raw[ee_idxs], ee_vol_pred)
        # Tinsp = 0.5 s a flow of 1 L/s should give a volume of 0.5 L
        self.assertEqual(max(volume_raw), 0.50)
        # The actual volume should be slightly less due to baseline calculation
        # in the window before zero flow, rather than volume minimum.
        self.assertEqual(max(volume), 0.49)

    def test_volume_computation_mask(self):
        _flow = np.sin(2 * np.pi * self.t)
        vol_raw_pred = (1 - np.cos(2 * np.pi * self.t)) / (2 * np.pi)
        volume, _, volume_raw, ee_idxs = pneu.volume_computation(
            t=self.t,
            flow=_flow,
            fs=self.fs,
            zc_idxs=self.zc_idxs,
            method="Mask")
        # End-expiratory is determined by flow < 0.00 L/s, flow > -0.01 L/s
        # and volume < 0.1 L. For a sine wave, this is at the zero crossing.
        ee_idxs_pred = np.array(
            [100, 200, 300, 400])
        ee_vol_pred = vol_raw_pred[ee_idxs_pred]
        np.testing.assert_array_equal(ee_idxs, ee_idxs_pred)
        np.testing.assert_array_almost_equal(
            volume_raw[ee_idxs], ee_vol_pred, decimal=15)
        # Volume should be the integral of the flow signal, which is a cosine
        # wave.
        np.testing.assert_array_almost_equal(
            volume_raw, vol_raw_pred, decimal=2)
        # The actual volume should be slightly less due to baseline calculation
        # at zero flow, rather than volume minimum.
        np.testing.assert_array_almost_equal(
            max(volume), 1/np.pi, decimal=3)
