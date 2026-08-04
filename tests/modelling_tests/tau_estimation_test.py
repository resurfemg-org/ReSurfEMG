"""
Sanity tests for the modelling.tau_estimation module of the ReSurfEMG library.
"""
import unittest
import numpy as np
from scipy import signal

from resurfemg.modelling import tau_estimation as tau_est
from resurfemg.preprocessing.pneumatic import zero_cros_flow

class TestTauEstimation(unittest.TestCase):
    def setUp(self):
        self.fs = 100
        self.t = np.arange(-0.01, 22.51, 1/self.fs)
        self.tau = 0.5
        self.flow_sign = signal.square(2 * np.pi * self.t / 5)
        self.flow_local_t = self.t % 2.5
        self.flow = np.zeros(self.t.shape)
        self.flow[self.flow_sign >= 0] = np.exp(
            -self.flow_local_t[self.flow_sign >= 0] / self.tau)
        self.flow[self.flow_sign < 0] = - np.exp(
            -self.flow_local_t[self.flow_sign < 0] / self.tau)
        self.volume = np.zeros(self.t.shape)
        self.volume[self.flow_sign >= 0] = self.tau * (1 - np.exp(
            -self.flow_local_t[self.flow_sign >= 0] / self.tau))
        self.volume[self.flow_sign < 0] = self.tau * np.exp(
            -self.flow_local_t[self.flow_sign < 0] / self.tau)
        # self.volume = self.tau * self.flow
        self.peep = 5
        self.paw = self.peep + 5 / 2 * (
            1 + signal.square(2 * np.pi * self.t / 5))

    def test_tau_mask(self):
        zc_idxs, _ = zero_cros_flow(self.flow, flow_threshold=0.2)
        (_, _, df_submask) = tau_est.tau_mask(
            p_aw=self.paw,
            flow=self.flow,
            volume=self.volume,
            peep=self.peep,
            fs=self.fs,
            zc_idxs=zc_idxs)
        # Paw - PEEP mask
        mask_paw_peep_pred = self.paw == self.peep
        np.testing.assert_array_equal(
            df_submask['paw_peep'].to_numpy(),
            mask_paw_peep_pred
            )
        # Expiration mask
        mask_exp_pred = self.flow < 0
        np.testing.assert_array_equal(
            df_submask['expiration'].to_numpy(),
            mask_exp_pred
            )
        # Quality mask
        self.assertTrue(np.all(df_submask.loc[1:2000, 'quality']))

    def test_tau_estimation(self):
        df_tau, mask, _ = tau_est.tau_mask(
            p_aw=self.paw,
            flow=self.flow,
            volume=self.volume,
            peep=self.peep,
            fs=self.fs,
            flow_threshold=0.2
            )
        tau, _ = tau_est.tau_switch_smf(
            df_tau[mask],
            theta_act_exp=4.685,
            verbose=False)
        self.assertAlmostEqual(tau, self.tau, places=10)
