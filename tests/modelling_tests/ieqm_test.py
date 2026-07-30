"""
Sanity tests for the modelling.integrated_equation_of_motion module of the
ReSurfEMG library.
"""
import unittest
import numpy as np
import pandas as pd
from scipy import signal

from resurfemg.modelling import integrated_equation_of_motion as ieqm
from resurfemg.preprocessing.pneumatic import zero_cros_flow
from resurfemg.modelling import tau_estimation as tau_est

class TestTauEstimation(unittest.TestCase):
    def setUp(self):
        self.fs = 100
        self.t = np.arange(-0.01, 22.51, 1/self.fs)
        self.tau = 0.5
        self.tau_est = 0.5
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
        self.zc, _ = zero_cros_flow(self.flow, flow_threshold=0.2)
        (self.tau_df, self.tau_mask, self.df_submask) = tau_est.tau_mask(
            p_aw=self.paw,
            flow=self.flow,
            volume=self.volume,
            peep=self.peep,
            zc=self.zc)
        self.v_rs = self.volume + self.tau_est * self.flow

    def test_v_rs_estimation(self):
        tau_est = 0.5
        v_rs = ieqm.v_rs_estimation(
            volume=self.volume,
            flow=self.flow,
            tau=tau_est,
            expiration_mask=self.df_submask['expiration'].to_numpy(),
            window_s=int(self.fs * 5),
            step_s=int(self.fs / 5),
            set_percentile=33,
            omit_nan=True
        )
        vrs_expected = self.volume + tau_est * self.flow
        np.testing.assert_array_equal(v_rs, vrs_expected)

    def test_compliance_estimation(self):
        ptp_aw = np.ones((10, )) * 10
        etp_mus = np.arange(0, 10)
        nmc_true = 3.0
        c_true = 0.05
        vtp_rs = c_true * (ptp_aw + nmc_true * etp_mus)
        ieqm_df = pd.DataFrame({
            'PTPaw-PTPpeep': ptp_aw,
            'ETPmus': etp_mus,
            'VTPrs': vtp_rs,
        })

        c_est, nmc_est, _ = ieqm.compliance_estimation(
            df_ieqm=ieqm_df,
            t=1.345,
            keys=None,
            verbose=False
        )
        self.assertAlmostEqual(c_est, 1000 * c_true, places=12)
        self.assertAlmostEqual(nmc_est, nmc_true, places=12)

    def test_p_mus_estimation(self):
        c = 50.0
        p_mus_est = ieqm.p_mus_estimation(
            p_aw=self.paw,
            peep=self.peep,
            v_rs=self.v_rs,
            c=c
        )
        p_mus_expected = 1000 * self.v_rs / c - (self.paw - self.peep)
        np.testing.assert_array_equal(p_mus_est, p_mus_expected)
