"""sanity tests for the preprocessing.ecg_removal functions"""

import os
import unittest

import numpy as np
import scipy

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing import ecg_removal as ecg_rm
from resurfemg.preprocessing import filtering as filt

sample_emg = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
    "emg_data_synth_quiet_breathing.Poly5",
)
synth_pocc_emg = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
    "emg_data_synth_quiet_breathing.Poly5",
)


class TestEcgPeakDetection(unittest.TestCase):
    data_emg = Poly5Reader(synth_pocc_emg)
    y_emg = data_emg.samples[: data_emg.num_samples]
    fs_emg = data_emg.sample_rate

    def test_detect_ecg_peaks(self):
        ecg_peaks = ecg_rm.detect_ecg_peaks(
            ecg_raw=self.y_emg[0],
            fs=self.fs_emg,
            bp_filter=True,
        )
        self.assertEqual(len(ecg_peaks), 449)


class TestGating(unittest.TestCase):
    sample_read = Poly5Reader(sample_emg)
    sample_emg_filtered = -filt.emg_bandpass_butter(
        sample_read.samples, 1, 500, 2048
    )
    sample_emg_filtered = sample_emg_filtered[: 30 * 2048]
    ecg_peaks, _ = scipy.signal.find_peaks(sample_emg_filtered[0, :])

    def test_gating_method_0(self):
        ecg_gated_0 = ecg_rm.gating(
            self.sample_emg_filtered[0, :],
            self.ecg_peaks,
            gate_width=205,
            method=0,
        )

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_0),
        )

    def test_gating_method_1(self):
        ecg_gated_1 = ecg_rm.gating(
            self.sample_emg_filtered[0, :],
            self.ecg_peaks,
            gate_width=205,
            method=1,
        )

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_1),
        )

    def test_gating_method_2(self):
        ecg_gated_2 = ecg_rm.gating(
            self.sample_emg_filtered[0, :],
            self.ecg_peaks,
            gate_width=205,
            method=2,
        )

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_gated_2),
        )

    def test_gating_method_2_no_prior_segment(self):
        ecg_gated_2 = ecg_rm.gating(
            self.sample_emg_filtered[0, :], [100], gate_width=205, method=2
        )

        self.assertFalse(np.isnan(np.sum(ecg_gated_2)))

    def test_gating_method_3(self):
        height_threshold = np.max(self.sample_emg_filtered) / 2
        ecg_peaks, _ = scipy.signal.find_peaks(
            self.sample_emg_filtered[0, : 10 * 2048 - 1],
            height=height_threshold,
        )

        ecg_gated_3 = ecg_rm.gating(
            self.sample_emg_filtered[0, : 10 * 2048],
            ecg_peaks,
            gate_width=205,
            method=3,
        )

        self.assertEqual(
            (len(self.sample_emg_filtered[0, : 10 * 2048])),
            len(ecg_gated_3),
        )


class TestWaveletDenoising(unittest.TestCase):
    sample_read = Poly5Reader(sample_emg)
    fs = sample_read.sample_rate
    sample_emg_filtered = -filt.emg_bandpass_butter(
        sample_read.samples, 1, 500, fs
    )
    sample_emg_filtered = sample_emg_filtered[: 30 * 2048]
    ecg_peaks, _ = scipy.signal.find_peaks(sample_emg_filtered[0, :])

    def wavelet_denoising(self):
        ecg_denoised = ecg_rm.wavelet_denoising(
            emg_raw=self.sample_emg_filtered[0, :],
            ecg_peak_idxs=self.ecg_peaks,
            fs=self.fs,
            hard_thresholding=True,
            n=4,
            wavelet_type="db2",
            fixed_threshold=4.5,
        )

        self.assertEqual(
            (len(self.sample_emg_filtered[0])),
            len(ecg_denoised),
        )
