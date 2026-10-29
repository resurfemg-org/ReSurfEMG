"""sanity tests for the pipelines"""

import os
import unittest

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.pipelines import processing as pipe
from resurfemg.preprocessing.ecg_removal import detect_ecg_peaks

synth_pocc_emg = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
    "emg_data_synth_quiet_breathing.Poly5",
)


class TestGatingPipeline(unittest.TestCase):
    data_emg = Poly5Reader(synth_pocc_emg)
    y_emg = data_emg.samples[: data_emg.num_samples]
    fs_emg = data_emg.sample_rate
    ecg_peaks = detect_ecg_peaks(
        ecg_raw=y_emg[0],
        fs=fs_emg,
        bp_filter=True,
    )

    def test_gating_pipeline(self):
        emg_di_gated = pipe.ecg_removal_gating(
            emg_raw=self.y_emg[1, :],
            ecg_peaks_idxs=self.ecg_peaks,
            gate_width_samples=self.fs_emg // 10,
            ecg_shift=10,
        )
        self.assertEqual(len(emg_di_gated), self.y_emg.shape[1])


class TestQuickLookPipeline(unittest.TestCase):
    data_emg = Poly5Reader(synth_pocc_emg)
    y_emg = data_emg.samples[: data_emg.num_samples]
    fs_emg = data_emg.sample_rate

    def test_gating_pipeline(self):
        emg_filt, emg_env = pipe.quick_look(
            emg_raw=self.y_emg[1, :],
            fs_emg=self.fs_emg,
            plot_raw=False,
            plot_clean=False,
            plot_env=False,
            plot_power_spectrum=False,
        )
        self.assertEqual(len(emg_filt), self.y_emg.shape[1])
        self.assertEqual(len(emg_env), self.y_emg.shape[1])


if __name__ == "__main__":
    unittest.main()
