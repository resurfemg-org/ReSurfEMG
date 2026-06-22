"""sanity tests for the preprocessing.filtering functions."""  # noqa: INP001

import unittest
from pathlib import Path

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.preprocessing import filtering as filt

sample_emg = str(
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath(
        "test_data",
        "emg_data_synth_quiet_breathing.Poly5",
    )
)
synth_pocc_emg = str(
    Path(__file__)
    .resolve()
    .parents[2]
    .joinpath(
        "test_data",
        "emg_data_synth_quiet_breathing.Poly5",
    )
)


class TestFilteringMethods(unittest.TestCase):
    """Test the filtering methods."""

    def test_emg_band_pass_butter(self) -> None:
        """Test the emg_band_pass_butter function."""
        sample_read = Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(
            sample_read.samples, 1, 10, sample_read.sample_rate
        )
        assert len(sample_emg_filtered[0]) == len(sample_read.samples[0])

    def test_emg_band_pass_butter_sample(self) -> None:
        """Test the emg_band_pass_butter function with sample parameters."""
        sample_read = Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_bandpass_butter(
            sample_read.samples, high_pass=1, low_pass=10, fs_emg=2048
        )
        assert len(sample_emg_filtered[0]) == len(sample_read.samples[0])

    def test_emg_lowpass_butter(self) -> None:
        """Test the emg_lowpass_butter function."""
        sample_read = Poly5Reader(sample_emg)
        sample_emg_filtered = filt.emg_lowpass_butter(sample_read.samples, 5.0, 2048)
        assert len(sample_emg_filtered[0]) == len(sample_read.samples[0])

    def test_notch_filter(self) -> None:
        """Test the notch_filter function."""
        sample_read = Poly5Reader(sample_emg)
        sample_emg_filtered = filt.notch_filter(sample_read.samples, 80, 2048, 2)
        assert len(sample_emg_filtered[0]) == len(sample_read.samples[0])


if __name__ == "__main__":
    unittest.main()
