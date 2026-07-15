"""Sanity tests for the visualization functions."""  # noqa: INP001

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.helper_functions.visualization import show_periodogram, show_power_spectrum, show_psd_welch
from resurfemg.preprocessing.filtering import emg_bandpass_butter

sample_emg = str(Path(__file__).resolve().parents[2] / "test_data" / "emg_data_synth_quiet_breathing.Poly5")


class TestVisualizationMethods(unittest.TestCase):
    """Test the visualization functions."""

    def setUp(self):
        """Set up the test case."""
        # Common setup for all tests
        sample_read = Poly5Reader(sample_emg)
        self.sample_emg_filtered = -emg_bandpass_butter(sample_read.samples, 1, 500, sample_read.sample_rate)
        self.sample_emg_filtered = self.sample_emg_filtered[: 30 * 2048]

    @patch("matplotlib.pyplot.show")
    def test_show_power_spectrum(self, mock_show: MagicMock):
        """Test the show_power_spectrum function."""
        f, pxx_den = show_power_spectrum(self.sample_emg_filtered[0, :], 2048, 1024)
        assert len(f) == len(self.sample_emg_filtered[0, :])
        assert len(pxx_den) == len(self.sample_emg_filtered[0, :])
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_show_psd_welch(self, mock_show: MagicMock):
        """Test the show_psd_welch function."""
        f, pxx_den = show_psd_welch(self.sample_emg_filtered[0, :], 2048, 256, axis_spec=0)
        expected_length = 256 // 2 + 1
        assert len(f) == expected_length
        assert len(pxx_den) == expected_length
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_show_periodogram(self, mock_show: MagicMock):
        """Test the show_periodogram function."""
        f, pxx_den = show_periodogram(self.sample_emg_filtered[0, :], 2048, 0)
        expected_length = len(self.sample_emg_filtered[0, :]) // 2 + 1
        assert len(f) == expected_length
        assert len(pxx_den) == expected_length
        mock_show.assert_called_once()
