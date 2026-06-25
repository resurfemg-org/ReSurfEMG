"""Sanity tests for the file_discovery submodule of the resurfemg library."""  # noqa: INP001

import platform
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from resurfemg.data_connector.converter_functions import load_file

base_path = Path(__file__).resolve().parents[2] / "test_data"


class TestLoadFile(unittest.TestCase):
    """Test the load_file function with various file formats."""

    file_path = base_path / "emg_data_synth_quiet_breathing"

    def test_load_file_poly5(self):
        """Test loading a .Poly5 file."""
        file_name = str(self.file_path.with_suffix(".Poly5"))
        np_data, df_data, metadata = load_file(file_name)
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata["file_extension"] == "Poly5"

    def test_load_file_adidat(self):
        """Test loading a .adidat file."""
        file_name = str(self.file_path.with_suffix(".adidat"))
        if platform.system() == "Windows":
            np_data, df_data, metadata = load_file(file_name)
            assert isinstance(np_data, np.ndarray)
            assert isinstance(df_data, pd.DataFrame)
            assert isinstance(metadata, dict)
            assert metadata["file_extension"] == "adidat"
        else:
            with pytest.raises(UserWarning):
                load_file(file_name)

    def test_load_file_csv(self):
        """Test loading a .csv file."""
        file_name = str(self.file_path.with_suffix(".csv"))
        np_data, df_data, metadata = load_file(file_name)
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata["file_extension"] == "csv"

    def test_load_file_mat(self):
        """Test loading a .mat file."""
        file_name = str(self.file_path.with_suffix(".mat"))
        np_data, df_data, metadata = load_file(file_name, key_name="mat5_data")
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata["file_extension"] == "mat"

    # def test_load_file_npy(self):
    #     """Test loading a .npy file."""
    #     file_name = str(self.file_path.with_suffix(".npy"))  # noqa: ERA001
    #     np_data, df_data, metadata = load_file(file_name)  # noqa: ERA001
    #     assert isinstance(np_data, np.ndarray)  # noqa: ERA001
    #     assert isinstance(df_data, pd.DataFrame)  # noqa: ERA001
    #     assert isinstance(metadata, dict)  # noqa: ERA001
    #     assert metadata["file_extension"] == "npy"  # noqa: ERA001
