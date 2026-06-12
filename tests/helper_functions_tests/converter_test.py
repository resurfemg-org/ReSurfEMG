"""Sanity tests for the converter functions."""  # noqa: INP001

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

# converter_functions
from resurfemg.data_connector import converter_functions
from resurfemg.data_connector.config import hash_it_up_right_all
from resurfemg.data_connector.converter_functions import poly5unpad

# tmsisdk_lite
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader

sample_path = Path(__file__).resolve().parents[2] / "test_data"
sample_emg_poly5 = str(sample_path / "emg_data_synth_quiet_breathing.Poly5")
sample_emg_mat = str(sample_path / "emg_data_synth_quiet_breathing.mat")
sample_emg_csv = str(sample_path / "emg_data_synth_quiet_breathing.csv")


class TestLoadData(unittest.TestCase):
    """Test the load_data function."""

    def test_load_poly5(self):
        """Test loading a Poly5 file."""
        np_array, *_ = converter_functions.load_file(file_path=sample_emg_poly5)
        assert np_array.shape[0] == 2

    def test_load_mat(self):
        """Test loading a MAT file."""
        np_array, *_ = converter_functions.load_file(
            file_path=sample_emg_mat,
            key_name="mat5_data",
        )
        assert np_array.shape[0] == 2

    def test_load_csv(self):
        """Test loading a CSV file."""
        np_array, *_ = converter_functions.load_file(file_path=sample_emg_csv)
        assert np_array.shape[0] == 2


class TestConverterMethods(unittest.TestCase):
    """Test the converter functions."""

    def test_poly5unpad(self):
        """Test the poly5unpad function."""
        reading = Poly5Reader(sample_emg_poly5)
        unpadded = poly5unpad(sample_emg_poly5)
        unpadded_line = unpadded[0]
        assert len(unpadded_line) == reading.num_samples

    def test_Poly5Reader(self):  # noqa: N802
        """Test the Poly5Reader."""
        reading = Poly5Reader(sample_emg_poly5)
        assert reading.num_channels == 2


class TestHashMethods(unittest.TestCase):
    """Test the hash methods."""

    def test_hash_it_up_right_all(self):
        """Test the hash_it_up_right_all function."""
        tempfile1 = "sample_emg_poly5_t.Poly5"
        tempfile2 = "sample_emg_poly5_t.Poly5"
        with TemporaryDirectory() as td:
            with Path(td).joinpath(tempfile1).open("w") as tf:
                tf.write("string")
            with Path(td).joinpath(tempfile2).open("w") as tf:
                tf.write("string")
            assert hash_it_up_right_all(td, ".Poly5").equals(hash_it_up_right_all(td, ".Poly5"))
