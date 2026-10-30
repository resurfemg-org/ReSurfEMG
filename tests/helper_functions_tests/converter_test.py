"""sanity tests for the converter functions"""

import os
import unittest
from tempfile import TemporaryDirectory

# converter_functions
from resurfemg.data_connector import converter_functions
from resurfemg.data_connector.config import hash_it_up_right_all
from resurfemg.data_connector.converter_functions import poly5unpad

# tmsisdk_lite
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader

sample_path = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
)
sample_emg_poly5 = os.path.join(
    sample_path,
    "emg_data_synth_quiet_breathing.Poly5",
)
sample_emg_mat = os.path.join(
    sample_path,
    "emg_data_synth_quiet_breathing.mat",
)
sample_emg_csv = os.path.join(
    sample_path,
    "emg_data_synth_quiet_breathing.csv",
)


class TestLoadData(unittest.TestCase):
    def test_load_poly5(self):
        np_array, *_ = converter_functions.load_file(
            file_path=sample_emg_poly5
        )
        self.assertEqual(np_array.shape[0], 2)

    def test_load_mat(self):
        np_array, *_ = converter_functions.load_file(
            file_path=sample_emg_mat,
            key_name="mat5_data",
        )
        self.assertEqual(np_array.shape[0], 2)

    def test_load_csv(self):
        np_array, *_ = converter_functions.load_file(file_path=sample_emg_csv)
        self.assertEqual(np_array.shape[0], 2)


class TestConverterMethods(unittest.TestCase):
    def test_poly5unpad(self):
        reading = Poly5Reader(sample_emg_poly5)
        unpadded = poly5unpad(sample_emg_poly5)
        unpadded_line = unpadded[0]
        self.assertEqual(len(unpadded_line), reading.num_samples)

    def Poly5Reader(self):
        reading = Poly5Reader(sample_emg_poly5)
        self.assertEqual(reading.num_channels, 3)


class TestHashMethods(unittest.TestCase):
    def test_hash_it_up_right_all(self):
        tempfile1 = "sample_emg_poly5_t.Poly5"
        tempfile2 = "sample_emg_poly5_t.Poly5"
        with TemporaryDirectory() as td:
            with open(os.path.join(td, tempfile1), "w") as tf:
                tf.write("string")
            with open(os.path.join(td, tempfile2), "w") as tf:
                tf.write("string")
            self.assertTrue(
                hash_it_up_right_all(td, ".Poly5").equals(
                    hash_it_up_right_all(td, ".Poly5")
                )
            )
