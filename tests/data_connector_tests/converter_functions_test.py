"""sanity tests for the file_discovery submodule of the resurfemg library"""

import os
import unittest
import platform
import numpy as np
import pandas as pd

from resurfemg.data_connector.converter_functions import (load_file)

base_path = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(
        __file__)))),
    'test_data',
)
class TestLoadFile(unittest.TestCase):
    file_path = os.path.join(base_path, 'emg_data_synth_quiet_breathing')
    def test_load_file_poly5(self):
        file_name = self.file_path + '.poly5'
        np_data, df_data, metadata = load_file(file_name)
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata['file_extension'] == 'poly5'

    def test_load_file_adidat(self):
        file_name = self.file_path + '.adidat'
        if platform.system() == 'Windows':
            np_data, df_data, metadata = load_file(file_name)
            assert isinstance(np_data, np.ndarray)
            assert isinstance(df_data, pd.DataFrame)
            assert isinstance(metadata, dict)
            assert metadata['file_extension'] == 'adidat'
        else:
            with self.assertRaises(UserWarning):
                load_file(file_name)

    def test_load_file_csv(self):
        file_name = self.file_path + '.csv'
        np_data, df_data, metadata = load_file(file_name)
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata['file_extension'] == 'csv'

    def test_load_file_mat(self):
        file_name = self.file_path + '.mat'
        np_data, df_data, metadata = load_file(file_name, key_name='mat5_data')
        assert isinstance(np_data, np.ndarray)
        assert isinstance(df_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert metadata['file_extension'] == 'mat'

    # def test_load_file_npy(self):
    #     file_name = self.file_path + '.npy'
    #     np_data, df_data, metadata = load_file(file_name)
    #     assert isinstance(np_data, np.ndarray)
    #     assert isinstance(df_data, pd.DataFrame)
    #     assert isinstance(metadata, dict)
    #     assert metadata['file_extension'] == 'npy'
