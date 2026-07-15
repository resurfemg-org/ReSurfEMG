"""Sanity tests for the ipy_widgets submodule of the resurfemg library."""  # noqa: INP001

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from scipy import io as sio

from resurfemg.data_connector import file_discovery
from resurfemg.data_connector.data_classes import EmgDataGroup, VentilatorDataGroup
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.pipelines import ipy_widgets

base_path = Path(__file__).resolve().parents[2] / "test_data"
emg_poly5 = base_path / "emg_data_synth_pocc.Poly5"


class TestIpyWidgets(unittest.TestCase):
    """Sanity tests for the ipy_widgets submodule of the resurfemg library."""

    def test_file_select(self):
        """Test the file_select function."""
        files = file_discovery.find_files(
            base_path=str(base_path), file_name_regex="*", extension_regex="Poly5", folder_levels=None, verbose=False
        )
        files.sort_values(by="files", inplace=True)
        button_list = ipy_widgets.file_select(
            files=files, folder_levels=["files"], default_value_select=[files["files"].values[0]]
        )
        assert button_list[0].value == files["files"].values[0]


class TestCheckBoxTree(unittest.TestCase):
    """Tests for the CheckBoxTree anywidget (Python trait side only)."""

    def test_default_initialization(self):
        """CheckBoxTree initialises with empty traits."""
        cbt = ipy_widgets.CheckBoxTree()
        self.assertEqual(cbt.tree_data, {})
        self.assertEqual(cbt.checked_files, [])
        self.assertEqual(cbt.file_types, {})

    def test_set_tree_data_on_init(self):
        """tree_data provided at construction is stored correctly."""
        tree_data = {"P_001": {"emg.Poly5": "EMG", "vent.Poly5": "Ventilator"}}
        cbt = ipy_widgets.CheckBoxTree(tree_data=tree_data)
        self.assertEqual(cbt.tree_data, tree_data)

    def test_update_checked_files(self):
        """checked_files can be updated after construction."""
        cbt = ipy_widgets.CheckBoxTree()
        cbt.checked_files = ["P_001/emg.Poly5"]
        self.assertEqual(cbt.checked_files, ["P_001/emg.Poly5"])

    def test_update_file_types(self):
        """file_types can be updated after construction."""
        cbt = ipy_widgets.CheckBoxTree()
        cbt.file_types = {"P_001/emg.Poly5": "EMG"}
        self.assertEqual(cbt.file_types, {"P_001/emg.Poly5": "EMG"})


class TestDatasetSelectorMethods(unittest.TestCase):
    """Tests for DatasetSelector logic methods (display is mocked)."""

    @classmethod
    def setUpClass(cls):
        with (
            patch("resurfemg.pipelines.ipy_widgets.display"),
            patch("resurfemg.pipelines.ipy_widgets.find_files") as mock_ff,
        ):
            mock_ff.return_value = pd.DataFrame({"files": ["P_001/emg_recording.Poly5", "P_001/vent_recording.Poly5"]})
            cls.selector = ipy_widgets.DatasetSelector(root_directory=Path("/tmp/test"))

    def test_guess_data_type_emg(self):
        """Files with 'emg' in the name are classified as EMG."""
        self.assertEqual(self.selector._guess_data_type("emg_recording.Poly5"), "EMG")

    def test_guess_data_type_ventilator(self):
        """Files with 'vent' or 'flow' in the name are classified as Ventilator."""
        self.assertEqual(self.selector._guess_data_type("vent_flow.Poly5"), "Ventilator")
        self.assertEqual(self.selector._guess_data_type("flow_data.Poly5"), "Ventilator")

    def test_guess_data_type_other(self):
        """Files with no recognisable keyword are classified as Other."""
        self.assertEqual(self.selector._guess_data_type("unknown_recording.Poly5"), "Other")

    def test_complete_regex_matches_poly5(self):
        """Completed regex matches a patient-prefixed Poly5 path."""
        import re

        pattern = re.compile(self.selector.patient_regex)
        self.assertIsNotNone(pattern.search("P_001/emg_file.Poly5"))

    def test_complete_regex_matches_adicht(self):
        """Completed regex matches a patient-prefixed .adicht path."""
        import re

        pattern = re.compile(self.selector.patient_regex)
        self.assertIsNotNone(pattern.search("P001/file.adicht"))

    def test_complete_regex_matches_mat(self):
        """Completed regex matches a patient-prefixed .mat path."""
        import re

        pattern = re.compile(self.selector.patient_regex)
        self.assertIsNotNone(pattern.search("p_42/data.mat"))

    def test_complete_regex_rejects_non_patient_dir(self):
        """Completed regex does not match paths without a patient prefix."""
        import re

        pattern = re.compile(self.selector.patient_regex)
        self.assertIsNone(pattern.search("data/file.Poly5"))

    def test_build_tree_data_flat_files(self):
        """Flat node dict with empty lists is converted to {fname: dtype}."""
        node = {"emg_file.Poly5": [], "vent_file.Poly5": []}
        result = self.selector._build_tree_data(node)
        self.assertEqual(result["emg_file.Poly5"], "EMG")
        self.assertEqual(result["vent_file.Poly5"], "Ventilator")

    def test_build_tree_data_nested_dict(self):
        """Nested dict structure is recursed correctly."""
        node = {"session1": {"emg_file.Poly5": [], "vent_file.Poly5": []}}
        result = self.selector._build_tree_data(node)
        self.assertIn("session1", result)
        self.assertEqual(result["session1"]["emg_file.Poly5"], "EMG")

    def test_build_tree_data_file_list(self):
        """Non-empty list values are expanded to per-file type dicts."""
        node = {"session1": ["emg_file.Poly5", "vent_file.Poly5"]}
        result = self.selector._build_tree_data(node)
        self.assertIn("session1", result)
        self.assertEqual(result["session1"]["emg_file.Poly5"], "EMG")
        self.assertEqual(result["session1"]["vent_file.Poly5"], "Ventilator")

    def test_get_nonmat_data_emg(self):
        """Poly5 data loaded as EMG yields an EmgDataGroup."""
        reader = Poly5Reader(str(emg_poly5))
        result = self.selector._get_nonmat_data(reader, "EMG")
        self.assertIsInstance(result, EmgDataGroup)

    def test_get_nonmat_data_ventilator(self):
        """Poly5 data loaded as Ventilator yields a VentilatorDataGroup."""
        reader = Poly5Reader(str(emg_poly5))
        result = self.selector._get_nonmat_data(reader, "Ventilator")
        self.assertIsInstance(result, VentilatorDataGroup)

    def test_get_nonmat_data_missing_attributes_raises(self):
        """Objects missing 'samples'/'sample_rate' raise ValueError."""

        class BadReader:
            pass

        with self.assertRaises(ValueError):
            self.selector._get_nonmat_data(BadReader(), "EMG")


class TestCustomizeMatlabImport(unittest.TestCase):
    """Tests for CustomizeMatlabImport (display is mocked)."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp()
        cls.mat_file = Path(cls.temp_dir) / "test_data.mat"
        rng = np.random.default_rng(0)
        mat_content = {
            "y_emg": rng.standard_normal((100, 1)),  # single-channel time series
            "fs_emg": np.array([[2000.0]]),  # scalar parameter
        }
        sio.savemat(str(cls.mat_file), mat_content)
        with patch("resurfemg.pipelines.ipy_widgets.display"):
            cls.importer = ipy_widgets.CustomizeMatlabImport(cls.mat_file)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir)

    def _fresh_importer(self):
        """Return a fresh importer instance from the shared mat file."""
        with patch("resurfemg.pipelines.ipy_widgets.display"):
            return ipy_widgets.CustomizeMatlabImport(self.mat_file)

    def test_keys_loaded(self):
        """Both mat keys are discovered."""
        self.assertIn("y_emg", self.importer.keys)
        self.assertIn("fs_emg", self.importer.keys)

    def test_map_keys_y_emg_type_time_series(self):
        """Multi-sample numeric array is classified as time_series."""
        self.assertEqual(self.importer.key_mapping["y_emg"]["Type"], "time_series")

    def test_map_keys_fs_emg_type_params(self):
        """Scalar numeric array (1x1) is classified as params."""
        self.assertEqual(self.importer.key_mapping["fs_emg"]["Type"], "params")

    def test_map_keys_emg_group(self):
        """Both keys with 'emg' in the name are assigned to the EMG group."""
        self.assertEqual(self.importer.key_mapping["y_emg"]["Group"], "EMG")
        self.assertEqual(self.importer.key_mapping["fs_emg"]["Group"], "EMG")

    def test_map_keys_y_emg_contains_none(self):
        """y_emg has no matching Contains pattern (falls back to y_raw in picker)."""
        self.assertIsNone(self.importer.key_mapping["y_emg"]["Contains"])

    def test_map_keys_fs_emg_contains_fs(self):
        """fs_emg Contains field is recognised as 'fs'."""
        self.assertEqual(self.importer.key_mapping["fs_emg"]["Contains"], "fs")

    def test_get_emg_data_none_before_import(self):
        """get_emg_data returns None before import_data is called."""
        importer = self._fresh_importer()
        self.assertIsNone(importer.get_emg_data())

    def test_get_ventilator_data_none_before_import(self):
        """get_ventilator_data returns None before import_data is called."""
        importer = self._fresh_importer()
        self.assertIsNone(importer.get_ventilator_data())

    def test_get_other_data_none_before_import(self):
        """get_other_data returns None before import_data is called."""
        importer = self._fresh_importer()
        self.assertIsNone(importer.get_other_data())

    def test_get_groups_all_none_before_import(self):
        """get_groups returns a 3-tuple of Nones before import_data is called."""
        importer = self._fresh_importer()
        groups = importer.get_groups()
        self.assertIsNone(groups[0])
        self.assertIsNone(groups[1])
        self.assertIsNone(groups[2])

    def test_import_data_creates_emg_data_group(self):
        """After import_data, get_emg_data returns an EmgDataGroup."""
        importer = self._fresh_importer()
        importer.import_data(MagicMock())
        result = importer.get_emg_data()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, EmgDataGroup)

    def test_import_data_emg_shape(self):
        """The imported EMG data has the expected number of samples."""
        importer = self._fresh_importer()
        importer.import_data(MagicMock())
        emg = importer.get_emg_data()
        self.assertEqual(emg[0].param["n_samp"], 100)
