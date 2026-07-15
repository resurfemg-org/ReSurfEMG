"""Sanity tests for the file_discovery submodule of the resurfemg library."""

import unittest
from pathlib import Path

from resurfemg.data_connector import file_discovery

base_path = Path(__file__).resolve().parents[2] / "test_data"


class TestFileDiscovery(unittest.TestCase):
    """Test the file discovery functions."""

    def test_find_files(self):
        """Test the find_files function."""
        found_files = file_discovery.find_files(
            base_path=str(base_path),
            file_name_regex="*",
            extension_regex="Poly5",
            folder_levels=None,
            verbose=False,
        )
        real_files = base_path.glob("*.Poly5")

        assert len(found_files) == len(list(real_files))

    def test_find_folder(self):
        """Test the find_folders function."""
        found_folders = file_discovery.find_folders(
            base_path=base_path, folder_levels=None, verbose=False
        )
        real_folders = [
            p for p in base_path.iterdir() if p.is_dir() and not p.name.startswith(".")
        ]

        assert len(found_folders) == len(real_folders)


class TestFilepathsDict(unittest.TestCase):
    """Test the filepaths_dict helper function."""

    def test_single_file(self):
        """A single (patient, file) pair produces the correct nested dict."""
        result = file_discovery.filepaths_dict([("P_001", "emg_file.Poly5")])
        self.assertIn("P_001", result)
        self.assertIn("emg_file.Poly5", result["P_001"])

    def test_multiple_files_same_patient(self):
        """Multiple files for the same patient are all stored under that patient."""
        result = file_discovery.filepaths_dict(
            [("P_001", "emg_file.Poly5"), ("P_001", "vent_file.Poly5")]
        )
        self.assertIn("emg_file.Poly5", result["P_001"])
        self.assertIn("vent_file.Poly5", result["P_001"])

    def test_multiple_patients(self):
        """Files for different patients are stored under separate keys."""
        result = file_discovery.filepaths_dict(
            [("P_001", "file.Poly5"), ("P_002", "file.Poly5")]
        )
        self.assertIn("P_001", result)
        self.assertIn("P_002", result)

    def test_nested_paths(self):
        """Three-part paths are stored with intermediate folders as keys."""
        result = file_discovery.filepaths_dict([("P_001", "session1", "emg.Poly5")])
        self.assertIn("P_001", result)
        self.assertIn("session1", result["P_001"])
        self.assertIn("emg.Poly5", result["P_001"]["session1"])

    def test_empty_input(self):
        """An empty path list produces an empty dict."""
        self.assertEqual(file_discovery.filepaths_dict([]), {})
