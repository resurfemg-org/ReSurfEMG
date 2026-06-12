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
