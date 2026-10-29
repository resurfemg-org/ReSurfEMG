"""sanity tests for the file_discovery submodule of the resurfemg library"""

import glob
import os
import platform
import unittest

from resurfemg.data_connector import file_discovery

base_path = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
)


class TestFileDiscovery(unittest.TestCase):
    def test_find_files(self):
        found_files = file_discovery.find_files(
            base_path=base_path,
            file_name_regex="*",
            extension_regex="Poly5",
            folder_levels=None,
            verbose=False,
        )
        real_files = glob.glob(
            os.path.join(base_path, "*.Poly5"), recursive=False
        )

        self.assertEqual(
            (len(found_files)),
            len(real_files),
        )

    def test_find_folder(self):
        if platform.system() == "Windows":
            path_sep = "\\"
        else:
            path_sep = "/"
        found_folders = file_discovery.find_folders(
            base_path=base_path, folder_levels=None, verbose=False
        )
        real_folders = glob.glob(
            os.path.join(base_path, "*" + path_sep), recursive=False
        )

        self.assertEqual(
            (len(found_folders)),
            len(real_folders),
        )
