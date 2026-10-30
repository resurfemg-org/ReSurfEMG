"""sanity tests for the ipy_widgets submodule of the resurfemg library"""

import os
import unittest

from resurfemg.data_connector import file_discovery
from resurfemg.pipelines import ipy_widgets

base_path = os.path.join(
    os.path.abspath(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ),
    "test_data",
)


class TestIpyWidgets(unittest.TestCase):
    def test_file_select(self):
        files = file_discovery.find_files(
            base_path=base_path,
            file_name_regex="*",
            extension_regex="Poly5",
            folder_levels=None,
            verbose=False,
        )
        files.sort_values(by="files", inplace=True)
        button_list = ipy_widgets.file_select(
            files=files,
            folder_levels=["files"],
            default_value_select=[files["files"].values[0]],
        )
        self.assertEqual(
            (button_list[0].value),
            files["files"].values[0],
        )
