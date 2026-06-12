"""Sanity tests for the ipy_widgets submodule of the resurfemg library."""  # noqa: INP001

import unittest
from pathlib import Path

from resurfemg.data_connector import file_discovery
from resurfemg.pipelines import ipy_widgets

base_path = Path(__file__).resolve().parents[2] / "test_data"


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
