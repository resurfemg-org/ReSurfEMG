"""Sanity tests for the config functions."""  # noqa: INP001

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import pytest

from resurfemg.data_connector.config import Config

logger = logging.getLogger(__name__)


class TestConfig(TestCase):
    """Test the Config class."""

    required_directories = {  # noqa: RUF012
        "root_data",
    }
    required_directories = ["root_data"]  # noqa: PIE794, RUF012

    def test_roots_only(self):
        """Test that the config loads when only the required directory is provided."""
        with TemporaryDirectory() as td:
            root_dir = Path(td) / "root"
            root_dir.mkdir()
            config_file = Path(td) / "config.json"
            logger.debug("Created temporary directory at %s", td)
            with config_file.open("w") as f:
                json.dump({"root_data": str(root_dir)}, f)

            config = Config(config_file)
            assert config.get_directory("root_data")

    def test_missing_config_path(self) -> None:
        """Test that Config raises when the config file is missing."""
        with pytest.raises(ValueError):
            Config("non existent")
