# Test linting of project # noqa: INP001

import sys
import unittest
from io import StringIO
from pathlib import Path

import pycodestyle


class TestCodeFormat(unittest.TestCase):
    """Test that the code conforms to PEP-8."""

    def test_conformance(self):
        """Test that we conform to PEP-8."""
        project_dir = Path(__file__).parent.parent
        file_paths = [*list(project_dir.glob("resurfemg/**/*.py")), project_dir / "setup.py"]

        buffer = StringIO()
        sys.stdout = buffer

        style = pycodestyle.StyleGuide()
        result = style.check_files(file_paths)

        print_output = buffer.getvalue()
        sys.stdout = sys.__stdout__

        assert result.total_errors == 0, "Found code style errors (and warnings):\n" + print_output
