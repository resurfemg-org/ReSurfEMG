# Test linting of project # noqa: INP001

import subprocess
import sys
import unittest
from pathlib import Path


class TestCodeFormat(unittest.TestCase):
    """Test that the code conforms to Ruff standards defined in pyproject.toml."""

    def test_conformance(self):
        """Test that we conform to Ruff linting rules."""
        project_dir = Path(__file__).parent.parent
        result = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "resurfemg/"],
            capture_output=True,
            check=False,
            text=True,
            cwd=project_dir,
        )
        assert result.returncode == 0, "Found Ruff linting errors:\n" + result.stdout + result.stderr
