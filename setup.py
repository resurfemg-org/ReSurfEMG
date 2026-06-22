#!/usr/bin/env python
import shutil
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

from setuptools import Command, setup

project_dir = Path(__file__).parent.resolve()


class SphinxApiDoc(Command):
    """Generate reStructuredText files from docstrings."""

    description = "run apidoc to generate documentation"

    user_options: ClassVar[list] = []

    def initialize_options(self) -> None:
        """Set default values for options."""

    def finalize_options(self) -> None:
        """Post-process options."""

    def run(self) -> None:
        """Run command."""
        from sphinx.ext.apidoc import main  # noqa: PLC0415

        src = Path(project_dir) / "docs"
        special = (
            "index.rst",
            "developers.rst",
            "medical-professionals.rst",
        )

        for f in Path(src).glob("*.rst"):
            for end in special:
                if str(f).endswith(end):
                    os.utime(f, None)
                    break
            else:
                Path(f).unlink()

        sys.exit(
            main(
                [
                    "-o",
                    str(src),
                    "-f",
                    str(Path(project_dir) / "resurfemg"),
                    "-d",
                    "4",
                    "--separate",
                ]
            )
        )


class SphinxDoc(Command):
    """Build Sphinx documentation."""

    description = "generate documentation"

    user_options: ClassVar[list] = [("wall", "W", "Warnings are errors")]

    def initialize_options(self) -> None:
        """Set default values for options."""
        self.wall = True

    def finalize_options(self) -> None:
        """Post-process options."""

    def run(self) -> None:
        """Run command."""
        from sphinx.application import Sphinx  # noqa: PLC0415
        from sphinx.cmd.build import handle_exception  # noqa: PLC0415
        from sphinx.util.console import nocolor  # noqa: PLC0415
        from sphinx.util.docutils import (  # noqa: PLC0415
            docutils_namespace,
            patch_docutils,
        )

        name = "resurfemg"
        try:
            tag = (
                subprocess.check_output(  # noqa: S603
                    [
                        shutil.which("git") or "git",
                        "--no-pager",
                        "describe",
                        "--abbrev=0",
                        "--tags",
                    ],
                    stderr=subprocess.DEVNULL,
                )
                .strip()
                .decode()
            )
        except subprocess.CalledProcessError:
            tag = "v0.0.0"

        version = tag[1:]

        nocolor()
        confoverrides = {}
        confoverrides["project"] = name
        confoverrides["version"] = version
        confoverrides["autodoc_mock_imports"] = ["adi"]
        confdir = Path(project_dir) / "docs"
        srcdir = confdir
        builder = "html"
        build_dir = Path(project_dir) / "build" / "sphinx"
        builder_target_dir = build_dir / builder
        app = None

        try:
            with patch_docutils(str(confdir)), docutils_namespace():
                app = Sphinx(
                    srcdir,
                    confdir,
                    builder_target_dir,
                    build_dir / "doctrees",
                    builder,
                    confoverrides,
                    sys.stdout,
                    freshenv=False,
                    warningiserror=self.wall,
                    verbosity=self.distribution.verbose - 1,
                    keep_going=False,
                )
                app.build(force_all=False)
                if app.statuscode:
                    sys.stderr.write(
                        f"Sphinx builder {app.builder.name} failed.",
                    )
                    raise SystemExit(8)  # noqa: TRY301
        except Exception as e:
            handle_exception(app, self, e, sys.stderr)
            raise


if __name__ == "__main__":
    setup(
        use_scm_version=True,
        long_description=(project_dir / "README.md").read_text(),
        long_description_content_type="text/markdown",
        cmdclass={
            "apidoc": SphinxApiDoc,
            "build_sphinx": SphinxDoc,
        },
    )
