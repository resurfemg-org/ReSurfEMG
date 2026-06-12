"""This file contains functions to automatically find specified files and folders.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)


def _resolve_data_pattern(
    file_name_regex: str | None, extension_regex: str | None
) -> str:
    """Validate the name/extension patterns and build the glob pattern."""
    if file_name_regex is None:
        file_name_regex = "**"
    elif not isinstance(file_name_regex, str):
        msg = "file_name_regex should be a str."
        raise ValueError(msg)

    if extension_regex is None:
        extension_regex = "**"
    elif not isinstance(extension_regex, str):
        msg = "extension_regex should be a str."
        raise ValueError(msg)
    extension_regex = extension_regex.removeprefix(".")

    return f"**/{file_name_regex}.{extension_regex}"


def _resolve_folder_levels(
    folder_levels: list[str] | None,
) -> tuple[int | None, list[str]]:
    """Determine the matching depth and the DataFrame column names."""
    if isinstance(folder_levels, list):
        return len(folder_levels), [*folder_levels, "files"]
    if folder_levels is None:
        return None, ["files"]
    msg = "Provide either a list, or None as folder_levels."
    raise ValueError(msg)


def _classify_files(
    matching_files: Iterable[Path], base_path: Path, depth: int | None
) -> tuple[list[str | list[str]], list[list[str]]]:
    """Split matches into rows that fit the depth and those that don't."""
    matching: list[str | list[str]] = []
    non_matching: list[list[str]] = []
    for file in matching_files:
        rel = file.relative_to(base_path)
        parts = list(rel.parts)
        if depth is None:
            matching.append(str(rel))
        elif len(parts) == depth + 1:
            matching.append(parts)
        else:
            non_matching.append(parts)
    return matching, non_matching


def find_files(
    base_path: str,
    file_name_regex: str | None = None,
    extension_regex: str | None = None,
    folder_levels: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Find files matching the provided name and extension patterns.

    Find files with the file name and extension according to filename pattern
    `file_name_regex`.`extension_regex` starting from the provided base_path
    according to the provided folder_levels. If `folder_levels` is None, all
    files matching the name pattern are included, no matter the data
    organisation.

    Args:
        base_path (str): Path to starting directory.
        file_name_regex (str): File name pattern, see Python Regex docs.
        extension_regex (str): File extension pattern, see Python Regex docs.
        folder_levels (list[str] | None): Data directory organisation, e.g.
            ['patient', 'date'].
        verbose (bool): Provide feedback about non-included files.

    Returns:
        pd.DataFrame: Matching file paths tabled by the folder_levels.

    Raises:
        ValueError: If base_path does not exist or the patterns or
            folder_levels have an invalid type.
    """
    root = Path(base_path)
    if not root.is_dir():
        msg = f"Specified base_path {base_path} cannot be found."
        raise ValueError(msg)

    data_pattern = _resolve_data_pattern(file_name_regex, extension_regex)
    depth, folder_levels = _resolve_folder_levels(folder_levels)

    matching_files = root.glob(data_pattern)
    matching_files_structure, non_matching_files_structure = _classify_files(
        matching_files, root, depth
    )

    files = pd.DataFrame(matching_files_structure, columns=folder_levels)
    if verbose and non_matching_files_structure:
        logger.info(
            "These files did not match the provided depth:\n%s",
            non_matching_files_structure,
        )
    return files


def find_folders(
    base_path: str | Path,
    folder_levels: list[str] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Find folders up to the depth of the provided folder_levels.

    Find folders starting from the provided base_path. If `folder_levels` is
    None, all folders in the provided base_path are included, no matter the
    data organisation.

    Args:
        base_path (str | Path): Path to starting directory.
        folder_levels (list[str] | None): Data directory organisation, e.g.
            ['patient', 'date'].
        verbose (bool): Provide feedback about non-included folders.

    Returns:
        pd.DataFrame: Folder paths tabled by the folder_levels.

    Raises:
        ValueError: If base_path does not exist or folder_levels has an
            invalid type.
    """
    root = Path(base_path)
    if not root.is_dir():
        msg = "Specified base_path cannot be found."
        raise ValueError(msg)

    if isinstance(folder_levels, list):
        depth = len(folder_levels)
    elif folder_levels is None:
        depth = None
        folder_levels = ["destination"]
    else:
        msg = "Provide either a list, or None as folder_levels."
        raise ValueError(msg)

    pattern = "*" if depth is None else "/".join(["*"] * depth)
    matching_dirs = [
        p
        for p in root.glob(pattern)
        if p.is_dir()
        and not any(part.startswith(".") for part in p.relative_to(root).parts)
    ]

    matching_path_structure = []
    non_matching_path_structure = []
    for path in matching_dirs:
        parts = list(path.relative_to(root).parts)
        if depth is None:
            matching_path_structure.append(parts[0])
        elif len(parts) >= depth:
            matching_path_structure.append(parts[:depth])
        else:
            non_matching_path_structure.append(parts)

    folders = pd.DataFrame(matching_path_structure, columns=folder_levels)
    if verbose and non_matching_path_structure:
        logger.info(
            "These paths did not match the provided depth:\n%s",
            non_matching_path_structure,
        )
    return folders
