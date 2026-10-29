"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions to automatically find specified files and folders.
"""

import glob
import os
import platform

import pandas as pd


def find_files(
    base_path,
    file_name_regex=None,
    extension_regex=None,
    folder_levels=None,
    verbose=True,
):
    """Find files with the file name and extension according to filename pattern
    `file_name_regex`.`extension_regex` starting from the provided base_path
    according to the provided folder_leves. If `folder_levels` is None, all
    files matching the name pattern included, no matter the data organisation.
    ---------------------------------------------------------------------------
    :param base_path: Path to starting directory
    :type base_path: str
    :param file_name_regex: file name pattern, see Python Regex docs
    :type file_name_regex: str
    :param extension_regex: file extension pattern, see Python Regex docs
    :type extension_regex: str
    :param folder_levels: data directory organisation, e.g. ['patient', 'date']
    :type folder_levels: list(str) or str
    :param verbose: Provide feedback about non-included files
    :type verbose: bool

    :returns files: Matching file paths tabled by the folder_levels
    :rtype files: pd.DataFrame
    """
    if not os.path.isdir(base_path):
        msg = "Specified base_path cannot be found."
        raise ValueError(msg)

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

    if isinstance(folder_levels, list):
        depth = len(folder_levels)
        folder_levels.append("files")
    elif folder_levels is None:
        depth = None
        folder_levels = ["files"]
    else:
        msg = "Provide either a list, or None as folder_levels."
        raise ValueError(msg)

    if platform.system() == "Windows":
        path_sep = "\\"
    else:
        path_sep = "/"

    data_pattern = os.path.join(
        base_path, "**" + path_sep + file_name_regex + "." + extension_regex
    )
    matching_files = glob.glob(data_pattern, recursive=True)
    files_structure = list()
    for file_name in matching_files:
        files_structure.append(
            file_name.replace(base_path + path_sep, "").split(path_sep)
        )

    matching_files_structure = list()
    non_matching_files_structure = list()
    for file in files_structure:
        if depth is None:
            matching_files_structure.append(path_sep.join(file))
        elif (isinstance(file, str) and (depth == 0)) or (
            isinstance(file, list) and (len(file) == depth + 1)
        ):
            matching_files_structure.append(file)
        else:
            non_matching_files_structure.append(file)

    files = pd.DataFrame(matching_files_structure, columns=folder_levels)
    if verbose is True and len(non_matching_files_structure) > 0:
        print(
            "These files did not match the provided depth:\n",
            (non_matching_files_structure),
        )
    return files


def find_folders(base_path, folder_levels=None, verbose=True):
    """Find folders up to the depth of the provided folder_levels starting from
    the provided base_path. If `folder_levels` is None, all folders in the
    provided are included, no matter the data organisation.
    ---------------------------------------------------------------------------
    :param base_path: Path to starting directory
    :type base_path: str
    :param folder_levels: data directory organisation, e.g. ['patient', 'date']
    :type folder_levels: list(str) or str
    :param verbose: Provide feedback about non-included files
    :type verbose: bool

    :returns folders: Folder paths tabled by the folder_levels
    :rtype folders: pd.DataFrame
    """
    if not os.path.isdir(base_path):
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

    if platform.system() == "Windows":
        path_sep = "\\"
    else:
        path_sep = "/"

    if depth is None:
        data_pattern = os.path.join(base_path, "*" + path_sep)
    else:
        data_pattern = os.path.join(base_path, depth * ("*" + path_sep))
    matching_paths = glob.glob(data_pattern, recursive=False)

    path_structure = list()
    for path_name in matching_paths:
        path_structure.append(
            path_name.replace(base_path + path_sep, "").split(path_sep)
        )

    matching_path_structure = list()
    non_matching_path_structure = list()
    for sub_path in path_structure:
        if depth is None:
            if isinstance(sub_path, str):
                matching_path_structure.append(path_sep.join(sub_path))
            elif isinstance(sub_path, list):
                matching_path_structure.append(sub_path[0])
        elif isinstance(sub_path, str) and (depth == 0):
            matching_path_structure.append(sub_path)
        elif isinstance(sub_path, list) and (len(sub_path) >= depth):
            matching_path_structure.append(sub_path[:depth])
        else:
            non_matching_path_structure.append(sub_path)

    folders = pd.DataFrame(matching_path_structure, columns=folder_levels)
    if verbose is True and len(non_matching_path_structure) > 0:
        print(
            "These paths did not match the provided depth:\n",
            (non_matching_path_structure),
        )
    return folders
