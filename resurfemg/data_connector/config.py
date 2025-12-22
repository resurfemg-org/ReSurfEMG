"""
Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to let the user configure all paths for data
instead of hard-coding them, as well as methods to check data integrity.
The data integrity can be checked because this file contains hash functions
to track data. Synthetic data can be made with several methods.
"""

import json
import logging
import os
import textwrap
import hashlib
import glob

from pathlib import Path
from typing import Optional, Sequence

from importlib.resources import files
import pandas as pd


def convert_to_os_path(
    path: str,
):
    """
    This function converts a path to a os readable path.
    -----------------------------------------------------------------------
    :param path: The path to convert.
    :type path: str
    """
    readable_path = path.replace(
        os.sep if os.altsep is None else os.altsep, os.sep)
    return readable_path


def find_project_root(
        current_dir=Path.cwd(), marker_file=None, prefer_build_markers=True):
    """
    Walks upward from `current_dir` to find a plausible Python project root.
    Returns the path if found, else None. If `marker_file` is provided,
    it is uses the `marker_file` in favor of looking for standard build/
    VCS markers. If `prefer_build_markers` is True, prioritize pyproject/setup
    files over .git.
    :param current_dir: The directory to start searching from.
    :type current_dir: Path
    :param marker_file: The marker file to look for (default is
    'config_example.json').
    :type marker_file: Optional[str]
    :param prefer_build_markers: Whether to prefer build markers over VCS
    markers.
    :type prefer_build_markers: bool
    :return: The absolute path to the root directory of the repository.
    :rtype: Optional[Path]
    """
    start = current_dir.resolve()
    if marker_file is not None:
        build_markers = {marker_file}
        vcs_markers = set()
        prefer_build_markers = True
    else:
        build_markers = {"pyproject.toml", "setup.cfg", "setup.py",
                         "poetry.lock", "Pipfile", "requirements.txt",
                         "tox.ini"}
        vcs_markers = {".git"}

    candidate_by_priority = None

    for current in [start, *start.parents]:
        items = ({p.name for p in current.iterdir()}
                 if current.exists() else set())

        # Build-markers = Check intersection between items and build markers
        if prefer_build_markers and items & build_markers:
            return current

        # Record .git as a candidate (don’t return immediately if preferring
        # build markers)
        if marker_file is None and items & vcs_markers:
            candidate_by_priority = candidate_by_priority or current

    # If we didn’t find a build marker, use .git candidate
    if candidate_by_priority:
        return candidate_by_priority
    if marker_file is not None:
        print(f"Marker file '{marker_file}' not found in any parent directory")
    return None


class Config:
    """
    This class allows configuration on the home computer or remote workspace,
    of a file setup for data, which is then processed into a variable.
    Essentially by setting up and modifying a .json file in the appropriate
    directory users can avoid the need for any hardcoded paths to data.
    -----------------------------------------------------------------------
    """

    required_directories = ['root_data']

    def __init__(self, location=None, verbose=False, force=False):
        """
        This function initializes the configuration file. If no location is
        specified it will try to load the configuration file from the default
        locations:
        - ./config.json
        - ~/.resurfemg/config.json
        - /etc/resurfemg/config.json
        - PROJECT_ROOT/config.json
        -----------------------------------------------------------------------
        :param location: The location of the configuration file.
        :type location: str
        :param verbose: A boolean to print the loaded configuration.
        :type verbose: bool
        :param force: A boolean to overwrite the configuration file.
        :type force: bool
        :raises ValueError: If the configuration file is not found.
        """
        self._raw = None
        self._loaded = None
        self.example = 'config_example.json'
        self.repo_root = find_project_root()
        self.force = force
        self.created_config = False
        self.relative_paths = []
        if self.repo_root is not None:
            # In the ResurfEMG project, the test data is stored in ./test_data
            test_path = os.path.join(self.repo_root, 'test_data')
            if len(glob.glob(test_path)) == 1:
                test_data_path = os.path.join(self.repo_root, 'test_data')
            else:
                test_data_path = '{}/test_data'

            self.default_locations = (
                './config.json',
                os.path.expanduser('~/.resurfemg/config.json'),
                '/etc/resurfemg/config.json',
                os.path.join(self.repo_root, 'config.json'),
            )
        else:
            test_data_path = '{}/test_data'
            self.default_locations = (
                './config.json',
                os.path.expanduser('~/.resurfemg/config.json'),
                '/etc/resurfemg/config.json',
            )
        self.default_layout = {
                'root_data': '{}/not_pushed',
                'test_data': test_data_path,
                'patient_data': '{}/patient_data',
                'simulated_data': '{}/simulated',
                'preprocessed_data': '{}/preprocessed',
                'output_data': '{}/output',
            }
        _path = self.load(location)
        self.parse_paths(_path)
        if self.created_config:
            print(f'Created config. See and edit it at:\n {_path}\n')
        elif verbose:
            print(f'Loaded config from:\n {_path}\n')
        if verbose or self.created_config:
            print('The following name-path combinations are configured:')
            self.print_config()
        self.validate(_path, force=force)

    def usage(self):
        """
        Provide feedback if the paths are not configured or not configured
        correctly. It contains instructions on how to configure the paths.
        -----------------------------------------------------------------------
        """
        return textwrap.dedent(
            '''
            Cannot load config.

            Please create a file in either one of the locations
            listed below:
            {}

            With the contents that specifies at least the root
            directory as follows:

            {{
                "root_data": "{}"
            }}

            The default directory layout is expected to be based on the above
            "root_data" directory and adding subdirectories.

            You can override any individual directory (or subdirectory) by
            specifying it in the config.json file.

            "root_data" is expected to exist.
            The "patient_data", "simulated_data", "preprocessed_data",
            "output_data" are optional. They will be created if missing.
            '''
        ).format(convert_to_os_path('\n'.join(self.default_locations)),
                 convert_to_os_path('/path/to/storage'))

    def print_config(self):
        """
        This function prints the current configuration.
        -----------------------------------------------------------------------
        """
        print(79*'-')
        print(f'  {"Name": <15}\t{"Path": <50}')
        print(79*'-')
        print(f'  {"root_data": <15}\t{self._loaded["root_data"]: <50}')
        print(79*'-')
        for key, value in self._loaded.items():
            if key != 'root_data':
                _rel_flag = '* ' if key in self.relative_paths else '  '
                print(_rel_flag + f'{key: <15}' + f'\t{value: <50}')
        print(79*'-')
        if len(self.relative_paths) > 0:
            print('* Path is relative to root_data')

    def create_config_from_example(self, location: str, force=False):
        """
        This function creates a config file from an example file.
        -----------------------------------------------------------------------
        :param location: The location of the example file.
        :type location: str
        :raises ValueError: If the example file is not found.
        """
        example_path = files("resurfemg").joinpath(
            "data_connector/config_example.json")
        config_path = os.path.join(location, 'config.json')
        if os.path.isfile(config_path) and not force:
            raise ValueError(
                f'Config file already exists at {config_path}.'
                + ' Use `force=True` to overwrite.')
        with open(example_path, 'r') as f:
            example = json.load(f)
        # Adjust root_path to an absolute path
        if example['root_data'].startswith('.'):
            example['root_data'] = os.path.join(
                location, example['root_data'][2:])
        # Write the example config to the config path
        with open(config_path, 'w') as f:
            json.dump(example, f, indent=4, sort_keys=False)
        self.created_config = True
        # Create the root data directory if it does not exist
        if not os.path.isdir(example['root_data']):
            if force:
                os.makedirs(example['root_data'])
                print(f'Created root directory at:\n {example["root_data"]}\n')
            else:
                raise ValueError(
                    'Root data directory specified in the config file does '
                    + 'not exist. Create it yourself or re-run with '
                    + '`force=True` to create the root data directory at:\n '
                    + f'{example["root_data"]}')

    def load(self, location):
        """
        This function loads the configuration file. If no location is specified
        it will try to load the configuration file from the default locations:
        - ./config.json
        - ~/.resurfemg/config.json
        - /etc/resurfemg/config.json
        - PROJECT_ROOT/config.json
        -----------------------------------------------------------------------
        :param location: The location of the configuration file.
        :type location: str
        :param verbose: A boolean to print the loaded configuration.
        :type verbose: bool
        :raises ValueError: If the configuration file is not found.
        """
        locations = (
            [location] if location is not None else self.default_locations
        )

        for _path in locations:
            try:
                with open(_path) as f:
                    self._raw = json.load(f)
                    break
            except Exception as e:
                logging.info('Failed to load %s: %s', _path, e)
        else:
            if self.repo_root is not None:
                self.create_config_from_example(
                    self.repo_root,
                    force=self.force,)
                with open(os.path.join(self.repo_root, 'config.json')) as f:
                    self._raw = json.load(f)
            else:
                raise ValueError(self.usage())
        return _path

    def parse_paths(self, _path):
        """
        This function parses the paths in the configuration file.
        -----------------------------------------------------------------------
        """
        # Check if user specified all required directories.
        for directory in self.required_directories:
            if directory not in self._raw:
                raise ValueError(f'Missing required directory `{directory}` '
                                 + 'in config file.' + self.usage())

        # Convert all paths to OS readable paths.
        root = self._raw.get('root_data')
        root = convert_to_os_path(root)
        config_path = os.path.abspath(_path.replace('config.json', ''))
        if isinstance(root, str) and root.startswith('.'):
            root = root.replace('.', config_path, 1)
        self._loaded = dict(self._raw)
        self._loaded['root_data'] = root
        for key, value in self._raw.items():
            if isinstance(value, str) and value.startswith('.'):
                new_value = value.replace('.', root, 1)
                self._loaded[key] = convert_to_os_path(new_value)
                self.relative_paths.append(key)
            else:
                self._loaded[key] = convert_to_os_path(value)
        # User possibly specified only a subset of optional directories.
        # The missing directories will be back-filled with the default
        # layout relative to the root directory.
        missing = set(self.default_layout.keys()) - set(self._raw.keys())
        for m in missing:
            self._loaded[m] = convert_to_os_path(
                self.default_layout[m].format(root))

    def validate(self, _path, force=False):
        """
        This function validates the configuration file. It checks if the
        required directories exist.
        -----------------------------------------------------------------------
        :raises ValueError: If a required directory does not exist.
        """
        for req_dir in self.required_directories:
            if not os.path.isdir(self._loaded[req_dir]):
                if force:
                    os.makedirs(self._loaded[req_dir])
                    print(f'Created required directory at:\n '
                          + f'{self._loaded[req_dir]}\n')
                    continue
                logging.error(
                    f'Required directory {req_dir} specified in the config '
                    + 'file does not exist. Create it yourself or re-run with '
                    + '`force=True` to create the root data directory at:\n '
                    + f'{self._loaded[req_dir]}\n'
                    + 'Alternatively, edit the config file at:\n '
                    + f'{_path}\n')

    def get_directory(self, directory, value=None):
        """
        This function returns the directory specified in the configuration
        file. If the directory is not specified, it will return the value.
        -----------------------------------------------------------------------
        :param directory: The directory to return.
        :type directory: str
        :param value: The value to return if the directory is not specified.
        :type value: str
        :return: The directory specified in the configuration file.
        :rtype: str
        """
        if value is None:
            if directory in self._loaded:
                return self._loaded[directory]
        print(f"Directory `{directory}` not found in config.\n"
              + "The following directories are configured:")
        self.print_config()
        return value

    def get_config(self):
        """
        This function returns the configuration file.
        -----------------------------------------------------------------------
        :return: The configuration file.
        :rtype: dict
        """
        return self._loaded


def hash_it_up_right_all(origin_directory, file_extension):
    """Hashing function to check files are not corrupted or to assure
    files are changed. This function hashes all files in a directory.
    -----------------------------------------------------------------------
    :param origin_directory: The string of the directory with files to hash
    :type origin_directory: str
    :param file_extension: File extension
    :type file_extension: str

    :returns df: The hash values of the files
    :rtype df: pandas.DataFrame
    """
    hash_list = []
    file_names = []
    files = '*' + file_extension
    non_suspects1 = glob.glob(os.path.join(origin_directory, files))
    BUF_SIZE = 65536
    for file in non_suspects1:
        sha256 = hashlib.sha256()
        with open(file, 'rb') as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        result = sha256.hexdigest()
        hash_list.append(result)
        file_names.append(file)

    df = pd.DataFrame(hash_list, file_names)
    df.columns = ["hash"]
    df = df.reset_index()
    df = df.rename(columns={'index': 'file_name'})

    return df
