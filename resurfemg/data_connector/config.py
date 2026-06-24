"""Module for configuration of data paths and integrity checks.

Copyright 2022 Netherlands eScience Center and Twente University.
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains methods to let the user configure all paths for data
instead of hard-coding them, as well as methods to check data integrity.
The data integrity can be checked because this file contains hash functions
to track data. Synthetic data can be made with several methods.
"""

from __future__ import annotations

import hashlib
import json
import logging
import textwrap
import tkinter as tk
from pathlib import Path
from tkinter import filedialog
from typing import (
    ClassVar,
    Literal,
    TypeAlias,  # Python 3.10+
    cast,
    get_args,
)

import pandas as pd
from IPython.display import display
from ipywidgets import Layout, widgets

logger = logging.getLogger(__name__)
BUF_SIZE = 65536

PropertyName: TypeAlias = Literal[
    "root_data",
    "test_data",
    "patient_data",
    "simulated_data",
    "preprocessed_data",
    "output_data",
]


def convert_to_os_path(
    path: str,
) -> str:
    """LEGACY. This function converts a path to a os readable path.

    Args:
        path (str): The path to convert.

    Returns:
        str: The converted path.
    """
    import os  # noqa: PLC0415

    return path.replace(os.sep if os.altsep is None else os.altsep, os.sep)


def find_repo_root(marker_file: str = "config_example.json") -> Path | None:
    """Find the root directory of the repository by looking for a marker file.

    Args:
        marker_file (str): The marker file to look for. Defaults to
            "config_example.json".

    Returns:
        Path | None: The absolute path to the root directory of the
            repository, or None if the marker file is not found.
    """
    current_dir = Path(__file__).resolve().parent

    while True:
        if (current_dir / marker_file).exists():
            return current_dir
        if current_dir.parent == current_dir:
            msg = f'Marker file "{marker_file}" not found in any parent directory.'
            logger.error(msg)
            return None
        current_dir = current_dir.parent


class Config:
    """Configuration class for data paths and integrity checks.

    This class allows configuration on the home computer or remote workspace,
    of a file setup for data, which is then processed into a variable.
    Essentially by setting up and modifying a .json file in the appropriate
    directory users can avoid the need for any hardcoded paths to data.
    """

    required_directories: ClassVar[list[str]] = ["root_data"]

    def __init__(
        self,
        location: str | Path | None = None,
        verbose: bool = False,
        force: bool = False,
    ) -> None:
        """Initialize the configuration.

            This function initializes the configuration file. If no location is
            specified it will try to load the configuration file from the default
            locations:
            - ./config.json
            - ~/.resurfemg/config.json
            - /etc/resurfemg/config.json
            - PROJECT_ROOT/config.json


        Args:
                location (str | Path | None): The location of the configuration file.
                verbose (bool): A boolean to print the loaded configuration.
                force (bool): A boolean to overwrite the configuration file.

        Raises:
                ValueError: If the configuration file is not found.
        """
        self._raw: dict | None = None
        self._loaded: dict = {}
        self.example: str = "config_example_resurfemg.json"
        self.repo_root: Path | None = find_repo_root(self.example)
        self.force: bool = force
        self.created_config: bool = False
        # In the ResurfEMG project, the test data is stored in ./test_data
        default_locations = [
            Path("config.json"),
            Path.home() / ".resurfemg" / "config.json",
            Path("/etc/resurfemg/config.json"),
        ]
        if self.repo_root is not None:
            test_path = self.repo_root / "test_data"
            test_data_path = str(test_path) if test_path.is_dir() else "{}/test_data"
            default_locations.append(self.repo_root / "config.json")
        else:
            test_data_path = "{}/test_data"
        self.default_locations = tuple(default_locations)
        self.default_layout = {
            "root_data": "{}/not_pushed",
            "test_data": test_data_path,
            "patient_data": "{}/patient_data",
            "simulated_data": "{}/simulated",
            "preprocessed_data": "{}/preprocessed",
            "output_data": "{}/output",
        }
        self.load(Path(location) if location else None, verbose=verbose)
        self.validate()

    def usage(self) -> str:
        """Provide feedback if the paths are not configured or not configured correctly.

        It contains instructions on how to configure the paths.

        Returns:
            str: Instructions on how to configure the paths.
        """
        locations = "\n".join(str(p) for p in self.default_locations)
        return textwrap.dedent("""
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
            `root_data` directory and adding subdirectories.

            You can override any individual directory (or subdirectory) by
            specifying it in the config.json file.

            "root_data" is expected to exist.
            The "patient_data", "simulated_data", "preprocessed_data",
            "output_data" are optional. They will be created if missing.
            """).format(locations, "/path/to/storage")

    def create_config_from_example(
        self,
        location: Path,
        force: bool = False,
    ) -> None:
        """This function creates a config file from an example file.

        Args:
                location (Path): The location of the example file.
                force (bool): A boolean to overwrite an existing config file.

        Raises:
                ValueError: If the config file already exists and force is False.
        """
        config_path = Path(location).with_name("config.json")
        if Path(config_path).is_file() and not force:
            msg = f"Config file already exists at {config_path}. Use `force=True` to overwrite."
            raise ValueError(msg)
        with Path(location).open("r") as f:
            example = json.load(f)
        with Path(config_path).open("w") as f:
            json.dump(example, f, indent=4, sort_keys=True)

    def load(self, location: str | Path | None, verbose: bool = False) -> None:
        """This function loads the configuration file.

            If no location is specified
            it will try to load the configuration file from the default locations:
            - ./config.json
            - ~/.resurfemg/config.json
            - /etc/resurfemg/config.json
            - PROJECT_ROOT/config.json


        Args:
                location (str | Path | None): The location of the configuration file.
                verbose (bool): A boolean to print the loaded configuration.

        Raises:
                ValueError: If the configuration file is not found.
        """
        locations = [Path(location)] if location is not None else self.default_locations

        for path in locations:
            try:
                with Path(path).open() as f:
                    self._raw = json.load(f)
                    logger.info("Loaded config from: %s", path)
                break
            except (OSError, json.JSONDecodeError) as e:
                logger.info("Failed to load %s: %s", path, e)
        else:
            path = self._create_config_from_example()

        if self._raw is None:
            raise ValueError(self.usage())
        config_dir = Path(path).parent.resolve()
        root = self._raw.get("root_data")
        if isinstance(root, str):
            # A relative root_data is resolved against the config file
            # location.
            root = str((config_dir / root).resolve())

        self._loaded = dict(self._raw)
        self._loaded["root_data"] = root

        if root is None:
            # Without root_data, all other directories must be specified
            # explicitly.
            missing = self.default_layout.keys() - {"root_data"} - self._raw.keys()
            if missing:
                raise ValueError(self.usage())
        else:
            # Resolve the configured directories and back-fill missing ones
            # with the default layout relative to root_data.
            for key in self._raw.keys() | self.default_layout.keys():
                value = self._raw.get(key)
                if isinstance(value, str):
                    self._loaded[key] = str((config_dir / value).resolve())
                elif key in self.default_layout:
                    self._loaded[key] = str(Path(self.default_layout[key].format(root)))

        self._log_loaded(path, verbose=verbose)

    def _create_config_from_example(self) -> Path:
        """Bootstrap a config file in the repo root from the example file.

        Also creates the default root_data directory if it is missing, and
        loads the new config into ``self._raw``.

        Returns:
            Path: The path of the created config file.

        Raises:
            ValueError: If no example file is available.
        """
        if self.repo_root is None or not (self.repo_root / self.example).is_file():
            raise ValueError(self.usage())
        self.create_config_from_example(
            self.repo_root / self.example,
            force=self.force,
        )
        root_path = self.repo_root / "not_pushed"
        if not root_path.is_dir():
            root_path.mkdir(parents=True, exist_ok=True)
            print(f"Created root directory at:\n {root_path}\n")  # noqa: T201
        config_path = self.repo_root / "config.json"
        with config_path.open() as f:
            self._raw = json.load(f)
        self.created_config = True
        return config_path

    def _log_configured_paths(self, directory: str | None = None) -> None:
        if directory is not None:
            msg = f"Directory `{directory}` not found in config. The following directories are configured:"
        else:
            msg = "The following directories are configured:"
        logger.info(msg)
        logger.info("-" * 79)
        logger.info(" %-15s\t%-50s", "Name", "Path")
        logger.info("-" * 79)
        logger.info(" %-15s\t%-50s", "root", self._loaded["root_data"])
        for key, value in self._loaded.items():
            if key != "root_data":
                logger.info(" %-15s\t%-50s", key, value)

    def _log_loaded(self, path: Path, verbose: bool = False) -> None:
        """Log where the config came from and the configured paths.

        Args:
                path (Path): The location of the loaded configuration file.
                verbose (bool): A boolean to print the loaded configuration.
        """
        if self.created_config:
            logger.info("Created config. See and edit it at: %s", path)
        elif verbose:
            logger.info("Loaded config from: %s", path)
        if verbose or self.created_config:
            self._log_configured_paths()

    def validate(self) -> None:
        """This function validates the configuration file.

        It checks if the required directories exist.

        Raises:
            ValueError: If a required directory does not exist.
        """
        for req_dir in self.required_directories:
            if not Path(self._loaded[req_dir]).is_dir():
                logger.error("Required directory %s does not exist", self._loaded[req_dir])
                raise ValueError(self.usage())

    def get_directory(self, directory: str, value: str | None = None) -> str:
        """Return the directory specified in the configuration file.

            If the directory is not specified, it will return the value.


        Args:
                directory (str): The directory to return.
                value (str): The value to return if the directory is not
                    specified.

        Returns:
                str: The directory specified in the configuration file.
        """
        if value is None:
            if directory in self._loaded:
                return self._loaded[directory]
            self._log_configured_paths(directory)
        return str(value)

    def get_config(self) -> dict:
        """This function returns the configuration file.

        Returns:
            dict: The configuration file.
        """
        return self._loaded


def hash_it_up_right_all(origin_directory: str, file_extension: str) -> pd.DataFrame:
    """Hash all files in a directory.

    Hashing function to check files are not corrupted or to assure files are
    changed.

    Args:
        origin_directory (str): The string of the directory with files to
            hash.
        file_extension (str): File extension.

    Returns:
        pandas.DataFrame: The hash values of the files.
    """
    hash_list = []
    file_names = []

    for file in Path(origin_directory).glob("*" + file_extension):
        sha256 = hashlib.sha256()
        with file.open("rb") as f:
            while True:
                data = f.read(BUF_SIZE)
                if not data:
                    break
                sha256.update(data)
        hash_list.append(sha256.hexdigest())
        file_names.append(str(file))

    df = pd.DataFrame(hash_list, file_names)
    df.columns = ["hash"]
    df = df.reset_index()
    return df.rename(columns={"index": "file_name"})


class PathSelector:
    """Create a widget to select a path for a given property name."""

    property_name: PropertyName | None = None
    selected_path: str | Path | None = None

    def __init__(
        self,
        property_name: PropertyName | None = None,
        selected_path: str | Path | None = None,
    ):
        self.property_name = property_name
        self.selected_path = selected_path
        self.path_box = widgets.Text(
            value=(
                str(selected_path)
                if selected_path is not None
                else (Config().get_directory(property_name) if property_name is not None else "")
            ),
            placeholder="Enter path",
            description=(self.property_name if self.property_name is not None else "Path:"),
            disabled=False,
            layout=Layout(width="100%"),
        )
        self.browse_button = widgets.Button(
            icon="folder-open",
            disabled=False,
            layout=Layout(width="40px"),
        )
        self.browse_button.on_click(self._select_path)
        self.widget = widgets.HBox([self.path_box, self.browse_button])

    def _select_path(self, _button: widgets.Button) -> None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # dialog appears on top
        folder = filedialog.askdirectory(title="Select folder")
        root.destroy()
        if folder:  # user didn't cancel
            self.path_box.value = folder

    def _ipython_display_(self) -> None:
        display(self.widget)

    def get_path(self) -> str:
        """Get the selected path."""
        return self.path_box.value


class ConfigCreator:
    """Aggregate multiple path selectors into a single config creator.

    This widget allows users to select paths for all properties and save them.

    Example usage:
    config_creator = ConfigCreator()
    Creates the widget with path selectors for all properties and a save button. The user can select paths and
    save the configuration, which will be loaded into the Config object.

    config = config_creator.get_config()
    Returns the current configuration as a Config object, which can be used in the rest of the codebase
    to access the configured paths.
    """

    def __init__(self, config_path: str | Path | None = None):

        self._config_file_path: Path = Path().cwd() / "config.json" if config_path is None else Path(config_path)
        self.config: Config = Config(location=self._config_file_path, force=True)
        self._path_selectors = {
            name: PathSelector(property_name=name, selected_path=self.config.get_directory(name))
            for name in cast("tuple[PropertyName, ...]", get_args(PropertyName))
        }
        self.save_button = widgets.Button(
            description="Save Config",
            icon="save",
            disabled=False,
        )
        self._config_picker = ConfigPicker(_standalone=True)
        self._config_picker.select_button.on_click(self._load_from_picker)

        self.save_button.on_click(self.save_config)
        self.config_file_selector = widgets.Text(
            value=self._config_file_path.name,
        )
        self.widget = widgets.VBox(
            [selector.widget for selector in self._path_selectors.values()]
            + [self.save_button, self._config_picker.widget],
            layout=Layout(width="100%"),
        )
        self._ipython_display_()

    def _get_config_file_path(self) -> None:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filepath = filedialog.asksaveasfilename(
            title="Save config file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        root.destroy()

        if filepath:  # user didn't cancel
            self._config_file_path = Path(filepath)

    def _ipython_display_(self) -> None:
        display(self.widget)

    def save_config(self, _button: widgets.Button) -> None:
        """Save the config file.

        This function collects the paths from the path selectors and opens a file dialog to save the config file.
        It then saves the config file and updates the current config.
        """
        self._get_config_file_path()
        with self._config_file_path.open("w") as f:
            config = {}
            for property_name, selector in self._path_selectors.items():
                path = selector.get_path()
                if path:
                    config[property_name] = path
            json.dump(config, f)
        self.config = Config(location=self._config_file_path)

    def _load_from_picker(self, _button: widgets.Button) -> None:
        """Update path selectors from the config loaded by config_picker."""
        if self._config_picker.config_file_path is None:
            return
        self.config = self._config_picker.config
        self._config_file_path = self._config_picker.config_file_path
        for name, selector in self._path_selectors.items():
            path = self.config.get_directory(name)
            if path:
                selector.path_box.value = path

    def get_config(self) -> Config:
        """Return the current configuration as Config object."""
        return self.config


class ConfigPicker:
    """Create a widget to select a config file."""

    def __init__(self, _standalone: bool = False):
        self.config: Config = Config()
        self.config_file_path: Path | None = None
        self.select_button = widgets.Button(
            description="Select Config File",
            icon="folder_open",
            disabled=False,
        )
        self.select_button.on_click(self.select_config)
        self.widget = widgets.VBox([self.select_button], layout=Layout(width="100%"))
        if not _standalone:
            self._ipython_display_()

    def _ipython_display_(self) -> None:
        display(self.widget)

    def get_config(self) -> Config:
        """Return the current configuration as Config object."""
        return self.config

    def select_config(self, _button: widgets.Button) -> None:
        """Select a config file.

        Opens a file dialog to select a config file, then loads the config and updates the current config.
        """
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filepath = filedialog.askopenfilename(
            title="Select config file",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )
        root.destroy()

        if filepath:  # user didn't cancel
            self.config_file_path = Path(filepath)
            self.config = Config(location=self.config_file_path)
