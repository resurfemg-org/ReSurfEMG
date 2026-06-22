"""This file contains Jupyter widgets to perform default procedures.

NB The functions in this file required the development installation including
Jupyter (see README.md)

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, ClassVar, Literal, cast, overload

import ipywidgets as widgets
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import Button, Checkbox, Dropdown, HBox, Text, VBox
from scipy import io as sio

from resurfemg.data_connector.data_classes import (
    EmgDataGroup,
    TimeSeriesGroup,
    VentilatorDataGroup,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)


@overload
def _check_defaults(
    default_select: list | None,
    folder_levels: list[str],
    default_type: Literal["value"],
) -> tuple[list[str] | None, list[bool]]: ...


@overload
def _check_defaults(
    default_select: list | None,
    folder_levels: list[str],
    default_type: Literal["idx"],
) -> tuple[list[int] | None, list[bool]]: ...


def _check_defaults(
    default_select: list | None,
    folder_levels: list[str],
    default_type: Literal["value", "idx"] = "value",
) -> tuple[list[str] | list[int] | None, list[bool]]:
    name_types_regex: dict[str, type] = {"value": str, "idx": int}
    if default_select is None:
        default_select = len(folder_levels) * [None]
        options_bool = len(folder_levels) * [False]
    elif not isinstance(default_select, list):
        msg = f"default_{default_type}_select must be a list or None"
        raise TypeError(msg)
    else:
        if len(default_select) < len(folder_levels):
            msg_0 = f"len(default_{default_type}_select) < len(folder_levels)"
            raise IndexError(msg_0)
        options_bool = []
        for value in default_select:
            if value is not None and not isinstance(
                value, name_types_regex[default_type]
            ):
                msg_0 = (
                    f"default_{default_type}_select values need to be "
                    f"{name_types_regex[default_type].__name__} or None"
                )
                raise TypeError(msg_0)
            options_bool.append(value is not None)
    return default_select, options_bool


def file_select(
    files: pd.DataFrame,
    folder_levels: list[str],
    default_value_select: list[str] | None = None,
    default_idx_select: list[int] | None = None,
) -> list[widgets.Dropdown]:
    """A widget for file selection for organised/nested data.

    default_value_select precedes default_idx_select in default value identification.

    Args:
        files (pd.DataFrame): File paths tabled by the folder_levels.
        folder_levels (list[str]): Data directory organisation, e.g.
            ["patient", "date"].
        default_value_select (list[str], optional): Default values to select
            per folder_level.
        default_idx_select (list[int], optional): Default index to select per
            folder_level.

    Returns:
        list[ipywidgets.Dropdown]: List of dropdown widgets for file selection.
    """
    if not isinstance(files, pd.DataFrame):
        msg = "Files not provided in valid format."
        raise TypeError(msg)

    if not isinstance(folder_levels, list):
        msg = "Provide either a list as folder_levels."
        raise TypeError(msg)

    default_value_select, value_options_bool = _check_defaults(
        default_value_select, folder_levels, default_type="value"
    )
    default_idx_select, idx_options_bool = _check_defaults(
        default_idx_select, folder_levels, default_type="idx"
    )

    button_list = []
    btn_dict = {}
    for _, folder_level in enumerate(folder_levels):
        _btn = widgets.Dropdown(
            description=folder_level + ":",
            disabled=False,
        )
        button_list.append(_btn)
        btn_dict[folder_level] = _btn

    prev_values: list[str | None] = cast(
        "list[str | None]", [None] * len(folder_levels)
    )

    @widgets.interact(**btn_dict)
    def _update_select(**kwargs) -> None:
        _update_dropdowns(
            kwargs,
            files,
            folder_levels,
            button_list,
            prev_values,
            value_options_bool,
            idx_options_bool,
            default_value_select,
            default_idx_select,
        )

    return button_list


def _update_dropdowns(
    btn_dict: dict,
    files: pd.DataFrame,
    folder_levels: list[str],
    button_list: list[widgets.Dropdown],
    prev_values: list[str | None],
    value_options_bool: list[bool],
    idx_options_bool: list[bool],
    default_value_select: list[str] | None,
    default_idx_select: list[int] | None,
) -> None:
    btn_changed = [
        button_list[_idx].value != prev_values[_idx]
        for _idx in range(len(folder_levels))
    ]

    for idx, dict_key in enumerate(btn_dict):
        btn_idx = folder_levels.index(dict_key)
        _btn = button_list[btn_idx]

        if idx == 0:
            filter_files = files
        else:
            bool_list = [
                button_list[_idx].value == files[folder_levels[_idx]].values
                for _idx in range(idx)
            ]
            filter_files = files[np.all(np.array(bool_list), 0)]

        col_values: np.ndarray = filter_files[dict_key].to_numpy()
        options = list(np.unique(col_values))
        options.sort()

        _btn.options = options
        if options:
            value = _btn.value if _btn.value in options else options[0]
            if any(btn_changed[:btn_idx]) or prev_values[btn_idx] is None:
                if value_options_bool[btn_idx] and default_value_select is not None:
                    value = (
                        default_value_select[btn_idx]
                        if default_value_select[btn_idx] in options
                        else options[0]
                    )
                elif idx_options_bool[btn_idx] and default_idx_select is not None:
                    value = (
                        options[default_idx_select[btn_idx]]
                        if default_idx_select[btn_idx] < len(options)
                        else options[0]
                    )
            _btn.value = value
            prev_values[btn_idx] = value


class CustomizeMatlabImport:
    """A widget to customize the import of .mat files.

    The widget identifies the keys in the .mat file and classifies them according to their type
    (time series, descriptors, params) and group (EMG, ventilator, other). The user can then select which keys to import
    how to name them, and which signal group they belong to (EMG, ventilator, other).
    The widget then builds the EmgDataGroup, VentilatorDataGroup and TimeSeriesGroup objects accordingly.
    """

    data_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "EMG": ("emg", "exg"),
        "ventilator": ("vent", "flow", "pressure", "rip"),
        "other": (),
    }

    name_types_regex: ClassVar[dict[str, dict[str, tuple[str, ...]]]] = {
        "time_series": {
            "t_data": (r"(?:^|_)t(?:$|_)", r"time"),
            "y_raw": (),
            "other": (),
        },
        "descriptors": {"labels": (), "y_units": (), "other": ()},
        "params": {
            "fs": (r"(?:^|_)fs(?:$|_)", r"sampl(?:e|ing)", r"sample.?rate"),
            "n_samp": (),
            "n_channels": (),
            "other": (),
        },
    }

    def __init__(self, mat_file: str | Path | None = None):
        self.mat_dict: dict = sio.loadmat(mat_file, mdict=None, appendmat=False)
        self.keys = [key for key in self.mat_dict if not key.startswith("__")]
        self.key_mapping: dict[
            str, dict[Literal["Type", "Group", "Contains"] | None, str | None]
        ] = {}
        self._map_keys()
        self.emg_data: EmgDataGroup
        self.ventilator_data: VentilatorDataGroup
        self.other_data: TimeSeriesGroup
        self.picker = self._create_pickers()
        display(self.picker)
        self.import_button = Button(description="Import selected data")
        self.import_button.on_click(self._import_data)
        display(self.import_button)

    def _map_keys(self) -> None:
        """Map the keys in the .mat file to their type, group and content based on regex matching."""
        for key in self.keys:
            self.key_mapping[str(key)] = {"Type": None, "Group": None, "Contains": None}
            _num_type_check = isinstance(
                self.mat_dict[key], np.ndarray
            ) and np.issubdtype(
                self.mat_dict[key].dtype, np.number
            )  # check if the key corresponds to a numerical array, which is required for time series and params types
            if (
                self.mat_dict[key].shape
                == (
                    1,
                    1,
                )  # check if the key corresponds to a single value, which is required for params type
                and _num_type_check
            ):
                self.key_mapping[str(key)]["Type"] = "params"
            elif max(self.mat_dict[key].shape) > 1 and _num_type_check:
                self.key_mapping[str(key)]["Type"] = "time_series"
            elif not _num_type_check:
                self.key_mapping[str(key)]["Type"] = "descriptors"

            name_types_compiled = {
                category: re.compile("|".join(patterns), re.IGNORECASE)
                for category, patterns in self.name_types_regex.items()
            }
            # check for Group
            self.key_mapping[str(key)]["Group"] = next(
                (
                    group
                    for group, pattern in name_types_compiled.items()
                    if pattern.search(key)
                ),
                None,
            )

            _type = self.key_mapping[str(key)]["Type"]
            if _type is None:
                continue
            # check for Contains given the key type
            self.key_mapping[str(key)]["Contains"] = next(
                (
                    subtype
                    for subtype, regexes in self.name_types_regex[_type].items()
                    if any(re.search(r, key, re.IGNORECASE) for r in regexes)
                ),
                None,
            )

    def _picker(self, data_type: str, key: str) -> HBox:
        key_group = (
            self.key_mapping[key]["Group"]
            if self.key_mapping[key]["Group"] is not None
            else "other"
        )
        key_type = (
            self.key_mapping[key]["Contains"]
            if self.key_mapping[key]["Contains"] is not None
            else ("other" if data_type != "time_series" else "y_raw")
        )

        import_checker = Checkbox(value=key_group != "other", description=key)
        name_input = Text(value=key, description="Import as:")
        data_group_selector = Dropdown(
            options=self.data_groups.keys(), description="in group:", value=key_group
        )
        type_picker = Dropdown(
            options=self.name_types_regex[data_type].keys(),
            description="containing:",
            value=key_type,
        )
        return HBox([import_checker, name_input, data_group_selector, type_picker])

    def _create_pickers(self) -> VBox:
        pickers = {}
        for key, properties in self.key_mapping.items():
            data_type = (
                properties["Type"] if properties["Type"] is not None else "other"
            )
            pickers[key] = self._picker(str(data_type), str(key))
        return VBox(list(pickers.values()))

    def _import_data(self, _button: Button) -> None:
        # build a dataframe with the state of the pickers
        picker_state = pd.DataFrame(
            [
                [row_child.children[0].description]
                + [child.value for child in row_child.children]
                for row_child in self.picker.children
            ],
            columns=["key", "import", "name", "group", "type"],
        )

        self._import_group("EMG", picker_state)
        self._import_group("ventilator", picker_state)
        self._import_group("other", picker_state)

    def _import_group(self, group: str, picker_state: pd.DataFrame) -> None:
        # first, get the required arguments: the y_data
        _y_raw: np.ndarray | None = None
        _y_raw_keys = picker_state[
            (picker_state["import"])
            & (picker_state["group"] == group)
            & (picker_state["type"] == "y_raw")
        ]["key"].tolist()
        # now, build the _y_raw 2D array, by concatenating the selected keys along the second axis (channels)
        for key in _y_raw_keys:
            _y_temp = self.mat_dict[key]
            _y_temp = _y_temp if _y_temp.shape[0] > _y_temp.shape[1] else _y_temp.T
            _y_raw = (
                _y_temp if _y_raw is None else np.concatenate((_y_raw, _y_temp), axis=1)
            )
        if _y_raw is None:
            msg = f"No y_raw data selected for {group} group. Skipping import for this group."
            logger.warning(msg)
            return
        # then, optional arguments: t_data, labels, y_units, fs, n_samp, n_channels
        # regarding the labels: the labels are for the CHANNELS, not the time series.
        # Ex: we may have multiple 2D time series ("EMG_raw" and "EMG_filtered"),
        # while the channels may be "diaphragm" and "intercostal".
        # Only ONE raw time series is imported for each group; the others are imported as Other.
        optionals = {}
        for arg in ["t_data", "labels", "y_units", "fs", "n_samp", "n_channels"]:
            _key = picker_state[
                (picker_state["import"])
                & (picker_state["group"] == group)
                & (picker_state["type"] == arg)
            ]["key"].tolist()
            optionals[arg] = self.mat_dict[_key[0]].squeeze() if len(_key) > 0 else None
        # Check that there's at least one argument of t_data or fs. if only one is present, generate the other one.
        # If both are missing, raise an error.
        optionals["n_samp"] = (
            optionals["n_samp"]
            if optionals["n_samp"] is not None
            else max(_y_raw.shape)
        )
        optionals["n_channels"] = (
            optionals["n_channels"]
            if optionals["n_channels"] is not None
            else min(_y_raw.shape)
        )
        if optionals["t_data"] is None and optionals["fs"] is None:
            raise ValueError(
                "At least one of t_data or fs must be provided for " + group + " data."
            )
        if optionals["t_data"] is None:
            optionals["t_data"] = np.arange(optionals["n_samp"]) / optionals["fs"]
        elif optionals["fs"] is None:
            optionals["fs"] = 1 / np.mean(np.diff(optionals["t_data"]))
        # now check the existance of labels. If not provided, generate them
        optionals["labels"] = (
            (
                _y_raw_keys
                if _y_raw_keys is not None
                else [f"{group}_channel_{i}" for i in range(optionals["n_channels"])]
            )
            if optionals["labels"] is None
            else optionals["labels"]
        )

        if group == "EMG":
            self.emg_data = EmgDataGroup(
                y_raw=_y_raw,
                t_data=optionals["t_data"],
                labels=optionals["labels"],
                units=optionals["y_units"],
                fs=optionals["fs"],
            )
        elif group == "ventilator":
            self.ventilator_data = VentilatorDataGroup(
                y_raw=_y_raw,
                t_data=optionals["t_data"],
                labels=optionals["labels"],
                units=optionals["y_units"],
                fs=optionals["fs"],
            )
        elif group == "other":
            self.other_data = TimeSeriesGroup(
                y_raw=_y_raw,
                t_data=optionals["t_data"],
                labels=optionals["labels"],
                units=optionals["y_units"],
                fs=optionals["fs"],
            )

    def get_groups(self) -> tuple[EmgDataGroup, VentilatorDataGroup, TimeSeriesGroup]:
        """Return the imported data groups."""
        return self.emg_data, self.ventilator_data, self.other_data

    def get_emg_data(self) -> EmgDataGroup:
        """Return the imported EMG data group."""
        return self.emg_data

    def get_ventilator_data(self) -> VentilatorDataGroup:
        """Return the imported ventilator data group."""
        return self.ventilator_data

    def get_other_data(self) -> TimeSeriesGroup:
        """Return the imported other data group."""
        return self.other_data
