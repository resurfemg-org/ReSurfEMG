"""This file contains Jupyter widgets to perform default procedures.

NB The functions in this file required the development installation including
Jupyter (see README.md)

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

from typing import Literal, cast, overload

import ipywidgets as widgets
import numpy as np
import pandas as pd


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
    type_mapping: dict[str, type] = {"value": str, "idx": int}
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
            if value is not None and not isinstance(value, type_mapping[default_type]):
                msg_0 = f"default_{default_type}_select values need to be {type_mapping[default_type].__name__} or None"
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

    :param files: file paths tabled by the folder_levels
    :type files: pd.DataFrame
    :param folder_levels: data directory organisation, e.g. ['patient', 'date']
    :type folder_levels: list(str)
    :param default_value_select: default values to select per folder_level
    :type default_value_select: list(str)
    :param default_idx_select: default index to select per folder_level
    :type default_idx_select: list(int)

    :returns button_list: file paths tabled by the folder_levels
    :rtype button_list: [ipywidgets.widgets.widget_selection.Dropdown]
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
