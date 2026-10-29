"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains Jupyter widgets to perform default procedures.
NB The functions in this file required the development installation including
Jupyter (see README.md)
"""

import ipywidgets as widgets
import numpy as np
import pandas as pd


def file_select(
    files, folder_levels, default_value_select=None, default_idx_select=None
):
    """A widget for file selection for organised/nested data. default_value_select
    precedes default_idx_select in default value identification.
    ---------------------------------------------------------------------------
    :param files: file paths tabled by the folder_levels
    :type files: pd.DataFrame
    :param folder_levels: data directory organisation, e.g. ['patient', 'date']
    :type folder_levels: list(str)
    :param default_value_select: default values to select per folder_level
    :type default_value_select: list(int)
    :param default_idx_select: default index to select per folder_level
    :type default_idx_select: list(int)

    :returns button_list: file paths tabled by the folder_levels
    :rtype button_list: [ipywidgets.widgets.widget_selection.Dropdown]
    """
    if not isinstance(files, pd.core.frame.DataFrame):
        msg = "Files not provided in valid format."
        raise ValueError(msg)

    if not isinstance(folder_levels, list):
        msg = "Provide either a list as folder_levels."
        raise TypeError(msg)

    if default_value_select is None:
        default_value_select = len(folder_levels) * [None]
        value_options_bool = len(folder_levels) * [False]
    elif isinstance(default_value_select, list):
        if len(default_value_select) < len(folder_levels):
            msg = "len(default_value_select) < len(folder_levels)"
            raise IndexError(msg)
        value_options_bool = list()
        for value in default_value_select:
            if value is None:
                value_options_bool.append(False)
            elif isinstance(value, str):
                value_options_bool.append(True)
            else:
                msg = "default_value_select values need to be str"
                raise TypeError(msg)

    if default_idx_select is None:
        default_idx_select = len(folder_levels) * [None]
        idx_options_bool = len(folder_levels) * [False]
    elif isinstance(default_idx_select, list):
        if len(default_idx_select) < len(folder_levels):
            msg = "len(default_idx_select) < len(folder_levels)"
            raise IndexError(msg)
        idx_options_bool = list()
        for idx in default_idx_select:
            if isinstance(idx, int):
                idx_options_bool.append(True)
            elif idx is None:
                idx_options_bool.append(False)
            else:
                msg = "default_idx_select values need to be int"
                raise TypeError(msg)

    button_list = list()
    btn_dict = dict()
    for _, folder_level in enumerate(folder_levels):
        _btn = widgets.Dropdown(
            description=folder_level + ":",
            disabled=False,
        )

        button_list.append(_btn)
        btn_dict[folder_level] = _btn
    prev_values = len(folder_levels) * [None]

    @widgets.interact(**btn_dict)
    def update_select(**btn_dict):
        """Update the dropdown options based on the previous selection."""
        btn_changed = []
        for _idx in range(len(folder_levels)):
            btn_changed.append(button_list[_idx].value != prev_values[_idx])

        for idx, dict_key in enumerate(btn_dict):
            btn_idx = folder_levels.index(dict_key)
            _btn = button_list[btn_idx]

            if idx == 0:
                filter_files = files
            else:
                bool_list = list()
                for _idx in range(idx):
                    bool_list.append(
                        (
                            files[folder_levels[_idx]]
                            == button_list[_idx].value
                        ).values
                    )

                filter_files = files[np.all(np.array(bool_list), 0)]

            options = list(np.unique(filter_files[dict_key].values))
            options.sort()

            if len(options) > 0:
                if any(btn_changed[:btn_idx]) or prev_values[btn_idx] is None:
                    if value_options_bool[btn_idx] is True:
                        if default_value_select[btn_idx] in options:
                            value = options[
                                options.index(default_value_select[btn_idx])
                            ]
                        else:
                            value = options[0]
                    elif idx_options_bool[btn_idx] is True:
                        if default_idx_select[btn_idx] < len(options):
                            value = options[default_idx_select[btn_idx]]
                        else:
                            value = options[0]
                    elif _btn.value in options:
                        value = options[options.index(_btn.value)]
                    else:
                        value = options[0]
                elif _btn.value in options:
                    value = options[options.index(_btn.value)]
                else:
                    value = options[0]
            else:
                value = None

            _btn.options = options
            if len(options) > 0:
                _btn.value = value
                prev_values[btn_idx] = value

    return button_list
