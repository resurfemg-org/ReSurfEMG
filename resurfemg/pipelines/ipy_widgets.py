"""This file contains Jupyter widgets to perform default procedures.

NB The functions in this file required the development installation including
Jupyter (see README.md)

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import ClassVar, Literal, cast, overload

import anywidget
import ipywidgets as widgets
import numpy as np
import pandas as pd
import traitlets
from IPython.display import display
from ipywidgets import Button, Checkbox, Dropdown, HBox, Text, VBox
from scipy import io as sio

from resurfemg.data_connector.adicht_reader import AdichtReader
from resurfemg.data_connector.data_classes import EmgDataGroup, TimeSeriesGroup, VentilatorDataGroup
from resurfemg.data_connector.file_discovery import filepaths_dict, find_files
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader

logger = logging.getLogger(__name__)


def _wait_for_change(widget: widgets.Widget, value: str) -> asyncio.Future:
    future = asyncio.Future()

    def getvalue(change) -> None:  # noqa: ANN001
        # make the new value available
        future.set_result(change.new)
        widget.unobserve(getvalue, value)

    widget.observe(getvalue, value)
    return future


class CheckBoxTree(anywidget.AnyWidget):
    """Create a tree of checkboxes for hierarchical data selection."""

    _esm = """
    function render({ model, el }) {
        let currentBranch = null;
        let firstBranchPath = null;

        function isLeafParent(data) {
            return typeof data === 'object' && data !== null && !Array.isArray(data)
                && Object.keys(data).length > 0
                && Object.values(data).every(v => typeof v === 'string');
        }

        function getBranchPath(filePath) {
            const idx = filePath.lastIndexOf('/');
            return idx >= 0 ? filePath.substring(0, idx) : '';
        }

        function getFilesInBranch(branchPath) {
            return [...el.querySelectorAll('input.file-cb')]
                .filter(cb => getBranchPath(cb.dataset.path) === branchPath);
        }

        function getBranchCb(branchPath) {
            return [...el.querySelectorAll('input.branch-cb')]
                .find(cb => cb.dataset.branch === branchPath) ?? null;
        }

        function updateBranchCb(branchPath) {
            const bcb = getBranchCb(branchPath);
            if (!bcb) return;
            const files = getFilesInBranch(branchPath);
            const n = files.filter(f => f.checked).length;
            bcb.indeterminate = n > 0 && n < files.length;
            bcb.checked = n > 0;
        }

        function clearOtherBranches(keepBranch) {
            el.querySelectorAll('input.branch-cb').forEach(bcb => {
                if (bcb.dataset.branch !== keepBranch) {
                    getFilesInBranch(bcb.dataset.branch).forEach(f => { f.checked = false; });
                    bcb.checked = false;
                    bcb.indeterminate = false;
                }
            });
        }

        function buildFileRow(fname, dtype, fullPath, initiallyChecked) {
            const row = document.createElement('div');
            row.className = 'file-row';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.className = 'file-cb';
            cb.checked = initiallyChecked;
            cb.dataset.path = fullPath;
            cb.addEventListener('change', () => {
                const branch = getBranchPath(fullPath);
                if (cb.checked) {
                    if (currentBranch !== null && branch !== currentBranch) {
                        clearOtherBranches(branch);
                    }
                    currentBranch = branch;
                } else {
                    if (getFilesInBranch(branch).every(f => !f.checked)) currentBranch = null;
                }
                updateBranchCb(branch);
                syncChecked();
            });
            const label = document.createElement('label');
            label.textContent = fname;
            const sel = document.createElement('select');
            sel.className = 'type-sel';
            sel.dataset.path = fullPath;
            for (const opt of ['EMG', 'Ventilator', 'Other', 'Both']) {
                const o = document.createElement('option');
                o.value = opt;
                o.textContent = opt;
                if (opt === dtype) o.selected = true;
                sel.appendChild(o);
            }
            sel.addEventListener('change', syncTypes);
            row.appendChild(cb);
            row.appendChild(label);
            row.appendChild(sel);
            return row;
        }

        function buildLeafParent(key, data, pathParts, isFirst) {
            const branchPath = [...pathParts, key].join('/');
            const details = document.createElement('details');
            details.open = true;
            const summary = document.createElement('summary');
            summary.className = 'branch-summary';

            const bcb = document.createElement('input');
            bcb.type = 'checkbox';
            bcb.className = 'branch-cb';
            bcb.dataset.branch = branchPath;
            bcb.checked = isFirst;
            bcb.addEventListener('click', e => e.stopPropagation());
            bcb.addEventListener('change', () => {
                if (bcb.checked) {
                    clearOtherBranches(branchPath);
                    getFilesInBranch(branchPath).forEach(f => { f.checked = true; });
                    currentBranch = branchPath;
                    bcb.indeterminate = false;
                    details.open = true;
                } else {
                    getFilesInBranch(branchPath).forEach(f => { f.checked = false; });
                    currentBranch = null;
                }
                syncChecked();
            });

            summary.appendChild(bcb);
            summary.appendChild(Object.assign(document.createElement('span'), { textContent: ' ' + key }));
            details.appendChild(summary);
            for (const [fname, dtype] of Object.entries(data)) {
                details.appendChild(buildFileRow(fname, dtype, branchPath + '/' + fname, isFirst));
            }
            return details;
        }

        function buildTree(data, pathParts) {
            const container = document.createElement('div');
            for (const [key, value] of Object.entries(data)) {
                const currentPath = [...pathParts, key];
                if (typeof value === 'string') {
                    const branch = pathParts.join('/');
                    if (firstBranchPath === null) firstBranchPath = branch;
                    container.appendChild(buildFileRow(key, value, currentPath.join('/'), firstBranchPath === branch));
                } else if (isLeafParent(value)) {
                    const branchPath = currentPath.join('/');
                    if (firstBranchPath === null) firstBranchPath = branchPath;
                    container.appendChild(buildLeafParent(key, value, pathParts, firstBranchPath === branchPath));
                } else {
                    const details = document.createElement('details');
                    details.open = true;
                    details.appendChild(Object.assign(document.createElement('summary'), { textContent: key }));
                    details.appendChild(buildTree(value, currentPath));
                    container.appendChild(details);
                }
            }
            return container;
        }

        function syncChecked() {
            model.set('checked_files', [...el.querySelectorAll('input.file-cb:checked')].map(cb => cb.dataset.path));
            model.save_changes();
        }

        function syncTypes() {
            const types = {};
            el.querySelectorAll('select.type-sel').forEach(sel => { types[sel.dataset.path] = sel.value; });
            model.set('file_types', types);
            model.save_changes();
        }

        function rebuild() {
            currentBranch = null;
            firstBranchPath = null;
            el.innerHTML = '';
            el.appendChild(buildTree(model.get('tree_data'), []));
            const firstChecked = el.querySelector('input.branch-cb:checked');
            if (firstChecked) {
                currentBranch = firstChecked.dataset.branch;
                el.querySelectorAll('details').forEach(d => { d.open = false; });
                let node = firstChecked.parentElement;
                while (node && node !== el) {
                    if (node.tagName === 'DETAILS') node.open = true;
                    node = node.parentElement;
                }
            } else if (firstBranchPath !== null) {
                currentBranch = firstBranchPath;
            }
            syncChecked();
            syncTypes();
        }

        model.on('change:tree_data', rebuild);
        rebuild();
    }
    export default { render };
    """

    _css = """
    details { margin: 4px 0; padding-left: 12px; }
    summary { cursor: pointer; font-weight: bold; }
    .branch-summary { display: flex; align-items: center; gap: 4px; list-style: none; }
    .branch-summary::-webkit-details-marker { display: none; }
    .branch-summary::before { content: '▶'; font-size: 0.65em; transition: transform 0.15s; }
    details[open] > .branch-summary::before { transform: rotate(90deg); }
    .branch-cb { cursor: pointer; flex-shrink: 0; }
    .file-row { display: flex; gap: 8px; padding: 2px 0 2px 16px; align-items: center; }
    """

    tree_data = traitlets.Dict({}).tag(sync=True)
    checked_files = traitlets.List([]).tag(sync=True)
    file_types = traitlets.Dict({}).tag(sync=True)


class PatientSelector:
    """A widget to select patients and their corresponding files.

    Description: This class provides a user interface for selecting patients and their associated files.
    It allows users to choose a patient from a dropdown menu and then select specific files related to that patient
    using a tree of checkboxes. The selected files can be imported for further analysis.
    """

    data_groups: ClassVar[dict[str, tuple[str, ...]]] = {
        "EMG": ("emg", "exg"),
        "Ventilator": ("vent", "flow", "pressure", "rip", "draeger"),
        "Other": (),
        "Both": ("all", "both"),
    }

    default_labels: ClassVar[dict[str, tuple[str, ...]]] = {
        "EMG": ("ECG", "EMGdi", "EMGin"),
        "Ventilator": ("Flow", "Pressure", "RIP"),
        "Other": (),
        "Both": (),
    }

    default_units: ClassVar[dict[str, tuple[str, ...]]] = {
        "EMG": ("mV", "uV", "uV"),
        "Ventilator": ("L/min", "cmH2O", "mL"),
        "Other": (),
        "Both": (),
    }

    supported_extensions: ClassVar[tuple[str, ...]] = (".Poly5", ".adicht", ".mat")

    def __init__(self, root_directory: str | Path | None = None, patient_regex: str = r"^([Pp]_?\d+)"):
        self.root_directory = root_directory if root_directory is not None else Path.cwd()
        self._trees: dict[str, CheckBoxTree] = {}
        self._complete_regex(patient_regex)
        self._compiled_name_types = {
            category: re.compile("|".join(patterns), re.IGNORECASE) for category, patterns in self.data_groups.items()
        }
        self.widget = self._create_widget()
        self.data_emg: EmgDataGroup
        self.data_vent: VentilatorDataGroup
        self.data_other: TimeSeriesGroup

    def _complete_regex(self, regex: str) -> None:
        """Complete the regex pattern to match the supported file extensions.

        Args:
            regex (str): The regex pattern to complete.
        """
        self.patient_regex = (
            "("
            + regex
            + r")[/\\](?:.+[/\\])?[^/\\]+\.(?:"
            + "|".join(ext.lstrip(".") for ext in self.supported_extensions)
            + ")$"
        )

    def _dropdown_changed(self, _dropdown: widgets.Dropdown) -> None:
        """Update the file selection checkboxes when the patient selection dropdown changes.

        Args:
            dropdown (widgets.Dropdown): The patient selection dropdown widget.

        Returns:
            None
        """
        self._get_selected()

    def _build_tree_data(self, node: dict) -> dict:
        """Convert patient_dict subtree (leaves=lists) to CheckBoxTree tree_data (leaves=dtype strings).

        filepaths_dict produces three kinds of values:
          - empty list  → the key itself is a file (depth-2 path)
          - non-empty list → the key is a folder; items are file names
          - dict        → the key is a folder; recurse
        """
        result = {}
        for k, v in node.items():
            if isinstance(v, list):
                if not v:
                    result[k] = self._guess_data_type(k)
                else:
                    result[k] = {f: self._guess_data_type(f) for f in v}
            else:
                result[k] = self._build_tree_data(v)
        return result

    def _get_selected(self) -> None:
        """Read checked files and their data types from the currently visible tree."""
        self.selected_id = self.patient_selector.value
        if self.selected_id is None:
            return
        _tree = self._trees[self.selected_id]
        self.selected_files = _tree.checked_files
        self.selected_types = [_tree.file_types[t] for t in _tree.checked_files]

    def _create_widget(self) -> None:
        """Build the patient dropdown, per-patient CheckBoxTree stack, and import button."""
        self.files = find_files(self.root_directory)
        col = self.files["files"]
        col = [Path(c).parts for c in col if re.search(self.patient_regex, c)]
        self.patient_dict = filepaths_dict(col)
        ids = list(self.patient_dict.keys())

        self.patient_selector = widgets.Dropdown(options=ids)

        self._trees = {_id: CheckBoxTree(tree_data=self._build_tree_data(self.patient_dict[_id])) for _id in ids}

        self.file_selector = widgets.Stack(
            [self._trees[_id] for _id in ids],
            selected_index=0,
        )
        widgets.jslink((self.patient_selector, "index"), (self.file_selector, "selected_index"))

        self.import_button = widgets.Button(
            description="Import selected files",
            button_style="success",
            layout=widgets.Layout(width="auto", margin="5px"),
        )
        self.import_button.on_click(lambda b: asyncio.ensure_future(self._import_selected_files(b)))
        display(widgets.VBox([widgets.HBox([self.patient_selector, self.import_button]), self.file_selector]))

    async def _import_selected_files(self, _button: widgets.Button) -> None:
        """Import the selected files for the selected patient.

        Args:
            button (widgets.Button): The import button widget.
        """
        self._get_selected()
        _data = None
        _id = str(self.selected_id)
        _selected = {_id: dict(zip(self.selected_files, self.selected_types, strict=False))}
        for _file, _type in list(_selected[_id].items()):
            file = str(self.root_directory / Path(_id) / Path(_file))
            _extension = Path(file).suffix
            logger.info(_extension)
            if _extension in [".Poly5", ".adicht"]:
                _data = self._get_nonmat_data(
                    Poly5Reader(file) if _extension == ".Poly5" else AdichtReader(file), _type
                )
                if _type == "EMG" and isinstance(_data, EmgDataGroup):
                    self.data_emg = _data
                elif _type == "Ventilator" and isinstance(_data, VentilatorDataGroup):
                    self.data_vent = _data
                elif _type == "Other" and isinstance(_data, TimeSeriesGroup):
                    self.data_other = _data
            elif _extension == ".mat":
                _picker = CustomizeMatlabImport(file)
                _data = await self._import_matlab_file(_picker)
                for i, _ata_type in enumerate(["EMG", "Ventilator", "Other"]):
                    if _data[i] is not None:
                        setattr(self, f"data_{_ata_type.lower()}", _data[i])

            logger.info("data imported from %s file!", file)

            # now, we have the container, which is different for each extension type.
            # .mat already has the TimeSeriesGroup objects, while the other two still need converting.

    def _get_nonmat_data(
        self, data: Poly5Reader | AdichtReader, data_type: str
    ) -> EmgDataGroup | VentilatorDataGroup | TimeSeriesGroup:
        # Load the EMG and ventilator data recordings from the selected folders.
        if not hasattr(data, "samples") or not hasattr(data, "sample_rate"):
            msg = "The provided data does not contain the required attributes 'samples' and 'sample_rate'."
            raise ValueError(msg)
        if isinstance(data, Poly5Reader):
            y = data.samples[: data.num_samples] if hasattr(data, "num_samples") else data.samples[:]
            fs = data.sample_rate if hasattr(data, "sample_rate") else None
            n_channels = y.shape[0]
            labels = cast(
                "list[str]",
                (
                    data.ch_names[:n_channels]
                    if hasattr(data, "channel_labels")
                    else (self.default_labels[data_type][:n_channels])
                ),
            )
            units = data.ch_unit_names[:n_channels] if hasattr(data, "channel_units") else n_channels * ["uV"]
        elif isinstance(data, AdichtReader):
            # Extract the ventilator data
            select_channel_idxs = [*range(3)]
            record_idx = 0
            resample_channels_dict = None
            data_df, fs = data.extract_data(
                channel_idxs=select_channel_idxs,
                record_idx=record_idx,
                resample_channels=resample_channels_dict,
            )
            # Get the labels and units of the selected channels
            y = data_df.to_numpy().T
            labels = data.get_labels(select_channel_idxs)
            units = data.get_units(select_channel_idxs, record_idx)
            # NB: The units in the example data are in mV, so overwrite them:
            labels = ["Paw", "Flow", "Volume"]
            units = ["cmH2O", "L/min", "mL"]
        if data_type == "EMG":
            return EmgDataGroup(y, fs=fs, labels=labels, units=units)
        if data_type == "Ventilator":
            return VentilatorDataGroup(y, fs=fs, labels=labels, units=units)
        return TimeSeriesGroup(y, fs=fs, labels=labels, units=units)

    async def _wait_for_matlab_import_button(self, _picker: CustomizeMatlabImport) -> None:
        await _wait_for_change(_picker.import_button, "value")

    async def _import_matlab_file(
        self, _picker: CustomizeMatlabImport
    ) -> tuple[EmgDataGroup | None, VentilatorDataGroup | None, TimeSeriesGroup | None]:
        await self._wait_for_matlab_import_button(_picker)
        _picker.import_data(_picker.import_button)
        logger.info("data imported from .mat file!")
        return _picker.get_groups()

    def _update_data_type(self, _dropdown: widgets.Dropdown) -> None:
        self._get_selected()
        self.patient_dict[self.selected_id][_dropdown.tooltip.replace("Data type for ", "")] = _dropdown.value

    def _create_checkbox(self, description: str, _type: str) -> widgets.HBox:
        """Create a checkbox widget and a dropdown widget for selecting the data type.

        Args:
            _type (str): The default value for the dropdown

            description (str): The description for the checkbox.

        Returns:
            widgets.HBox: A horizontal box containing the checkbox and dropdown widgets.
        """
        _checkbox = widgets.Checkbox(
            description=description,
            disabled=False,
            value=True,
            layout=widgets.Layout(margin="5px", width="auto"),
            style={"description_width": "initial"},
        )

        _widget = widgets.Dropdown(
            options=["EMG", "Ventilator", "Other", "Both"],
            value=_type,
            on_trait_change=self._update_data_type,
            tooltip=f"Data type for {description}",
            layout=widgets.Layout(width="auto"),
        )

        return widgets.HBox([_checkbox, _widget])

    def _guess_data_type(self, file_name: str | Path) -> str:
        """Guess the data type of a file based on its extension.

        Args:
            file_name (str | Path): The name of the file.

        Returns:
            str: The guessed data type ("EMG", "Ventilator", or "Other").
        """
        _type = next(
            (cat for cat, pattern in self._compiled_name_types.items() if pattern.search(str(file_name))), None
        )
        return _type if _type is not None else "Other"


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
            if value is not None and not isinstance(value, name_types_regex[default_type]):
                msg_0 = (
                    f"default_{default_type}_select values need to be {name_types_regex[default_type].__name__} or None"
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
    default_idx_select, idx_options_bool = _check_defaults(default_idx_select, folder_levels, default_type="idx")

    button_list = []
    btn_dict = {}
    for _, folder_level in enumerate(folder_levels):
        _btn = widgets.Dropdown(
            description=folder_level + ":",
            disabled=False,
        )
        button_list.append(_btn)
        btn_dict[folder_level] = _btn

    prev_values: list[str | None] = cast("list[str | None]", [None] * len(folder_levels))

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
    btn_changed = [button_list[_idx].value != prev_values[_idx] for _idx in range(len(folder_levels))]

    for idx, dict_key in enumerate(btn_dict):
        btn_idx = folder_levels.index(dict_key)
        _btn = button_list[btn_idx]

        if idx == 0:
            filter_files = files
        else:
            bool_list = [button_list[_idx].value == files[folder_levels[_idx]].values for _idx in range(idx)]
            filter_files = files[np.all(np.array(bool_list), 0)]

        col_values: np.ndarray = filter_files[dict_key].to_numpy()
        options = list(np.unique(col_values))
        options.sort()

        _btn.options = options
        if options:
            value = _btn.value if _btn.value in options else options[0]
            if any(btn_changed[:btn_idx]) or prev_values[btn_idx] is None:
                if value_options_bool[btn_idx] and default_value_select is not None:
                    value = default_value_select[btn_idx] if default_value_select[btn_idx] in options else options[0]
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
        self.key_mapping: dict[str, dict[Literal["Type", "Group", "Contains"] | None, str | None]] = {}
        self._compiled_groups = {
            category: re.compile("|".join(patterns), re.IGNORECASE) for category, patterns in self.data_groups.items()
        }
        self._compiled_name_types = {
            category: re.compile("|".join(patterns), re.IGNORECASE)
            for category, patterns in self.name_types_regex.items()
        }
        self._map_keys()
        self.emg_data: EmgDataGroup
        self.ventilator_data: VentilatorDataGroup
        self.other_data: TimeSeriesGroup
        self.picker = self._create_pickers()
        display(self.picker)
        self.import_button = Button(description="Import selected data")
        self.import_button.on_click(self.import_data)
        display(self.import_button)

    def _map_keys(self) -> None:
        """Map the keys in the .mat file to their type, group and content based on regex matching."""
        for key in self.keys:
            self.key_mapping[str(key)] = {"Type": None, "Group": None, "Contains": None}
            _num_type_check = isinstance(self.mat_dict[key], np.ndarray) and np.issubdtype(
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

            self.key_mapping[str(key)]["Group"] = next(
                (group for group, pattern in self._compiled_groups.items() if pattern.search(key)),
                None,
            )

            _type = self.key_mapping[str(key)]["Type"]
            if _type is None:
                continue
            # check for Contains given the key type
            _match = next(
                (pattern.search(key) for _, pattern in self._compiled_name_types.items() if pattern.search(key)),
                None,
            )

            self.key_mapping[str(key)]["Contains"] = _match.group(0) if _match is not None else None

    def _picker(self, data_type: str, key: str) -> HBox:
        """Create a picker for a given key in the .mat file.

        Args:
            data_type (str): The type of the key (time_series, descriptors, params).
            key (str): The key for which to create a picker.
        """
        # key_group: EMG, ventilator, other.
        key_group = self.key_mapping[key]["Group"] if self.key_mapping[key]["Group"] is not None else "other"
        # key_type:
        key_type = (
            self.key_mapping[key]["Contains"]
            if self.key_mapping[key]["Contains"] is not None
            else ("other" if data_type != "time_series" else "y_raw")
        )

        # Select whether to import the key or not. Default is True for EMG and ventilator groups, False for other group.
        import_checker = Checkbox(value=key_group != "other", description=key)
        # Name for the imported TimeSeriesGroup
        name_input = Text(value=key, description="Import as:")
        # data_group_selector options: EMG (EMGDataGroup) or Ventilator (VentilatorDataGroup) or Other (TimeSeriesGroup)
        data_group_selector = Dropdown(options=self.data_groups.keys(), description="in group:", value=key_group)
        # type_picker options: if data_type is time_series: t_data, y_raw, other;
        # if data_type is descriptors: labels, y_units, other; if data_type is params: fs, n_samp, n_channels, other
        type_picker = Dropdown(
            options=self.name_types_regex[data_type].keys(),
            description="containing:",
            value=key_type,
        )
        return HBox([import_checker, name_input, data_group_selector, type_picker])

    def _create_pickers(self) -> VBox:
        pickers = {}
        for key, properties in self.key_mapping.items():
            data_type = properties["Type"] if properties["Type"] is not None else "other"
            pickers[key] = self._picker(str(data_type), str(key))
        return VBox(list(pickers.values()))

    def import_data(self, _button: Button) -> None:
        """Import the selected data from the .mat file.

        Returns:EmgDataGroup, VentilatorDataGroup and TimeSeriesGroup objects.
        """
        # build a dataframe with the state of the pickers
        picker_state = pd.DataFrame(
            [
                [row_child.children[0].description] + [child.value for child in row_child.children]
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
            (picker_state["import"]) & (picker_state["group"] == group) & (picker_state["type"] == "y_raw")
        ]["key"].tolist()
        # now, build the _y_raw 2D array, by concatenating the selected keys along the second axis (channels)
        for key in _y_raw_keys:
            _y_temp = self.mat_dict[key]
            _y_temp = _y_temp if _y_temp.shape[0] > _y_temp.shape[1] else _y_temp.T
            _y_raw = _y_temp if _y_raw is None else np.concatenate((_y_raw, _y_temp), axis=1)
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
                (picker_state["import"]) & (picker_state["group"] == group) & (picker_state["type"] == arg)
            ]["key"].tolist()
            optionals[arg] = self.mat_dict[_key[0]].squeeze() if len(_key) > 0 else None
        # Check that there's at least one argument of t_data or fs. if only one is present, generate the other one.
        # If both are missing, raise an error.
        optionals["n_samp"] = optionals["n_samp"] if optionals["n_samp"] is not None else max(_y_raw.shape)
        optionals["n_channels"] = optionals["n_channels"] if optionals["n_channels"] is not None else min(_y_raw.shape)
        if optionals["t_data"] is None and optionals["fs"] is None:
            raise ValueError("At least one of t_data or fs must be provided for " + group + " data.")
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

    def get_groups(self) -> tuple[EmgDataGroup | None, VentilatorDataGroup | None, TimeSeriesGroup | None]:
        """Return the imported data groups."""
        return self.get_emg_data(), self.get_ventilator_data(), self.get_other_data()

    def get_emg_data(self) -> EmgDataGroup | None:
        """Return the imported EMG data group."""
        return self.emg_data if hasattr(self, "emg_data") else None

    def get_ventilator_data(self) -> VentilatorDataGroup | None:
        """Return the imported ventilator data group."""
        return self.ventilator_data if hasattr(self, "ventilator_data") else None

    def get_other_data(self) -> TimeSeriesGroup | None:
        """Return the imported other data group."""
        return self.other_data if hasattr(self, "other_data") else None
