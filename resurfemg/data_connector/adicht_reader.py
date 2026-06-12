"""Load EMG data from ADInstrument devices.

The class AdichtReader is designed to load EMG data from an ADInstruments
device using the .adicht file format (Labchart) and prepares it for use in
ReSurfEMG. The foundation of the AdichtReader class is the repository
"adinstruments_sdk_python" by Jim Hokanson, available at:
https://github.com/JimHokanson/adinstruments_sdk_python

An example of how to use this class is provided in the main block of this file.
This example executes only if the script is run directly by the Python
interpreter and not when imported as a module.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from prettytable import PrettyTable

from resurfemg.helper_functions.math_operations import get_dict_key_where_value

if platform.system() == "Windows":
    import adi


class AdichtReader:
    """Load EMG data from ADInstrument devices.

    Class for loading timeseries data from an ADInstruments devices using
    the .adicht/.adidat/.adibin file formats (LabChart, BIOPAC) and prepare it
    for use in ReSurfEMG.
    Based on the "adinstruments_sdk_python" repository by Jim Hokanson,
    available at: https://github.com/JimHokanson/adinstruments_sdk_python
    """

    def __init__(self, file_path: str):
        """Initialize the AdichtReader with the provided file path.

        Args:
            file_path (str): The file path to the import.
        """
        if platform.system() != "Windows":
            msg = "AdichtReader is only available on Windows."
            raise ImportError(msg)
        self.file_path = file_path
        self.metadata: list[dict] = []
        self.metadata_table = None
        self.channel_map: dict[int, int] = {}  # Dictionary mapping channel names to IDs
        self.adicht_data: Any = None  # Reader object for the file
        self.record_map: dict[int, int] = {}  # Dictionary mapping record idx to IDs

        self._validate_file_path()
        self._initialize_reader()
        self._initialize_channel_map()
        self._initialize_record_map()

    def _validate_file_path(self) -> None:
        """Validate the provided file path.

        Validates whether the provided file path exists and is readable.
        """
        if not Path(self.file_path).exists():
            msg = f"The file '{self.file_path}' was not found."
            raise FileNotFoundError(msg)
        if not Path(self.file_path).is_file():
            msg = f"The path '{self.file_path}' does not refer to a file."
            raise ValueError(msg)

    def _initialize_reader(self) -> None:
        """Initialize the ADInstruments reader.

        Initializes the adi-reader and loads the file.
        """
        try:
            self.adicht_data = adi.read_file(  # pyright: ignore[reportPossiblyUnboundVariable]
                self.file_path
            )
        except Exception as e:
            msg = f"Error loading the file: {e}"
            raise RuntimeError(msg) from e

    def _initialize_channel_map(self) -> None:
        """Map channel names to their IDs.

        Creates a dictionary mapping the channel names to their IDs.
        """
        self.channel_map = {i: channel.id for i, channel in enumerate(self.adicht_data.channels)}

    def _initialize_record_map(self) -> None:
        """Map record indices to their IDs.

        Creates a dictionary mapping the record indices to their IDs.
        """
        self.record_map = {i: record.id for i, record in enumerate(self.adicht_data.records)}

    def __repr__(self):
        return f"<AdichtReader(file_path={self.file_path})>"

    def generate_metadata(self) -> list[dict]:
        """Extract channel metadata.

        Extracts metadata on channels, samples, records, sampling rates, units,
        and time step and sets it in self.metadata and self.metadata_table.

        Returns:
            list[dict]: List of metadata dicts per channel.
        """
        table = PrettyTable()
        table.field_names = [
            "idx",
            "Channel ID",
            "Name",
            "Records",
            "Samples",
            "Sampling Rate (Hz)",
            "timestep (s)",
            "Units",
        ]
        table.align["Name"] = "l"

        channel_info = []
        for idx, channel in enumerate(self.adicht_data.channels):
            info = {
                "idx": idx,
                "id": channel.id,
                "name": channel.name,
                "records": channel.n_records,
                "samples": channel.n_samples,
                "fs": channel.fs,
                "time_step": channel.dt,
                "units": channel.units,
            }
            channel_info.append(info)
            table.add_row(
                [
                    idx,
                    channel.id,
                    channel.name,
                    channel.n_records,
                    ", ".join(map(str, channel.n_samples)),
                    ", ".join(map(str, channel.fs)),
                    ", ".join(map(str, channel.dt)),
                    channel.units,
                ]
            )
        self.metadata = channel_info
        self.metadata_table = table
        return channel_info

    def print_metadata(self) -> None:
        """Print channel metadata.

        Extracts and provides a tabular overview of the channels, samples,
        records, sampling rates, units, and time step.
        """
        self.generate_metadata()
        print(f"Available channels and metadata:\n{self.metadata_table}")  # noqa: T201

    def _resolve_one(self, idx: int | None, id_: int | None, mapping: dict[int, int], name: str) -> int:
        if idx is not None:
            return idx
        if id_ is None:
            msg = f"Either {name}_idx or {name}_id must be set."
            raise ValueError(msg)
        resolved = get_dict_key_where_value(mapping, id_)
        if resolved is None:
            msg = f"{name} id {id_} not found."
            raise ValueError(msg)
        return resolved

    def _resolve_many(
        self,
        idxs: list[int] | None,
        ids: list[int] | None,
        mapping: dict[int, int],
        name: str,
    ) -> list[int]:
        if idxs is not None:
            return idxs
        if ids is None:
            msg = f"Either {name}_idxs or {name}_ids must be set."
            raise ValueError(msg)
        return [self._resolve_one(None, i, mapping, name) for i in ids]

    def get_labels(self, channel_idxs: list[int] | None = None, channel_ids: list | None = None) -> list[str]:
        """Return channel names based on channel indices or IDs.

        Args:
            channel_idxs (list[int], optional): List of channel indices.
            channel_ids (list, optional): List of channel IDs. Either
                channel_idxs or channel_ids must be set.

        Returns:
            list[str]: List of channel names.
        """
        channel_idxs = self._resolve_many(channel_idxs, channel_ids, self.channel_map, "channel")
        return [self.adicht_data.channels[idx].name for idx in channel_idxs]

    def get_units(
        self,
        channel_idxs: list[int] | None = None,
        record_idx: int | None = None,
        channel_ids: list | None = None,
        record_id: int | None = None,
    ) -> list[str]:
        """Return channel units based on channel indices and a record index or ID.

        Args:
            channel_idxs (list[int], optional): List of channel indices. Either
                channel_idxs or channel_ids must be set.
            channel_ids (list, optional): List of channel IDs. Either
                channel_idxs or channel_ids must be set.
            record_idx (int, optional): The record index to retrieve the units
                for. Either record_idx or record_id must be set.
            record_id (int, optional): The record ID to retrieve the units for.
                Either record_idx or record_id must be set.

        Returns:
            list[str]: List of units.
        """
        channel_idxs = self._resolve_many(channel_idxs, channel_ids, self.channel_map, "channel")
        record_idx = self._resolve_one(record_idx, record_id, self.record_map, "record")
        return [self.adicht_data.channels[idx].units[record_idx] for idx in channel_idxs]

    def resample_channel(
        self,
        fs_target: int,
        channel_idx: int | None = None,
        record_idx: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        """Resample a channel to a target sampling rate.

        Resamples the specified channel using linear interpolation.

        Args:
            fs_target (int): The target sampling rate in Hz.
            channel_idx (int, optional): The channel index to be resampled.
            record_idx (int, optional): The record index to be resampled.
                Either record_idx or record_id must be set.
            **kwargs: Additional arguments to specify channel_id or record_id
                instead of indices.

        Returns:
            pd.DataFrame: Record DataFrame with resampled data for the
                specified index.
        """
        channel_idx = self._resolve_one(channel_idx, kwargs.get("channel_id"), self.channel_map, "channel")
        record_idx = self._resolve_one(record_idx, kwargs.get("record_id"), self.record_map, "record")

        _channel = self.adicht_data.channels[channel_idx]
        if fs_target == 1 / _channel.dt[record_idx]:
            msg = "target_rate equals current_rate"
            raise UserWarning(msg)

        # Create DataFrame and set time index
        df = pd.DataFrame({_channel.name: _channel.get_data(self.record_map[record_idx])})
        df.index = pd.to_timedelta(df.index * _channel.dt[record_idx], unit="s")

        # New interval based on target rate
        dt_target_timedelta = pd.to_timedelta(1 / fs_target, unit="s")

        fs_original = _channel.fs[record_idx]
        n_samples_target = int(_channel.n_samples[record_idx] * (fs_target / fs_original))

        # Create an empty DataFrame with target sample rate
        timedelta_index = pd.to_timedelta(np.arange(n_samples_target) * dt_target_timedelta.value)
        empty_df = pd.DataFrame(index=timedelta_index, columns=[_channel.name])
        empty_df[_channel.name] = np.nan

        # Merge DataFrames
        df_combined = empty_df.combine_first(df)
        df_combined = df_combined.interpolate(method="linear")
        return df_combined.resample(dt_target_timedelta).interpolate(method="linear")

    def extract_data(
        self,
        channel_idxs: list[int] | None = None,
        record_idx: int | None = None,
        resample_channels: dict[int, int] | None = None,
        **kwargs,
    ) -> tuple[pd.DataFrame, int]:
        """Extract channel data from specified channels and record.

        Optionally resamples specified channels to equalize sampling rates
        across channels. Resampling all channels to a rate not yet used is not
        supported; at least one channel must already have the target rate and
        must not be listed in resample_channels.

        Args:
            channel_idxs (list[int], optional): List of channel indices.
            record_idx (int, optional): The record index to extract data from.
            resample_channels (dict[int, int], optional): Map of
                channel_idx to target rate. Example: ``{1: 2000, 3: 2000}``
                resamples channels 1 and 3 to 2000 Hz.
            **kwargs: Additional arguments to specify channel_ids or record_id
                instead of indices.

        Returns:
            tuple:
                - pd.DataFrame: Extracted (and optionally resampled) data.
                - int: Sampling rate (Hz) of the leading channel.
        """
        channel_idxs = self._resolve_many(channel_idxs, kwargs.get("channel_ids"), self.channel_map, "channel")
        record_idx = self._resolve_one(record_idx, kwargs.get("record_id"), self.record_map, "record")

        fs_out = []
        data_dict = {}
        non_resampled_channels = []
        for idx in channel_idxs:
            if idx not in self.channel_map:
                msg = f"Channel idx '{idx}' is invalid."
                raise ValueError(msg)

            if resample_channels and idx in resample_channels:
                resampled_df = self.resample_channel(resample_channels[idx], idx, record_idx)

                for column in resampled_df.columns:
                    data_dict[column] = resampled_df[column].values
                fs_out.append(resample_channels[idx])
            else:
                np_data = self.adicht_data.channels[idx].get_data(self.record_map[record_idx])
                channel_name = self.adicht_data.channels[idx].name
                data_dict[channel_name] = np_data
                non_resampled_channels.append(idx)
                fs_out.append(self.metadata[idx]["fs"][0])

        if len(set(fs_out)) > 1:
            msg = "Output channels have different sampling rates."
            raise ValueError(msg)
        df = pd.DataFrame(data_dict)
        # Select an unsampled channel to read out target sampling
        leader_channel = self.adicht_data.channels[non_resampled_channels[0]]
        df.index = pd.to_timedelta(df.index * leader_channel.dt[record_idx], unit="s")

        return df, int(leader_channel.fs[record_idx])
