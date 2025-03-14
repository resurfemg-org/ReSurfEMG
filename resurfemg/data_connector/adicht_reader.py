"""The class AdichtReader is designed to load EMG data from an ADInstruments
device using the .adicht file format (Labchart) and prepares it for use in
ReSurfEMG. The foundation of the AdichtReader class is the repository
'adinstruments_sdk_python' by Jim Hokanson, available at:
https://github.com/JimHokanson/adinstruments_sdk_python

An example of how to use this class is provided in the main block of this file.
This example executes only if the script is run directly by the Python
interpreter and not when imported as a module.
"""
import os
import platform
import numpy as np
import pandas as pd
from prettytable import PrettyTable
from resurfemg.helper_functions.math_operations import get_dict_key_where_value
if platform.system() == 'Windows':
    import adi


class AdichtReader:
    """
    Class for loading timeseries data from an ADInstruments devices using
    the .adicht/.adidat/.adibin file formats (LabChart, BIOPAC) and prepare it
    for use in ReSurfEMG.
    Based on the 'adinstruments_sdk_python' repository by Jim Hokanson,
    available at: https://github.com/JimHokanson/adinstruments_sdk_python
    """
    def __init__(self, file_path: str):
        """
        :param file_path: The file path to the import
        """
        if platform.system() != 'Windows':
            raise ImportError("AdichtReader is only available on Windows.")
        self.file_path = file_path
        self.metadata = None
        self.metadata_table = None
        self.channel_map = None     # Dictionary mapping channel names to IDs
        self.adicht_data = None     # Reader object for the file
        self.record_map = None      # Dictionary mapping record idx to IDs

        self._validate_file_path()
        self._initialize_reader()
        self._initialize_channel_map()
        self._initialize_record_map()

    def _validate_file_path(self):
        """
        Validates whether the provided file path exists and is readable.
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(
                f"The file '{self.file_path}' was not found.")
        if not os.path.isfile(self.file_path):
            raise ValueError(
                f"The path '{self.file_path}' does not refer to a file.")

    def _initialize_reader(self):
        """
        Initializes the adi-reader and loads the file.
        """
        try:
            self.adicht_data = adi.read_file(self.file_path)
        except Exception as e:
            raise RuntimeError(f"Error loading the file: {e}") from e

    def _initialize_channel_map(self):
        """
        Creates a dictionary mapping the channel names to their IDs.
        """
        self.channel_map = {
            i: channel.id
            for i, channel in enumerate(self.adicht_data.channels)}

    def _initialize_record_map(self):
        """
        Creates a dictionary mapping the channel names to their idxs.
        """
        self.record_map = {
            i: record.id
            for i, record in enumerate(self.adicht_data.records)}

    def __repr__(self):
        return f"<AdichtReader(file_path={self.file_path})>"

    def generate_metadata(self):
        """
        Extracts metadata on channels, samples, records, sampling rates, units,
        and time step and sets it in self.metadata and self.metadata_table.

        :return: List of metadata dicts per channel
        :rtype: list[dict]
        """
        table = PrettyTable()
        table.field_names = [
            "idx", "Channel ID", "Name", "Records", "Samples",
            "Sampling Rate (Hz)", "timestep (s)", "Units"]
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
            table.add_row([
                idx,
                channel.id,
                channel.name,
                channel.n_records,
                ", ".join(map(str, channel.n_samples)),
                ", ".join(map(str, channel.fs)),
                ", ".join(map(str, channel.dt)),
                channel.units
            ])
        self.metadata = channel_info
        self.metadata_table = table
        return channel_info

    def print_metadata(self):
        """
        Extracts and provides a tabular overview of the channels, samples,
        records, sampling rates, units, and time step.

        :return: List of metadata dicts per channel
        :rtype: list[dict]
        """
        _ = self.generate_metadata()
        print("Available channels and metadata:")
        print(self.metadata_table)

    def get_labels(self, channel_idxs=None, channel_ids=None):
        """
        Returns a list of channel names based on a list of channel indices.
        -----------------------------------------------------------------------
        :param channel_idxs: List of channel indices
        :type channel_idxs: list[int]
        :param channel_ids: List of channel IDs to retrieve the labels for
        :type channel_ids: list

        :return: List of channel names.
        :rtype: list[str]
        """
        if channel_ids is not None and channel_idxs is None:
            channel_idxs = [
                get_dict_key_where_value(self.channel_map, channel_id)
                for channel_id in channel_ids]
        elif channel_ids is None and channel_idxs is None:
            raise ValueError("Either channel_idxs or channel_ids must be set.")
        labels = [
            self.adicht_data.channels[idx].name for idx in channel_idxs]
        return labels

    def get_units(self, channel_idxs=None, record_idx=None, channel_ids=None,
                  record_id=None):
        """
        Returns a list of units based on a list of channel indices and a record
        id.
        -----------------------------------------------------------------------
        :param channel_idxs: List of channel indices
        :type channel_idxs: list[int]
        :param channel_ids: List of channel IDs to retrieve the labels for
        :type channel_ids: list
        Either channel_idxs or channel_ids must be set.
        :param record_idx: The record index to retrieve the units for
        :type record_idx: int
        :param record_id: The record ID to retrieve the units for
        :type record_id: int
        Either record_idx or record_id must be set.
        :return: List of units
        :rtype: list[str]
        """
        if channel_ids is not None and channel_idxs is None:
            channel_idxs = [
                get_dict_key_where_value(self.channel_map, channel_id)
                for channel_id in channel_ids]
        elif channel_ids is not None and channel_idxs is None:
            raise ValueError("Either channel_idxs or channel_ids must be set.")
        if record_idx is None and record_id is not None:
            record_idx = get_dict_key_where_value(self.record_map, record_id)
        elif record_idx is None and record_id is None:
            raise ValueError("Either record_idx or record_id must be set.")

        units = [
            self.adicht_data.channels[idx].units[record_idx]
            for idx in channel_idxs]
        return units

    def resample_channel(
            self, fs_target, channel_idx=None, record_idx=None, **kwargs):
        """
        Resample the specified channel using a linear interpolation
        method and adds an additional row to the resampled DataFrame.
        -----------------------------------------------------------------------
        :param channel_idx: The channel index to be resampled.
        :type channel_idx: int
        :param record_idx: The record index to be resampled.
        :type record_idx: int
        Either record_idx or record_id must be set.
        :param fs_target: The target sampling rate in Hz.
        :type fs_target: int
        :param kwargs: Additional arguments to specify the channel ID or record
        ID instead of indices.

        :return: Record DataFrame with resampled data for the specified idx.
        :rtype: pd.DataFrame
        """
        if kwargs.get('channel_id') is not None and channel_idx is None:
            channel_idx = get_dict_key_where_value(
                self.channel_map, kwargs.get('channel_id'))
        elif kwargs.get('channel_id') is None and channel_idx is None:
            raise ValueError("Either channel_idx or channel_id must be set.")
        if record_idx is None and kwargs.get('record_id') is not None:
            record_idx = get_dict_key_where_value(
                self.record_map, kwargs.get('record_id'))
        elif record_idx is None and kwargs.get('record_id') is None:
            raise ValueError("Either record_idx or record_id must be set.")

        # Extract data for the specific channel
        if channel_idx not in self.channel_map:
            raise ValueError(f"Channel ID '{channel_idx}' is invalid.")

        # Load channel data and Retrieve sampling details: Original time step
        _channel = self.adicht_data.channels[channel_idx]
        if fs_target == 1 / _channel.dt[record_idx]:
            raise UserWarning("target_rate equals current_rate")

        # Create DataFrame and set time index
        df = pd.DataFrame({_channel.name: _channel.get_data(
            self.record_map[record_idx])})
        df.index = pd.to_timedelta(
            df.index * _channel.dt[record_idx], unit='s')

        # New interval based on target rate
        dt_target_timedelta = pd.to_timedelta(1 / fs_target, unit='s')

        fs_original = _channel.fs[record_idx]
        n_samples_target = int(
            _channel.n_samples[record_idx] * (fs_target / fs_original))

        # Create an empty DataFrame with target sample rate
        timedelta_index = pd.to_timedelta(
            np.arange(n_samples_target) * dt_target_timedelta.value)
        empty_df = pd.DataFrame(
            index=timedelta_index, columns=[_channel.name])
        empty_df[_channel.name] = np.nan

        # Merge DataFrames
        df_combined = empty_df.combine_first(df)
        df_combined = df_combined.interpolate(method='linear')
        df_resampled = df_combined.resample(
            dt_target_timedelta).interpolate(method='linear')

        return df_resampled

    def extract_data(self, channel_idxs=None, record_idx=None,
                     resample_channels=None, **kwargs):
        """
        Extract channel data from specified channels and record. Optionally,
        resample specified channels to equalize sampling rates across channels.
        Resampling all channels to a different, not yet used, rate is not
        supported. There must be at least one channel that already has the
        target rate and is not specified in the resampling.
        -----------------------------------------------------------------------
        :param channel_idxs: List of channel indices.
        :type channel_idxs: list[int]
        :param record_idx: The record index to extract the data from.
        :type record_idx: int
        :param resample_channels: Resample specified channels to a new rate.
        :type resample_channels: {channel_idx: target_rate}
        Example: {1: 2000, 3: 2000} - Resample ch 1 and ch 3 to 2000 Hz
        :param kwargs: Additional arguments to specify the channel IDs or
        record IDs instead of indices.
        :return: A tuple containing:
        - A pandas DataFrame with the extracted and resampled data.
        - The sampling rate (in Hz) of the leading channel.
        :rtype: Tuple[pd.DataFrame, int]
        """
        if kwargs.get('channel_ids') is not None and channel_idxs is None:
            channel_idxs = [
                get_dict_key_where_value(self.channel_map, channel_id)
                for channel_id in kwargs.get('channel_ids')]
        elif kwargs.get('channel_ids') is None and channel_idxs is None:
            raise ValueError("Either channel_idxs or channel_ids must be set.")
        if record_idx is None and kwargs.get('record_id') is not None:
            record_idx = get_dict_key_where_value(
                self.record_map, kwargs.get('record_id'))
        elif record_idx is None and kwargs.get('record_id') is None:
            raise ValueError("Either record_idx or record_id must be set.")

        fs_out = []
        data_dict = {}
        non_resampled_channels = []
        for idx in channel_idxs:
            if idx not in self.channel_map:
                raise ValueError(f"Channel idx '{idx}' is invalid.")

            if resample_channels and idx in resample_channels:
                resampled_df = self.resample_channel(
                    resample_channels[idx], idx, record_idx)

                for column in resampled_df.columns:
                    data_dict[column] = resampled_df[column].values
                fs_out.append(resample_channels[idx])
            else:
                np_data = self.adicht_data.channels[idx].get_data(
                    self.record_map[record_idx])
                channel_name = self.adicht_data.channels[idx].name
                data_dict[channel_name] = np_data
                non_resampled_channels.append(idx)
                fs_out.append(self.metadata[idx]['fs'][0])

        if len(set(fs_out)) > 1:
            raise ValueError("Output channels have different sampling rates.")

        df = pd.DataFrame(data_dict)
        # Select an unsampled channel to read out target sampling
        leader_channel = self.adicht_data.channels[
            non_resampled_channels[0]]
        df.index = pd.to_timedelta(
            df.index * leader_channel.dt[record_idx], unit='s')

        return df, int(leader_channel.fs[record_idx])
