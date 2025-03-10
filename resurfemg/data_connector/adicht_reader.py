"""The class AdichtReader is designed to load EMG data from an ADInstruments
device using the .adicht file format (Labchart) and prepares it for use in
ReSurfEMG. The foundation of the AdichtReader class is the repository
'adinstruments_sdk_python' by Jim Hokanson, available at:
https://github.com/JimHokanson/adinstruments_sdk_python

An example of how to use this class is provided in the main block of this file.
This example executes only if the script is run directly by the Python
interpreter and not when imported as a module.
"""
from typing import Tuple

import numpy as np
import pandas as pd
import adi
from prettytable import PrettyTable

class AdichtReader:
    """
    This class is designed to load EMG data from an ADInstruments device using
    the .adicht file format (Labchart) and prepare it for use in ReSurfEMG.
    The foundation of the AdichtReader class is the repository
    'adinstruments_sdk_python' by Jim Hokanson, available at:
    https://github.com/JimHokanson/adinstruments_sdk_python
    """
    def __init__(self, file_path: str):
        """
        Initializes the AdichtReader class with the path to a file.

        :param file_path: The path to the file in the proprietary Adicht format
        """
        self.file_path = file_path
        self.metadata = None
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
        import os
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
            print(f"Error loading the file: {e}")
            raise

    def _initialize_channel_map(self):
        """
        Creates a dictionary mapping the channel names to their IDs.
        """
        try:
            self.channel_map = {
                channel.name: channel.id
                for channel in self.adicht_data.channels}
        except Exception as e:
            print(f"Error initializing the channel map: {e}")
            raise

    def _initialize_record_map(self):
        """
        Creates a dictionary mapping the channel names to their IDs.
        """
        try:
            self.record_map = {
                i: record.id
                for i, record in enumerate(self.adicht_data.records)}
        except Exception as e:
            print(f"Error initializing the record map: {e}")
            raise

    def __repr__(self):
        return f"<AdichtReader(file_path={self.file_path})>"


    def print_metadata(self):
        """
        Extracts and provides a tabular overview of the channels, samples,
        records, sampling rates, units, and time step.

        :return: Dictionary with metadata (Channels, Sampling Rates, etc.)
        """
        try:
            table = PrettyTable()
            table.field_names = [
                "Channel ID", "Name", "Records", "Samples", 
                "Sampling Rate (Hz)", "timestep (s)", "Units"]
            table.align["Name"] = "l"

            channel_info = []
            for channel in self.adicht_data.channels:
                info = {
                    "id": channel.id,
                    "name": channel.name,
                    "records": channel.n_records,
                    "samples": channel.n_samples,
                    "sampling_rate": channel.fs,
                    "time_step": channel.dt,
                    "units": channel.units,
                }
                channel_info.append(info)
                table.add_row([
                    channel.id,
                    channel.name,
                    channel.n_records,
                    ", ".join(map(str, channel.n_samples)),
                    ", ".join(map(str, channel.fs)),
                    ", ".join(map(str, channel.dt)),
                    channel.units
                ])
            self.metadata = channel_info
            print("Available Channels and Metadata:")
            print(table)
            return channel_info
        except Exception as e:
            print(f"Error reading the file: {e}")
            raise

    def get_labels(self, channel_idxs: list[str])-> list[str]:
        """
        Returns a list of channel names based on a list of channel IDs.

        :param channel_idxs: List of channel indices.
        :return: List of channel names.
        """
        try:
            labels = [
                self.adicht_data.channels[idx].name for idx in channel_idxs]
            return labels
        except Exception as e:
            print(f"Error getting labels: {e}")
            raise

    def get_units(self, channel_idxs: list[str], record_idx: int)-> list[str]:
        """
        Returns a list of units based on a list of channel IDs and a record ID.

        :param channel_idxs: List of channel indices.
        :param record_idx: The record index to retrieve the units for.
        :return: List of units.
        """
        try:
            units = [
                self.adicht_data.channels[idx].units[record_idx]
                for idx in channel_idxs]
            return units
        except Exception as e:
            print(f"Error getting units: {e}")
            raise

    def resample_channel(
            self, channel_idx: int, record_idx: int, target_rate: int):
        """
        Performs resampling for a specific channel using a linear interpolation
        method and adds an additional row to the resampled DataFrame.
        
        :param channel_idx: The ID of the channel to be resampled.
        :param record_idx: The ID of the record to extract the data from.
        :param target_rate: The target sampling rate in Hz.
        :return: A Pandas DataFrame with resampled data for the channel.
        """
        try:
            # Extract data for the specific channel
            if channel_idx < 1 or channel_idx > len(self.adicht_data.channels):
                raise ValueError(f"Channel ID '{channel_idx}' is invalid.")
            
            # Load channel data
            _channel = self.adicht_data.channels[channel_idx]
            data = _channel.get_data(record_idx)
            channel_name = _channel.name
            
            # Retrieve sampling details: Original time step
            time_step = _channel.dt[record_idx]
            current_rate = 1 / time_step  # Current sampling rate
            if target_rate == current_rate:
                raise UserWarning("target_rate equals current_rate")

            # Create DataFrame and set time index
            df = pd.DataFrame({channel_name: data})
            df.index = pd.to_timedelta(df.index * time_step, unit='s')
            
            # New interval based on target rate
            new_interval = 1 / target_rate 
            new_interval_timedelta = pd.to_timedelta(new_interval, unit='s')

            original_sampling_rate = _channel.fs[record_idx]
            num_of_samples = _channel.n_samples[record_idx]
            target_num_samples = int(
                num_of_samples * (target_rate / original_sampling_rate))

            # Create an empty DataFrame with target sample rate
            timedelta_index = pd.to_timedelta(
                np.arange(target_num_samples) * new_interval_timedelta.value)
            empty_df = pd.DataFrame(
                index=timedelta_index, columns=[channel_name])
            empty_df[channel_name] = np.nan

            # Merge DataFrames
            df_combined = empty_df.combine_first(df)
            df_combined = df_combined.interpolate(method='linear')
            df_resampled = df_combined.resample(
                new_interval_timedelta).interpolate(method='linear')

            return df_resampled
        except Exception as e:
            print(f"Error resampling channel {channel_idx}: {e}")
            raise

    

    def extract_data(
        self,
        channel_idxs: list,
        record_idx: int,
        resample_channels: dict = None) -> Tuple[pd.DataFrame, int]:
        """
        Extracts specific data based on a list of channel IDs and a record ID.
        Optionally, certain channels can be resampled to a new sampling rate.

        Limitation: This function is designed to equalise deviating sampling
        rates between, different channels for example, ECG and EMG data. It is
        not supported to sample all specified channel_ids to a different rate.
        There must be at least one channel_id that already has the target rate
        and is not specified in the resampling. The first of these channels
        presents the leading sampling rate.

        :param channel_ids: List of channel IDs whose data are to be extracted.
        :param record_id: The record ID for which the data are to be extracted.
        :param resample_channels: A dictionary mapping channel IDs (as keys) to
            target sampling rates (in Hz) as values.
            Example: {1: 2000, 3: 2000} - Resample ch 1 and ch 3 to 2000 Hz

        :return: A tuple containing:
            - A pandas DataFrame with the extracted and resampled data.
            - The sampling rate (in Hz) of the leading channel.
        """
        try:
            data_dict = {}
            non_resampled_channels = []
            for idx in channel_idxs:
                if (idx < 0 or idx >= len(self.adicht_data.channels)):
                    raise ValueError(f"Channel ID '{idx}' is invalid.")
                
                if resample_channels and idx in resample_channels:
                    target_rate = resample_channels[idx]
                    resampled_df = self.resample_channel(
                        idx, self.record_map[record_idx], target_rate)
                    for column in resampled_df.columns:
                        data_dict[column] = resampled_df[column].values
                else:
                    np_data = self.adicht_data.channels[idx].get_data(
                        self.record_map[record_idx])
                    channel_name = self.adicht_data.channels[idx].name
                    data_dict[channel_name] = np_data
                    non_resampled_channels.append(idx)

            df = pd.DataFrame(data_dict)
            # Select an unsampled channel to read out target sampling
            leader_channel = self.adicht_data.channels[
                non_resampled_channels[0]]
            time_step = leader_channel.dt[record_idx]
            df.index = pd.to_timedelta(df.index * time_step, unit='s')

            return df, int(leader_channel.fs[record_idx])

        except Exception as e:
            print(f"Error extracting the data: {e}")
            raise
