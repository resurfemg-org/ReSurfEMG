# The class AdichtReader is designed to load EMG data from an ADInstruments device using the .adicht file format (Labchart) 
# and prepares it for use in ReSurfEMG. The foundation of the AdichtReader class is the repository
# 'adinstruments_sdk_python' by Jim Hokanson, available at:
# https://github.com/JimHokanson/adinstruments_sdk_python
#
# An example of how to use this class is provided in the main block of this file. This example executes only if the script 
# is run directly by the Python interpreter and not when imported as a module.

import adi                              # adi-reader==0.0.13        adinstruments_sdk_python by Jim Hokanson, https://github.com/JimHokanson/adinstruments_sdk_python
import numpy as np                      # numpy==2.1.2
import pandas as pd                     # pandas==2.2.3
from prettytable import PrettyTable     # prettytable==3.12.0

class AdichtReader:
    """
    This class is designed to load EMG data from an ADInstruments device using the .adicht file format (Labchart)
    and prepare it for use in ReSurfEMG. The foundation of the AdichtReader class is the repository
    'adinstruments_sdk_python' by Jim Hokanson, available at:
    https://github.com/JimHokanson/adinstruments_sdk_python
    """
    def __init__(self, file_path: str):
        """
        Initializes the AdichtReader class with the path to a file.

        :param file_path: The path to the file in the proprietary Adicht format.
        """
        self.file_path = file_path
        self.metadata = None
        self.channel_map = None         # Dictionary mapping channel names to IDs
        self.adicht_data = None         # Reader object for the file

        self._validate_file_path()
        self._initialize_reader()
        self._initialize_channel_map()

    def _validate_file_path(self):
        """
        Validates whether the provided file path exists and is readable.
        """
        import os
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"The file '{self.file_path}' was not found.")
        if not os.path.isfile(self.file_path):
            raise ValueError(f"The path '{self.file_path}' does not refer to a file.")

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
            self.channel_map = {channel.name: channel.id for channel in self.adicht_data.channels}
        except Exception as e:
            print(f"Error initializing the channel map: {e}")
            raise

    def __repr__(self):
        return f"<AdichtReader(file_path={self.file_path})>"


    def print_metadata(self):
        """
        Extracts and provides a tabular overview of the channels, samples, records,
        sampling rates, units, and time step.

        :return: Dictionary with metadata (Channels, Sampling Rates, etc.)
        """
        try:
            table = PrettyTable()
            table.field_names = ["Channel ID", "Name", "Records", "Samples", "Sampling Rate (Hz)", "Zeitschritt (s)", "Units"]
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

    def get_labels(self, channel_ids: list[str])-> list[str]:
        """
        Returns a list of channel names based on a list of channel IDs.

        :param channel_ids: List of channel IDs.
        :return: List of channel names.
        """
        try:
            labels = [self.adicht_data.channels[channel_id - 1].name for channel_id in channel_ids]
            return labels
        except Exception as e:
            print(f"Error getting labels: {e}")
            raise

    def get_units(self, channel_ids: list[str], record_id: int)-> list[str]:
        """
        Returns a list of units based on a list of channel IDs and a record ID.

        :param channel_ids: List of channel IDs.
        :param record_id: The record ID for which the units are to be retrieved.
        :return: List of units.
        """
        try:
            units = [self.adicht_data.channels[channel_id - 1].units[record_id-1] for channel_id in channel_ids]
            return units
        except Exception as e:
            print(f"Error getting units: {e}")
            raise

    def resample_channel(self, channel_id: int, record_id: int, target_rate: int):
        """
        Performs resampling for a specific channel using a linear interpolation method and adds an additional row to the resampled DataFrame.
        
        :param channel_id: The ID of the channel to be resampled.
        :param record_id: The ID of the record from which data is to be extracted.
        :param target_rate: The target sampling rate in Hz.
        :return: A Pandas DataFrame with resampled data for the channel.
        """
        try:
            # Extract data for the specific channel
            if channel_id < 1 or channel_id > len(self.adicht_data.channels):
                raise ValueError(f"Channel ID '{channel_id}' is invalid.")
            
            # Retrieve sampling details
            time_step = self.adicht_data.channels[channel_id - 1].dt[record_id-1]  # Original time step  # convert id to index
            current_rate = 1 / time_step  # Current sampling rate

            if target_rate == current_rate:
                raise Exception("target_rate = current_rate")
            
            # Load channel data
            data = self.adicht_data.channels[channel_id - 1].get_data(record_id)
            channel_name = self.adicht_data.channels[channel_id - 1].name
            
            # Create DataFrame and set time index
            df = pd.DataFrame({channel_name: data})
            df.index = pd.to_timedelta(df.index * time_step, unit='s')
            
            # New interval based on target rate
            new_interval = 1 / target_rate 
            new_interval_timedelta = pd.to_timedelta(new_interval, unit='s')

            original_sampling_rate = self.adicht_data.channels[channel_id - 1].fs[record_id-1]
            num_of_samples = self.adicht_data.channels[channel_id - 1].n_samples[record_id-1]
            target_num_samples = int(num_of_samples * (target_rate / original_sampling_rate))

            # Create an empty DataFrame with target sample rate
            timedelta_index = pd.to_timedelta(np.arange(target_num_samples) * new_interval_timedelta.value)
            empty_df = pd.DataFrame(index=timedelta_index, columns=[channel_name])  # create new empty DataFrame with target sampling rate
            empty_df[channel_name] = np.nan

            # Merge DataFrames
            df_combined = empty_df.combine_first(df)
            df_combined = df_combined.interpolate(method='linear')  # Interpolate missing values
            df_resampled = df_combined.resample(new_interval_timedelta).interpolate(method='linear')

            return df_resampled


        except Exception as e:
            print(f"Error resampling channel {channel_id}: {e}")
            raise

    from typing import Tuple

    def extract_data(self, channel_ids: list, record_id: int, resample_channels: dict = None) -> Tuple[pd.DataFrame, int]:
        """
        Extracts specific data based on a list of channel IDs and a record ID.
        Optionally, certain channels can be resampled to a new sampling rate.

        Limitation: This function is designed to equalise deviating sampling rates between, different channels for example, 
        ECG and EMG data. It is not supported to sample all specified channel_ids to a different rate. There must be at least 
        one channel_id that already has the target rate and is not specified in the resampling. The first of these 
        channels presents the leading sampling rate.

        :param channel_ids: List of channel IDs whose data are to be extracted.
        :param record_id: The record ID for which the data are to be extracted.
        :param resample_channels: A dictionary mapping channel IDs (as keys) to target sampling rates (in Hz) as values.
                                Example: {1: 2000, 3: 2000} - Resample channel 1 to 2000 Hz and channel 3 to 2000 Hz.

        :return: A tuple containing:
             - A pandas DataFrame with the extracted and optionally resampled data.
             - The sampling rate (in Hz) of the leading channel.
        """
        try:
            data_dict = {}
            non_resampled_channels = []
            for channel_id in channel_ids:
                if channel_id < 1 or channel_id > len(self.adicht_data.channels):
                    raise ValueError(f"Channel ID '{channel_id}' is invalid.")
                
                if resample_channels and channel_id in resample_channels:
                    target_rate = resample_channels[channel_id]
                    resampled_df = self.resample_channel(channel_id, record_id, target_rate)
                    for column in resampled_df.columns:
                        data_dict[column] = resampled_df[column].values
                else:
                    np_data = self.adicht_data.channels[channel_id - 1].get_data(record_id)
                    channel_name = self.adicht_data.channels[channel_id - 1].name
                    data_dict[channel_name] = np_data
                    non_resampled_channels.append(channel_id)

            df = pd.DataFrame(data_dict)
            
            leader_channel = self.adicht_data.channels[non_resampled_channels[0] - 1]       # Select an unsampled channel to read out target sampling
            time_step = leader_channel.dt[record_id-1]
            df.index = pd.to_timedelta(df.index * time_step, unit='s')

            return df, int(leader_channel.fs[record_id-1])

        except Exception as e:
            print(f"Error extracting the data: {e}")
            raise


# Example Usage
if __name__ == "__main__":
    emg_file_chosen = "C:/Example/Path/filename.adicht"
    adicht_data = AdichtReader(emg_file_chosen)
    adicht_data.print_metadata()    # Extracts and provides a tabular overview of the channels, samples, records, sampling rates, units, and time step.

    selected_channel_ids = [1,7,8]
    record_id = 2
    resample_channels_dict={        # In this example, Channel1 has a different sampling rate compared to the other channels, so it will be resampled to match their rate
        1: 2000,                          
    }

    # get the sekeceted data and the sample rate of the leading channel
    data_emg_df, fs_emg = adicht_data.extract_data(channel_ids= selected_channel_ids,record_id= record_id,resample_channels= resample_channels_dict)

    # you can also use the resample method only to resample a single channel for specific cases
    channel1_resampeld_df = adicht_data.resample_channel(channel_id= 1,record_id= 1,target_rate= 2000)     

    # Select a specific time range from the DataFrame
    start_time = pd.Timedelta('2 minutes 20 seconds')  # safety margin of 10seconds
    end_time = pd.Timedelta('2 minutes 30 seconds')
    selected_data_df = data_emg_df[(data_emg_df.index >= start_time) & (data_emg_df.index <= end_time)]

    # to pass the data for the EmgDataGroup constructor, the pandas dataframe must be converted into a numpy array and transformed 
    y_emg   = selected_data_df.to_numpy().T

    # get the labels and units of the selected channels
    label_list  = adicht_data.get_labels(selected_channel_ids)
    unit_list   = adicht_data.get_units(selected_channel_ids, record_id)