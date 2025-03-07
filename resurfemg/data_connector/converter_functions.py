"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains standardized functions to work with various EMG file types
from various hardware/software combinations, and convert them down to
an array that can be further processed with other modules.
"""

import os
import platform
import glob
import pandas as pd
import numpy as np
import scipy.io as sio
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.data_connector.adicht_reader import AdichtReader


def load_file(
    file_path,
    key_name=None,
    channels=None,
    drop_channels=None,
    force_col_reading=False,
    verbose=True
):
    """
    This function loads a file from a given path and returns the data as a
    numpy array. The function can handle .poly5, .mat, .csv, and .npy files.
    The function can also rename channels and drop channels from the data.
    --------------------------------------------------------------------------
    :param file_path: Path to the file to be loaded
    :type file_path: str
    :param key_name: Key name for .mat files
    :type key_name: str
    :param channels: List of channel names to rename the loaded channels
    :type channels: list
    :param drop_channels: List of channel names or indices to drop
    :type drop_channels: list
    :param force_col_reading: Force column reading for row based .csv files
    :type force_col_reading: bool
    :param verbose: Print verbose output
    :type verbose: bool

    :returns np_float_data: Numpy array of the loaded data
    :rtype: ~numpy.ndarray
    :returns data_df: Pandas DataFrame of the loaded data
    :rtype: ~pandas.DataFrame
    :returns metadata: Metadata of the loaded data
    :rtype: dict
    """

    if not isinstance(file_path, str):
        raise ValueError('file_path should be a str.')

    if platform.system() == 'Windows':
        path_sep = "\\"
    else:
        path_sep = '/'

    file_name = file_path.split(path_sep)[-1]
    file_dir = os.path.join(*file_path.split(path_sep)[:-1])
    file_extension = file_name.split('.')[-1]

    # 1. Load File types: .poly5, .mat, .csv,
    metadata = dict()
    if file_extension.lower() == 'poly5':
        print('Detected .Poly5')
        data_df, metadata = load_poly5(file_path, verbose=verbose)
    elif file_extension.lower() == 'mat':
        print('Detected .mat')
        data_df = load_mat(file_path, key_name, verbose)
    elif file_extension.lower() == 'csv':
        print('Detected .csv')
        data_df, metadata = load_csv(
            file_path, force_col_reading, verbose)
    elif file_extension.lower() == 'npy':
        print('Detected .npy')
        data_df, metadata = load_npy(file_path, verbose)
    elif file_extension.lower() == 'adicht':
        print('Detected .adicht')
        data_df, metadata = load_adicht(
            file_path, record_id=0, channel_ids=None, verbose=verbose)
    else:
        raise UserWarning("No methods availabe for file extension"
                          + f"{file_extension}.")

    metadata['file_name'] = file_name
    metadata['file_dir'] = file_dir
    metadata['file_extension'] = file_extension
    if isinstance(channels, list) and len(channels) == data_df.shape[1]:
        if not all(isinstance(channel, str) for channel in channels):
            raise TypeError('All channel names should be str')
        if len(channels) != len(set(channels)):
            raise UserWarning('Channel names should be unique')
        if verbose:
            print('Renamed channels:', *zip(data_df.columns, channels), '...')
        data_df.columns = channels

    # 2. Drop channels
    if isinstance(drop_channels, list):
        if all(channel in data_df.columns for channel in drop_channels):
            data_df.drop(columns=drop_channels, inplace=True)
            metadata['dropped_channels'] = drop_channels
            if verbose:
                print('Dropped channels:', drop_channels)
        elif all((isinstance(channel, int) and channel < len(data_df.columns)
                  for channel in drop_channels)):
            data_df.drop(columns=data_df.columns[drop_channels], inplace=True)
            metadata['dropped_channels'] = data_df.columns[
                drop_channels].values
            if verbose:
                print('Dropped channels:',
                      metadata['dropped_channels'], '...')
        else:
            raise UserWarning('drop_channels should be a list of channel '
                              + 'indices (int) or names (str).')
    else:
        metadata['dropped_channels'] = []

    # 3. Convert remaining float channels to numpy array
    float_data_df = data_df.select_dtypes(include=float)
    np_float_data = np.rot90(float_data_df.to_numpy())
    metadata['float_channels'] = float_data_df.columns.values
    if verbose:
        print('Loaded channels as np.array:',
              metadata['float_channels'], '...')

    return np_float_data, data_df, metadata


def load_poly5(file_path, verbose=True):
    """
    This function loads a .Poly5 file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.
    --------------------------------------------------------------------------
    :param file_path: Path to the file to be loaded
    :type file_path: str
    :param verbose: Print verbose output
    :type verbose: bool

    :returns data_df: Pandas DataFrame of the loaded data
    :rtype data_df: ~pandas.DataFrame
    :returns metadata: Metadata of the loaded data
    :rtype metadata: dict
    """
    if verbose:
        print('Loading .Poly5 ...')
    poly5_data = Poly5Reader(file_path)
    if verbose:
        print('Loaded .Poly5, extracting data ...')
    n_samples = poly5_data.num_samples
    loaded_data = poly5_data.samples[:, :n_samples]
    metadata = dict()
    metadata['fs'] = poly5_data.sample_rate
    metadata['loaded_channels'] = poly5_data.ch_names
    metadata['units'] = poly5_data.ch_unit_names
    data_df = pd.DataFrame(loaded_data.T, columns=metadata['loaded_channels'])
    if verbose:
        print('Loading data completed')

    return data_df, metadata


def load_mat(file_path, key_name, verbose=True):
    """
    This function loads a .mat file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the sampling rate,
    loaded channels, and units.
    --------------------------------------------------------------------------
    :param file_path: Path to the file to be loaded
    :type file_path: str
    :param key_name: Key name for .mat files
    :type key_name: str
    :param verbose: Print verbose output
    :type verbose: bool

    :returns data_df: Pandas DataFrame of the loaded data
    :rtype data_df: ~pandas.DataFrame
    :returns metadata: Metadata of the loaded data
    :rtype metadata: dict
    """
    if verbose:
        print('Loading .mat ...')
    mat_dict = sio.loadmat(file_path, mdict=None, appendmat=False)
    if verbose:
        print('Loaded .mat, extracting data ...')
    if isinstance(key_name, str):
        loaded_data = mat_dict[key_name]
        if loaded_data.shape[0] > loaded_data.shape[1]:
            loaded_data = np.rot90(loaded_data)
            if verbose:
                print('Transposed loaded data.')
        data_df = pd.DataFrame(loaded_data.T)
        print('Loading data completed')
    else:
        raise ValueError('No key_name provided.')

    return data_df


def load_csv(file_path, force_col_reading, verbose=True):
    """
    This function loads a .csv file and returns the data as a pandas
    DataFrame. The function also returns metadata such as the loaded channels.
    --------------------------------------------------------------------------
    :param file_path: Path to the file to be loaded
    :type file_path: str
    :param force_col_reading: Force column reading for row based .csv files
    :type force_col_reading: bool
    :param verbose: Print verbose output
    :type verbose: bool

    :returns data_df: Pandas DataFrame of the loaded data
    :rtype data_df: ~pandas.DataFrame
    :returns metadata: Metadata of the loaded data
    :rtype metadata: dict
    """
    def has_header(file_path, nrows=20):
        df = pd.read_csv(file_path, header=None, nrows=nrows)
        df_header = pd.read_csv(file_path, nrows=nrows)
        return tuple(df.dtypes) != tuple(df_header.dtypes)

    def chech_row_wise(file_path, nrows=20):
        with open(file_path, 'r') as f:
            n_lines = sum(1 for _ in f)

        with open(file_path, 'r') as f:
            col_lg_row = 0
            i = 0
            for line in f:
                if len(line) > n_lines:
                    col_lg_row += 1
                i += 1
                if i > nrows or col_lg_row > nrows:
                    break
            if col_lg_row > nrows // 2 or col_lg_row == n_lines:
                row_wise = False
            else:
                row_wise = True
            return row_wise

    if verbose:
        print('Loading .csv ...')
    row_wise = chech_row_wise(file_path, nrows=20)
    if (row_wise is False) and (force_col_reading is not True):
        raise UserWarning('The provided .csv is row based. This could yield '
                          + 'significant loading durations. If you want to '
                          + 'proceed, set force_col_reading=True')

    metadata = dict()
    if row_wise and has_header(file_path):
        data_df = pd.read_csv(file_path)
        if verbose:
            print('Loaded .csv, extracting data ...')
        metadata['loaded_channels'] = data_df.columns.values
    else:
        if verbose:
            print('Loaded .csv, extracting data ...')
        csv_data = pd.read_csv(file_path, header=None)
        data_df = pd.DataFrame(csv_data.to_numpy())
    print('Loading data completed')

    return data_df, metadata


def load_npy(file_path, verbose=True):
    """
    This function loads a .npy file and returns the data as a numpy array.
    --------------------------------------------------------------------------
    :param file_path: Path to the file to be loaded
    :type file_path: str
    :param verbose: Print verbose output
    :type verbose: bool

    :returns data_df: Pandas DataFrame of the loaded data
    :rtype data_df: ~pandas.DataFrame
    :returns metadata: Metadata of the loaded data
    :rtype metadata: dict
    """
    print('Loaded .npy, extracting data ...')
    np_data = np.load(file_path)
    if np_data.shape[0] > np_data.shape[1]:
        np_data = np.rot90(np_data)
        if verbose:
            print('Transposed loaded data.')
    data_df = pd.DataFrame(np_data)
    metadata = dict()

    return data_df, metadata


def load_adicht(file_path, record_id, channel_ids=None, verbose=True):
    """
    This function loads a .adicht file and returns the data as a numpy array.
    --------------------------------------------------------------------------
    :param file_path: Path to the file to be loaded
    :type file_path: str
    :param verbose: Print verbose output
    :type verbose: bool

    :returns data_df: Pandas DataFrame of the loaded data
    :rtype data_df: ~pandas.DataFrame
    :returns metadata: Metadata of the loaded data
    :rtype metadata: dict
    """
    if verbose:
        print('Loading .adicht ...')

    # Extract the data
    data_emg = AdichtReader(file_path)
    if verbose:
        print('Loaded .adicht, extracting data ...')
    if channel_idxs is None:
        channel_idxs = [*range(len(data_emg.channel_map))]
    data_df, fs_emg = data_emg.extract_data(
        channel_idxs=channel_idxs,
        record_id=record_id,
        resample_channels=None,
    )
    metadata = {
        'fs': fs_emg,
        'loaded_channels': data_emg.get_labels(channel_ids),
        'units': data_emg.get_units(channel_ids, record_id),
        'record_id': record_id,
    }
    if verbose:
        print('Loading data completed')

    return data_df, metadata

def poly5unpad(to_be_read):
    """Converts a Poly5 read into an array without padding. This padding is a
    quirk in the python Poly5 interface that pads with zeros on the end.
    --------------------------------------------------------------------------
    :param to_be_read: Filename of python read Poly5
    :type to_be_read: str

    :returns unpadded: Unpadded array
    :rtype unpadded: ~numpy.ndarray
    """
    read_object = Poly5Reader(to_be_read)
    sample_number = read_object.num_samples
    unpadded = read_object.samples[:, :sample_number]
    return unpadded


def matlab5_jkmn_to_array(file_name):
    """
    LEGACY FUNCTION
    This file reads matlab5 files as produced in the Jonkman laboratory, on the
    Biopac system and returns arrays in the format and shape our the ReSurfEMG
    functions work on.
    --------------------------------------------------------------------------
    :param file_name: Filename of matlab5 files
    :type file_name: str

    :returns arrayed: Arrayed data
    :rtype arrayed: ~numpy.ndarray
    """
    file = sio.loadmat(file_name, mdict=None, appendmat=False)
    arrayed = np.rot90(file['data_emg'])
    output_copy = arrayed.copy()
    arrayed[4] = output_copy[0]
    arrayed[3] = output_copy[1]
    arrayed[1] = output_copy[3]
    arrayed[0] = output_copy[4]
    return arrayed


def csv_from_jkmn_to_array(file_name):
    """
    LEGACY FUNCTION
    This function takes a file from the Jonkman lab in csv format and changes
    it into the shape the library functions work on.
    --------------------------------------------------------------------------
    :param file_name: Filename of csv files
    :type file_name: str

    :returns arrayed: Arrayed data
    :rtype arrayed: ~numpy.ndarray
    """
    file = pd.read_csv(file_name)
    new_df = (
        file.T.reset_index().T.reset_index(drop=True)
        .set_axis([f'lead.{i+1}' for i in range(file.shape[1])], axis=1)
    )
    arrayed = np.rot90(new_df)
    arrayed = np.flipud(arrayed)
    return arrayed


def poly_dvrman(file_name):
    """
    LEGACY FUNCTION
    This is a function to read in Duiverman type Poly5 files, which has 18
    layers/pseudo-leads, and return an array of the twelve  unprocessed leads
    for further pre-processing. The leads eliminated were RMS calculated on
    other leads (leads 6-12). The expected organization returned is from leads
    0-5 EMG data, then the following leads
    # 6 Paw: airway pressure (not always recorded)
    # 7 Pes: esophageal pressure (not always recorded)
    # 8 Pga: gastric pressure (not always recorded)
    # 9 RR: respiratory rate I guess (very unreliable)
    # 10 HR: heart rate
    # 11 Tach: number of breath (not reliable)
    --------------------------------------------------------------------------
    :param file_name: Filename of Poly5 Duiverman type file
    :type file_name: str

    :returns samps: Arrayed data
    :rtype samps: ~numpy.ndarray
    """
    data_samples = Poly5Reader(file_name)
    samps = np.vstack([data_samples.samples[:6], data_samples.samples[12:]])

    return samps


def dvrmn_csv_to_array(file_name):
    """
    LEGACY FUNCTION
    Transform an already preprocessed csv from the Duiverman lab into an EMG
    in the format our other functions can work on it. Note that some
    preprocessing steps are already applied so pipelines may need adjusting.
    --------------------------------------------------------------------------
    :param file_name: Filename of csv file
    :type file_name: str

    :returns: arrayed
    :rtype: ~numpy.ndarray
    """
    file = pd.read_csv(file_name)
    new_df = file.drop(['Events', 'Time'], axis=1)
    arrayed = np.rot90(new_df)
    arrayed = np.flipud(arrayed)

    return arrayed


def dvrmn_csv_freq_find(file_name):
    """
    LEGACY FUNCTION
    Extract the sampling rate of a Duiverman type csv of EMG. Note
    this data may be resampled down by a factor of 10.
    --------------------------------------------------------------------------
    :param file_name: Filename of csv file
    :type file_name: str

    :returns fs: Sampling frequency
    :rtype fs: int
    """
    file = pd.read_csv(file_name)
    sample_points = len(file)
    time_string = file['Time'][sample_points - 1]
    seconds = float(time_string[5:10])
    minutes = float(time_string[2:4])
    hours = int(time_string[0:1])
    sum_time = (hours*3600) + (minutes*60) + seconds
    fs = round(sample_points/sum_time)

    return fs
