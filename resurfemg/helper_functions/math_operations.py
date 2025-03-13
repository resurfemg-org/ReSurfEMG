"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains functions for math operations to support the functions in
ReSurfEMG
"""

import os
from collections import namedtuple
import numpy as np
import pandas as pd


class Range(namedtuple('RangeBase', 'start,end')):
    """Utility class for working with ranges (intervals).
    ---------------------------------------------------------------------------
    :param start: Start of the range
    :type start: ~int
    :param end: End of the range
    :type end: ~int
    """

    def intersects(self, other):
        """Returns `True` if this range intersects `other` range.
        -----------------------------------------------------------------------
        :param other: Another range to compare this one to
        :type other: ~resurfemg.helper_functions.Range

        :returns: `True` if this range intersects another range
        :rtype: bool
        """
        return (
            (self.end >= other.end) and (self.start < other.end) or
            (self.end >= other.start) and (self.start < other.start) or
            (self.end < other.end) and (self.start >= other.start)
        )

    def precedes(self, other):
        """Returns `True` if this range precedes `other` range.

        :param other: Another range to compare this one to
        :type other: ~resurfemg.helper_functions.Range

        :returns: `True` if this range strictly precedes another range
        :rtype: bool
        """
        return self.end < other.start

    def to_slice(self):
        """Converts this range to a :class:`slice`.
        -----------------------------------------------------------------------
        :returns: A slice with its start set to this range's start and end set
            to this range's end

        :rtype: slice
        """
        return slice(*map(int, self))   # maps whole tuple set


def zero_one_for_jumps_base(array, cut_off):
    """This function takes an array and makes it binary (0, 1) based
    on a cut-off value.
    ---------------------------------------------------------------------------
    :param array: An array
    :type array: ~numpy.ndarray
    :param cut_off: The number defining a cut-off line for binarization
    :type cut_off: float

    :returns array_list: Binarized list that can be turned into array
    :rtype array_list: list
    """
    array_list = []
    for i in array:
        if i < cut_off:
            i = 0
        else:
            i = 1
        array_list.append(i)
    return array_list


def slices_slider(array_sample, slice_len):
    """This function produces continuous sequential slices over an
    array of a certain length.  The inputs are the following -
    `array_sample`, the signal and `slice_len` - the
    window which you wish to slide with. The function yields, does
    not return these slices.
    ---------------------------------------------------------------------------
    :param array_sample: array containing the signal
    :type array_sample: ~numpy.ndarray
    :param slice_len: the length of window on the array
    :type slice_len: int

    :returns: Actually yields, no return
    :rtype: ~numpy.ndarray
    """
    for i in range(len(array_sample) - slice_len + 1):
        yield array_sample[i:i + slice_len]


def slices_jump_slider(array_sample, slice_len, jump):
    """
    This function produces continuous sequential slices over an
    array of a certain length spaced out by a 'jump'.
    The function yields, does not return these slices.
    ---------------------------------------------------------------------------
    :param array_sample: array containing the signal
    :type array_sample: ~numpy.ndarray
    :param slice_len: the length of window on the array
    :type slice_len: int
    :param jump: the amount by which the window is moved at iteration
    :type jump: int

    :returns: Actually yields, no return
    :rtype: ~numpy.ndarray
    """
    for i in range(len(array_sample) - (slice_len)):
        yield array_sample[(jump*i):((jump*i) + slice_len)]


def ranges_of(array):
    """This function is made to work with :class:`Range` class objects, such
    that is selects ranges and returns tuples of boundaries.
    ---------------------------------------------------------------------------
    :param array: array
    :type  array: ~numpy.ndarray

    :return: range_return
    :rtype: tuple
    """
    marks = np.logical_xor(array[1:], array[:-1])
    boundaries = np.hstack(
        (np.zeros(1), np.where(marks != 0)[0], np.zeros(1) + len(array) - 1)
    )
    if not array[0]:
        boundaries = boundaries[1:]
    if len(boundaries) % 2 != 0:
        boundaries = boundaries[:-1]
    range_return = tuple(
        Range(*boundaries[i:i+2]) for i in range(0, len(boundaries), 2)
    )
    return range_return


def intersections(left, right):
    """This function works over two arrays, `left` and
    `right`, and allows a picking based on intersections.  It
    only takes ranges on the left that intersect ranges on the right.
    ---------------------------------------------------------------------------
    :param left: List of ranges
    :type left: List[Range]
    :param right: List of ranges
    :type right: List[Range]

    :returns result: Ranges from the left that intersect ranges from the right.
    :rtype result: List[Range]
    """
    i, j = 0, 0
    result = []
    while i < len(left) and j < len(right):
        lelt, relt = left[i], right[j]
        if lelt.intersects(relt):
            result.append(lelt)
            i += 1
        elif relt.precedes(lelt):
            j += 1
        elif lelt.precedes(relt):
            i += 1
    return result


def raw_overlap_percent(signal1, signal2):
    """This function takes two binary 0 or 1 signal arrays and gives
    the percentage of overlap.
    ---------------------------------------------------------------------------
    :param signal1: Binary signal 1
    :type signal1: ~numpy.ndarray
    :param rsignal2: Binary signal 2
    :type rsignal2: ~numpy.ndarray

    :returns _raw_overlap_percent: Raw overlap percent
    :rtype _raw_overlap_percent: float
    """
    if len(signal1) != len(signal2):
        print('Warning: length of arrays is not matched')
        longer_signal_len = np.max([len(signal1), len(signal2)])
    else:
        longer_signal_len = len(signal1)

    _raw_overlap_percent = sum(
        signal1.astype(int) & signal2.astype(int)
    ) / longer_signal_len
    return _raw_overlap_percent


def merge(left, right):
    """
    Merge two lists based linear order.
    ---------------------------------------------------------------------------
    :param left: First list
    :type left: List
    :param right: Second list
    :type right: List

    :returns output: Merged lists
    :rtpe output: List
    """
    # Initialize an empty list output that will be populated
    # with sorted elements.
    # Initialize two variables i and j which are used pointers when
    # iterating through the lists.
    output = []
    i = j = 0

    # Executes the while loop if both pointers i and j are less than
    # the length of the left and right lists
    while i < len(left) and j < len(right):
        # Compare the elements at every position of both
        # lists during each iteration
        if left[i] < right[j]:
            # output is populated with the lesser value
            output.append(left[i])
            # 10. Move pointer to the right
            i += 1
        else:
            output.append(right[j])
            j += 1
    # The remnant elements are picked from the current
    # pointer value to the end of the respective list
    output.extend(left[i:])
    output.extend(right[j:])

    return output


def scale_arrays(array, maximum, minimum):
    """
    This function will scale all arrays along the vertical axis to have an
    absolute maximum value of the maximum parameter
    ---------------------------------------------------------------------------
    :param array: Original signal array with any number iflayers
    :type array: ~numpy.ndarray
    :param maximum: the absolute maximum below which the new array exists
    :type maximum: float
    :param minimum: the absolute maximum below which the new array exists
    :type minimum: float

    :returns reformed: a new array with absolute max of maximum
    :rtype reformed: ~numpy.ndarray
    """
    reformed = np.interp(
        array,
        (array.min(), array.max()),
        (maximum, minimum)
    )
    return reformed


def delay_embedding(data, emb_dim, lag=1):
    """
    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package nolds (NOnLinear measures for Dynamical
    Systems).It performs a time-delay embedding of a time series
    ---------------------------------------------------------------------------
    :param data: array-like
    :type data: array
    :param emb_dim: the embedded dimension
    :type emb_dim: int
    :param lag: the lag between elements in the embedded vectors
    :type lag: int

    :returns matrix_vectors: the embedded vectors
    :rtype matrix_vectors: ~nd.array
    """
    data = np.asarray(data)
    min_len = (emb_dim - 1) * lag + 1
    if len(data) < min_len:
        msg = "cannot embed data of length {} with embedding dimension {} " \
            + "and lag {}, minimum required length is {}"
        raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
    m = len(data) - min_len + 1
    indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
    indices += np.arange(m).reshape((m, 1))
    matrix_vectors = data[indices]

    return matrix_vectors


def save_preprocessed(array, out_fname, force):
    """
    This function is written to be called by the cli module.
    It stores arrays in a directory.
    ---------------------------------------------------------------------------
    :param array: array to be stored
    :type array: ~nd.array
    :param out_fname: output file name
    :type out_fname: str
    :param force: force writing the file
    :type force: bool

    :returns: None
    :rtype: None
    """
    if not force:
        if os.path.isfile(out_fname):
            return
    try:
        os.makedirs(os.path.dirname(out_fname))
    except FileExistsError:
        pass
    np.save(out_fname, array, allow_pickle=False)


def derivative(signal, fs, window_s=None):
    """
    This function calculates the first derivative of a signal. If
    window_s is given, the signal is smoothed before derivative calculation.
    ---------------------------------------------------------------------------
    :param signal: signal to calculate the derivate over
    :type signal: ~numpy.ndarray
    :param fs: sampling rate
    :type fs: int
    :param window_s: centralised averaging window length in samples
    :type window_s: int

    :returns dsignal_dt: The 1st derivative of the signal length len(signal)-1.
    :rtype dsignal_dt: ~numpy.ndarray
    """

    if window_s is not None:
        # Moving average over signal
        signal_series = pd.Series(signal)
        signal_moving_average = signal_series.rolling(
            window=window_s, center=True).mean().values
        dsignal_dt = (signal_moving_average[1:]
                      - signal_moving_average[:-1]) * fs
    else:
        dsignal_dt = (signal[1:] - signal[:-1]) * fs

    return dsignal_dt


def bell_curve(x, a, b, c):
    """
    This function calculates a bell curve on the samples of `x`, shifted by
    b, amplified by a for a standard amplitude of 1.
    ---------------------------------------------------------------------------
    :param x: x values to calculate the bell_curve for
    :type x: ~numpy.ndarray
    :param a: amplitude of the bell-curve
    :type a: ~float
    :param b: time shift of the bell-curve along the x-axis.
    :type b: ~float
    :param c: steepness factor of bell-curve.
    :type c: ~float

    :returns: bell curve
    :rtype: ~numpy.ndarray
    """
    return a * np.exp(-(x - b) ** 2 / c ** 2)


def running_smoother(array):
    """
    This is the smoother to use in time calculations
    ---------------------------------------------------------------------------
    :param array: array to be smoothed
    :type array: ~numpy.ndarray

    :returns smoothed_array: smoothed array
    :rtype smoothed_array: ~numpy.ndarray
    """
    n_samples = len(array) // 10
    new_list = np.convolve(abs(array), np.ones(n_samples), "valid") / n_samples
    zeros = np.zeros(n_samples - 1)
    smoothed_array = np.hstack((new_list, zeros))
    return smoothed_array


def get_dict_key_where_value(dictionary, value):
    """
    This function returns the key of a dictionary where the value matches
    the input value.
    ---------------------------------------------------------------------------
    :param dictionary: dictionary to search
    :type dictionary: dict
    :param value: value to search for
    :type value: any

    :returns key: key where value is found
    :rtype key: any
    """
    for key, val in dictionary.items():
        if val == value:
            return key
    return None
