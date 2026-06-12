"""This file contains functions for math operations to support the functions in ReSurfEMG.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, Self, TypeVar

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Generator

_KT = TypeVar("_KT")
_VT = TypeVar("_VT")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Range(NamedTuple):
    """Utility class for working with ranges (intervals).

    Args:
        start (int): Start of the range.
        end (int): End of the range.
    """

    end: int
    start: int

    def intersects(self, other: Self) -> bool:
        """Return True if this range intersects other range.

        Args:
            other (Range): Another range to compare this one to.

        Returns:
            bool: True if this range intersects another range.
        """
        return (
            ((self.end >= other.end) and (self.start < other.end))
            or ((self.end >= other.start) and (self.start < other.start))
            or ((self.end < other.end) and (self.start >= other.start))
        )

    def precedes(self, other: Self) -> bool:
        """Return True if this range precedes other range.

        Args:
            other (Range): Another range to compare this one to.

        Returns:
            bool: True if this range strictly precedes another range.
        """
        return self.end < other.start

    def to_slice(self) -> slice:
        """Convert this range to a slice.

        Returns:
            slice: A slice with its start set to this range's start and end
                set to this range's end.
        """
        return slice(*map(int, self))  # maps whole tuple set


def zero_one_for_jumps_base(array: np.ndarray, cut_off: float) -> list:
    """Make an array binary for jumps based on a cut-off value.

    This function takes an array and makes it binary (0, 1) based
    on a cut-off value.

    Args:
        array (numpy.ndarray): An array.
        cut_off (float): The number defining a cut-off line for binarization.

    Returns:
        list: Binarized list that can be turned into array.
    """
    return [i >= cut_off for i in array]


def slices_slider(
    array_sample: np.ndarray, slice_len: int
) -> Generator[np.ndarray, None, None]:
    """Produce sequential slices over an array of a certain length.

    This function produces continuous sequential slices over an
    array of a certain length. The function yields, does not return these slices.

    Args:
        array_sample (numpy.ndarray): Array containing the signal.
        slice_len (int): The length of window on the array.

    Yields:
        numpy.ndarray: Sequential slices of length slice_len.
    """
    for i in range(len(array_sample) - slice_len + 1):
        yield array_sample[i : i + slice_len]


def slices_jump_slider(
    array_sample: np.ndarray, slice_len: int, jump: int
) -> Generator[np.ndarray, None, None]:
    """Produce sequential slices over an array of a certain length with jumps.

    This function produces continuous sequential slices over an
    array of a certain length spaced out by a 'jump'.
    The function yields, does not return these slices.

    Args:
        array_sample (numpy.ndarray): Array containing the signal.
        slice_len (int): The length of window on the array.
        jump (int): The amount by which the window is moved at iteration.

    Yields:
        numpy.ndarray: Sequential slices of length slice_len.
    """
    for i in range(len(array_sample) - (slice_len)):
        yield array_sample[(jump * i) : ((jump * i) + slice_len)]


def ranges_of(array: np.ndarray) -> tuple[Range, ...]:
    """Select ranges of 1s in a binary array and return tuples of boundaries.

    This function is made to work with Range class objects, such
    that it selects ranges and returns tuples of boundaries.

    Args:
        array (numpy.ndarray): Array.

    Returns:
        tuple[Range, ...]: Tuple of Range objects representing boundaries.
    """
    marks = np.logical_xor(array[1:], array[:-1])
    boundaries = np.hstack(
        (np.zeros(1), np.where(marks != 0)[0], np.zeros(1) + len(array) - 1)
    )
    if not array[0]:
        boundaries = boundaries[1:]
    if len(boundaries) % 2 != 0:
        boundaries = boundaries[:-1]
    return tuple(Range(*boundaries[i : i + 2]) for i in range(0, len(boundaries), 2))


def intersections(left: list[Range], right: list[Range]) -> list[Range]:
    """Pick ranges from the left that intersect ranges from the right.

    This function works over two arrays, left and right, and allows
    a picking based on intersections. It only takes ranges on the left
    that intersect ranges on the right.

    Args:
        left (list[Range]): List of ranges.
        right (list[Range]): List of ranges.

    Returns:
        list[Range]: Ranges from the left that intersect ranges from the right.
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


def raw_overlap_percent(signal1: np.ndarray, signal2: np.ndarray) -> float:
    """Compute percentage of overlap between two binary signals.

    This function takes two binary 0 or 1 signal arrays and gives
    the percentage of overlap.

    Args:
        signal1 (numpy.ndarray): Binary signal 1.
        signal2 (numpy.ndarray): Binary signal 2.

    Returns:
        float: Raw overlap percent.
    """
    if len(signal1) != len(signal2):
        logger.warning("Warning: length of arrays is not matched")
        longer_signal_len = np.max([len(signal1), len(signal2)])
    else:
        longer_signal_len = len(signal1)

    return sum(signal1.astype(int) & signal2.astype(int)) / longer_signal_len


def merge(left: list, right: list) -> list:
    """Merge two lists based on linear order.

    Args:
        left (list): First list.
        right (list): Second list.

    Returns:
        list: Merged lists.
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


def scale_arrays(array: np.ndarray, maximum: float, minimum: float) -> np.ndarray:
    """Vertically scale arrays to have an absolute maximum value of the maximum parameter.

    This function will scale all arrays along the vertical axis to have an
    absolute maximum value of the maximum parameter.

    Args:
        array (numpy.ndarray): Original signal array with any number of layers.
        maximum (float): The absolute maximum below which the new array exists.
        minimum (float): The absolute minimum above which the new array exists.

    Returns:
        numpy.ndarray: A new array with absolute max of maximum.
    """
    return np.interp(array, (array.min(), array.max()), (maximum, minimum))


def delay_embedding(data: np.ndarray, emb_dim: int, lag: int = 1) -> np.ndarray:
    """Perform a time-delay embedding of a time series.

    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package nolds (NOnLinear measures for Dynamical
    Systems). It performs a time-delay embedding of a time series.

    Args:
        data (numpy.ndarray): Array-like time series data.
        emb_dim (int): The embedded dimension.
        lag (int): The lag between elements in the embedded vectors.

    Returns:
        numpy.ndarray: The embedded vectors.

    Raises:
        ValueError: If the data is too short for the embedding parameters.
    """
    data = np.asarray(data)
    min_len = (emb_dim - 1) * lag + 1
    if len(data) < min_len:
        msg = "cannot embed data of length {} with embedding dimension {} and lag {}, minimum required length is {}"
        raise ValueError(msg.format(len(data), emb_dim, lag, min_len))
    m = len(data) - min_len + 1
    indices = np.repeat([np.arange(emb_dim) * lag], m, axis=0)
    indices += np.arange(m).reshape((m, 1))
    return data[indices]


def save_preprocessed(array: np.ndarray, out_fname: str, force: bool) -> None:
    """Store arrays in a directory.

    This function is written to be called by the cli module.
    It stores arrays in a directory.

    Args:
        array (numpy.ndarray): Array to be stored.
        out_fname (str): Output file name.
        force (bool): Force writing the file.
    """
    if not force and Path(out_fname).exists():
        return
    with contextlib.suppress(FileExistsError):
        Path(out_fname).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_fname, array, allow_pickle=False)


def derivative(signal: np.ndarray, fs: int, window_s: int | None = None) -> np.ndarray:
    """Calculate the first derivative of a signal.

    If window_s is given, the signal is smoothed before derivative calculation.

    Args:
        signal (numpy.ndarray): Signal to calculate the derivative over.
        fs (int): Sampling rate.
        window_s (int, optional): Centralised averaging window length in samples.

    Returns:
        numpy.ndarray: The 1st derivative of the signal, length len(signal)-1.
    """
    if window_s is not None:
        # Moving average over signal
        signal_series = pd.Series(signal)
        signal_moving_average = (
            signal_series.rolling(window=window_s, center=True).mean().to_numpy()
        )
        dsignal_dt = (signal_moving_average[1:] - signal_moving_average[:-1]) * fs
    else:
        dsignal_dt = (signal[1:] - signal[:-1]) * fs

    return dsignal_dt


def bell_curve(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Calculate a shifted, smoothed and amplified bell curve.

    This function calculates a bell curve on the samples of x, shifted by
    b, amplified by a for a standard amplitude of 1.

    Args:
        x (numpy.ndarray): X values to calculate the bell_curve for.
        a (float): Amplitude of the bell-curve.
        b (float): Time shift of the bell-curve along the x-axis.
        c (float): Steepness factor of bell-curve.

    Returns:
        numpy.ndarray: Bell curve values.
    """
    return a * np.exp(-((x - b) ** 2) / c**2)


def running_smoother(array: np.ndarray) -> np.ndarray:
    """Smooth array for use in time calculations.

    Args:
        array (numpy.ndarray): Array to be smoothed.

    Returns:
        numpy.ndarray: Smoothed array.
    """
    n_samples = len(array) // 10
    new_list = np.convolve(abs(array), np.ones(n_samples), "valid") / n_samples
    zeros = np.zeros(n_samples - 1)
    return np.hstack((new_list, zeros))


def get_dict_key_where_value(dictionary: dict[_KT, _VT], value: _VT) -> _KT | None:
    """Return the key of a dictionary where the value matches the input value.

    Args:
        dictionary (dict): Dictionary to search.
        value: Value to search for.

    Returns:
        Key where value is found, or None if not found.
    """
    for key, val in dictionary.items():
        if val == value:
            return key
    return None
