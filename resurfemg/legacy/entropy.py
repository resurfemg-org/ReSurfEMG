"""Legacy code for entropy features.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains legacy functions to extract entropy features from
preprocessed EMG arrays. These methods are unfinished and untested, and hence
not included in the ReSurfEMG module.
"""  # noqa: INP001

from __future__ import annotations

import collections
import logging
import math
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import entropy

from resurfemg.helper_functions.math_operations import delay_embedding

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


def entropical(sig: np.ndarray) -> float:
    """Computes an entropy-like signal using math.log w/base 2.

    This function computes something close to certain type of entropy
    of a series signal array.  Input is sig, the signal, and output is an
    array of entropy measurements. The function can be used inside a generator
    to read over slices. Note it is not a true entropy, and works best with
    very small numbers.

    Args:
        sig (numpy.ndarray): Array containing the signal.

    Returns:
        float: Number for an entropy-like signal using math.log w/base 2.
    """
    probabilit = [n_x / len(sig) for x, n_x in collections.Counter(sig).items()]
    e_x = [-p_x * math.log2(p_x) for p_x in probabilit]
    return sum(e_x)


def entropy_scipy(sli: np.ndarray, base: float | None = None) -> float:
    """Wrap scipy.stats entropy for use in the resurfemg library.

    This function wraps scipy.stats entropy (which is a Shannon entropy)
    for use in the resurfemg library, it can be used in a slice iterator
    as a drop-in substitute for the hf.entropical but it is a true entropy.

    Args:
        sli (numpy.ndarray): Array.
        base (float, optional): Logarithm base for entropy calculation.

    Returns:
        float: The entropy of the slice.
    """
    _, counts = np.unique(sli, return_counts=True)
    return float(entropy(counts / len(counts), base=base))


def rowwise_chebyshev(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate the row-wise Chebyshev distance between two 2D arrays."""
    return np.max(np.abs(x - y), axis=1)


def sampen(
    data: np.ndarray,
    emb_dim: int = 2,
    tolerance: float | None = None,
    dist: Callable = rowwise_chebyshev,
    closed: bool = False,
) -> float:
    """Computes the sample entropy of time sequence data.

    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package
    nolds (NOnLinear measures for Dynamical Systems).
    It computes the sample entropy of time sequence data.

    Returns:
    the sample entropy of the data (negative logarithm of ratio between
    similar template vectors of length emb_dim + 1 and emb_dim)
    [c_m, c_m1]:
    list of two floats: count of similar template vectors of length emb_dim
    (c_m) and of length emb_dim + 1 (c_m1)
    [float list, float list]:
    Lists of lists of the form ``[dists_m, dists_m1]`` containing the
    distances between template vectors for m (dists_m)
    and for m + 1 (dists_m1).
    Reference
    .. [se_1] J. S. Richman and J. R. Moorman, “Physiological time-series
    analysis using approximate entropy and sample entropy,”
    American Journal of Physiology-Heart and Circulatory Physiology,
    vol. 278, no. 6, pp. H2039-H2049, 2000.

    Args:
        data (numpy.ndarray): Array-like time series data.
        emb_dim (int): The embedding dimension (length of vectors to compare).
        tolerance (float, optional): Distance threshold for two template vectors
            to be considered equal. Defaults to 0.2 * std(data) at emb_dim=2,
            corrected for dimension effect for other values of emb_dim.
        dist (callable): Distance function used to calculate the distance between
            template vectors. Defaults to rowwise_chebyshev.
        closed (bool): If True, checks for vector pairs whose distance is in the
            closed interval [0, r]; otherwise uses the open interval [0, r).

    Returns:
        float: Sample entropy of the data.
    """
    data = np.asarray(data)

    if tolerance is None:
        lint_helper = 0.5627 * np.log(emb_dim) + 1.3334
        tolerance = np.std(data, ddof=1) * 0.1164 * lint_helper
    n = len(data)

    # build matrix of "template vectors"
    # (all consecutive subsequences of length m)
    # x0 x1 x2 x3 ... xm-1
    # x1 x2 x3 x4 ... xm
    # x2 x3 x4 x5 ... xm+1
    # ...
    # x_n-m-1     ... xn-1

    # since we need two of these matrices for m = emb_dim and
    # "" m = emb_dim +1,
    # we build one that is large enough => shape (emb_dim+1, n-emb_dim)

    # note that we ignore the last possible template vector with
    #  length emb_dim,
    # because this vector has no corresponding vector of length m+
    # 1 and thus does
    # not count towards the conditional probability
    # (otherwise first dimension would be n-emb_dim+1 and not n-emb_dim)
    t_vecs = delay_embedding(np.asarray(data), emb_dim + 1, lag=1)
    counts = []
    for m in [emb_dim, emb_dim + 1]:
        counts.append(0)
        # get the matrix that we need for the current m
        t_vecs_m = t_vecs[: n - m + 1, :m]
        # successively calculate distances between each pair of templ vectrs
        for i in range(len(t_vecs_m) - 1):
            dsts = dist(t_vecs_m[i + 1 :], t_vecs_m[i])
            # count how many distances are smaller than the tolerance
            if closed:
                counts[-1] += np.sum(dsts <= tolerance)
            else:
                counts[-1] += np.sum(dsts < tolerance)
    if counts[0] > 0 and counts[1] > 0:
        saen = -np.log(1.0 * counts[1] / counts[0])
    else:
        # log would be infinite or undefined => cannot determine saen
        zcounts = []
        if counts[0] == 0:
            zcounts.append("emb_dim")
        if counts[1] == 0:
            zcounts.append("emb_dim + 1")
        print_message = (
            "Zero vectors are within tolerance for {}. Consider raising tolerance parameter to avoid {} result."
        )
        warnings.warn(
            print_message.format(
                " and ".join(zcounts),
                "NaN" if len(zcounts) == 2 else "inf",  # noqa: PLR2004
            ),
            RuntimeWarning,
        )
        if counts[0] == 0 and counts[1] == 0:
            saen = np.nan
        elif counts[0] == 0:
            saen = -np.inf
        else:
            saen = np.inf
    return saen


def sampen_optimized(data: np.ndarray, tolerance: float | None = None, closed: bool = False) -> float:
    """Computes the sample entropy of time sequence data, optimized for emb_dim=1.

    The following code is adapted from openly licensed code written by
    Christopher Schölzel in his package
    nolds (NOnLinear measures for Dynamical Systems).
    It computes the sample entropy of time sequence data.
    emb_dim has been set to 1 (not parameterized)

    Returns:
    the sample entropy of the data (negative logarithm of ratio between
    similar template vectors of length emb_dim + 1 and emb_dim)
    [c_m, c_m1]:
    list of two floats: count of similar template vectors of length emb_dim
    (c_m) and of length emb_dim + 1 (c_m1)
    [float list, float list]:
    Lists of lists of the form ``[dists_m, dists_m1]`` containing the
    distances between template vectors for m (dists_m)
    and for m + 1 (dists_m1).
    Reference:
    .. [se_1] J. S. Richman and J. R. Moorman, “Physiological time-series
    analysis using approximate entropy and sample entropy,”
    American Journal of Physiology-Heart and Circulatory Physiology,
    vol. 278, no. 6, pp. H2039-H2049, 2000.

    Kwargs are pre-set and not available. For more extensive
    you should use the sampen function.

    Args:
        data (numpy.ndarray): Array-like time series data.
        tolerance (float, optional): Distance threshold for two template vectors
            to be considered equal.
        closed (bool): If True, checks for vector pairs in the closed interval
            [0, r]; otherwise uses the open interval [0, r).

    Returns:
        float: Sample entropy of the data.
    """
    # TODO: this function can still be further optimized
    data = np.asarray(data)
    if tolerance is None:
        lint_helper = 0.5627 * np.log(1) + 1.3334
        tolerance = np.std(data, ddof=1) * 0.1164 * lint_helper
    n = len(data)

    if tolerance is None:
        raise ValueError

    # TODO(): This can be done with just using NumPy
    t_vecs = delay_embedding(np.asarray(data), 3, lag=1)

    counts = calc_closed_sampent(t_vecs, n, tolerance) if closed else calc_open_sampent(t_vecs, n, tolerance)

    if counts[0] > 0 and counts[1] > 0:
        saen = -np.log(1.0 * counts[1] / counts[0])
    else:
        # log would be infinite or undefined => cannot determine saen
        zcounts = []
        if counts[0] == 0:
            zcounts.append("1")
        if counts[1] == 0:
            zcounts.append("2")
        print_message = (
            "Zero vectors are within tolerance for {}. Consider raising tolerance parameter to avoid {} result."
        )
        warnings.warn(
            print_message.format(
                " and ".join(zcounts),
                "NaN" if len(zcounts) == 2 else "inf",  # noqa: PLR2004
            ),
            RuntimeWarning,
        )
        if counts[0] == 0 and counts[1] == 0:
            saen = np.nan
        elif counts[0] == 0:
            saen = -np.inf
        else:
            saen = np.inf
    return saen


def calc_closed_sampent(
    t_vecs: np.ndarray,  # noqa: ARG001
    n: int,  # noqa: ARG001
    tolerance: float,  # noqa: ARG001
) -> tuple[float, float]:
    """Calculate the counts for closed sample entropy."""
    # TODO(someone?): Analogous to calc_open_sampent
    return np.nan, np.nan


def calc_open_sampent(t_vecs: np.ndarray, n: int, tolerance: float) -> tuple[float, float]:
    """Calculate the counts for open sample entropy."""
    triplets = t_vecs[: n - 2, :3]

    raw_dsts = tuple(triplets[i + 1 :] - triplets[i] for i in range(len(triplets) - 1))
    dsts = np.concatenate(raw_dsts)
    dsts_abs = np.abs(dsts)
    dsts_gt = dsts_abs < tolerance
    dsts_max_a = np.logical_and(dsts_gt[:, 0], dsts_gt[:, 1])
    dsts_max = np.logical_and(dsts_max_a, dsts_gt[:, 2])
    return np.sum(dsts_max_a), np.sum(dsts_max)


def entropy_maker(
    array: np.ndarray,
    method: str = "sample_entropy",
    base: float | None = None,
) -> float:
    """Calculate entropy of an array using the specified method.

    The following code allows a user to input an array and calculate either
    a time-series specific entropy i.e. the nolds or a more general
    Shannon entropy as calculated in scipy.
    It calls entropy functions in the file.

    """
    if method == "scipy":
        output = entropy_scipy(array, base=base)
    elif method == "nolds":
        output = sampen(array)
    elif method == "sample_entropy":
        output = sampen_optimized(array)
    else:
        logger.info("your method is not an option,")
        logger.info("we defaulted to a slow unoptimized sample entropy")
        output = sampen(array)
    return output
