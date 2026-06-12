"""This file contains the quality assessment methods for data classes.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from resurfemg.postprocessing import quality_assessment as qa

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from resurfemg.data_connector.data_classes import TimeSeries
    from resurfemg.data_connector.peakset_class import PeaksSet


def _resolve_emg_cutoff(cutoff: dict | str | None, fs: int, skip_tests: list) -> dict:
    if cutoff is None or cutoff in ("tolerant", "strict"):
        cutoff_options = {
            "tolerant": {
                "interpeak_distance": 1.1,
                "snr": 1.4,
                "aub": 40,
                "curve_fit": 30,
                "aub_window_s": 5 * fs,
                "bell_window_s": 5 * fs,
                "relative_aub_percentile": 75.0,
                "relative_aub_factor": 4.0,
                "relative_etp_upper_percentile": 95.0,
                "relative_etp_upper_factor": 10.0,
                "relative_etp_lower_percentile": 5.0,
                "relative_etp_lower_factor": 0.1,
            },
            "strict": {
                "interpeak_distance": 1.1,
                "snr": 1.75,
                "aub": 30,
                "curve_fit": 25,
                "aub_window_s": 5 * fs,
                "bell_window_s": 5 * fs,
                "relative_aub_percentile": 75.0,
                "relative_aub_factor": 2.0,
                "relative_etp_upper_percentile": 95.0,
                "relative_etp_upper_factor": 10.0,
                "relative_etp_lower_percentile": 5.0,
                "relative_etp_lower_factor": 0.1,
            },
        }
        return cutoff_options[cutoff or "tolerant"]

    if not isinstance(cutoff, dict):
        msg = "Cutoff needs to be a dict, 'tolerant', 'strict', or None"
        raise TypeError(msg)
    for test in ["interpeak_distance", "snr", "aub", "curve_fit"]:
        if test not in skip_tests and test not in cutoff:
            msg = "No cut-off value provided for: " + test
            raise KeyError(msg)
    test_params = {
        "aub": ["window_s"],
        "relative_aub": ["percentile", "factor"],
        "relative_etp": [
            "upper_percentile",
            "upper_factor",
            "lower_percentile",
            "lower_factor",
        ],
    }
    for test_feat, test_param in test_params.items():
        if test_feat not in skip_tests:
            for _param in test_param:
                if (test_feat + "_" + _param) not in cutoff:
                    msg = f"No {test_feat}_{_param} provided for: {test_feat}"
                    raise KeyError(msg)
    return cutoff


def _resolve_peaks(timeseries: TimeSeries, peak_set_name: str, skip_tests: list | None) -> tuple[PeaksSet, list]:
    if peak_set_name not in timeseries.peaks:
        msg = "Non-existent PeaksSet key"
        raise KeyError(msg)

    if skip_tests is None:
        skip_tests = []

    return timeseries.peaks[peak_set_name], skip_tests


def initialize_emg_tests(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: dict | None = None,
    skip_tests: list | None = None,
    parameter_names: dict[str, str] | None = None,
) -> tuple[list, dict, PeaksSet, dict, int, pd.DataFrame, pd.DataFrame]:
    """Initialize local parameters.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes for more information.
    """
    peak_set, skip_tests = _resolve_peaks(timeseries, peak_set_name, skip_tests)

    cutoff = _resolve_emg_cutoff(cutoff, timeseries.param["fs"], skip_tests)

    if parameter_names is None:
        parameter_names = {}
    for parameter in ["ecg", "time_product", "baseline"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    peak_idx = peak_set.peak_df["peak_idx"]
    n_peaks = len(peak_idx)
    quality_values_df = pd.DataFrame(data=peak_idx, columns=["peak_idx"])
    quality_outcomes_df = pd.DataFrame(data=peak_idx, columns=["peak_idx"])
    return (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    )


def test_interpeak_distance(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    n_peaks: int,
    cutoff: dict,
    parameter_names: dict,
) -> pd.DataFrame:
    """Test interpeak distance.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes for more information.
    """
    if parameter_names["ecg"] not in timeseries.peaks:
        msg = "ECG peaks not determined, but required for interpeak distance evaluation."
        raise ValueError(msg)
    ecg_peaks = timeseries.peaks[parameter_names["ecg"]].peak_df["peak_idx"].to_numpy()
    valid_interpeak = qa.interpeak_dist(
        ecg_peak_idxs=ecg_peaks,
        emg_peak_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
        threshold=cutoff["interpeak_distance"],
    )

    valid_interpeak_vec = np.array(n_peaks * [valid_interpeak])
    quality_outcomes_df["interpeak_distance"] = valid_interpeak_vec
    return quality_outcomes_df


def test_snr(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict,
    parameter_names: dict,
) -> pd.DataFrame:
    """Test signal-to-noise ratio.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes for more information.
    """
    if parameter_names["baseline"] not in timeseries.y_data:
        raise ValueError("Baseline not determined, but required for " + " SNR evaluaton.")
    snr_peaks = qa.snr_pseudo(
        src_signal=peak_set.signal,
        peaks=peak_set.peak_df["peak_idx"].to_numpy(),
        baseline=timeseries.y_data[parameter_names["baseline"]],
        fs=timeseries.param["fs"],
    )
    quality_values_df["snr"] = snr_peaks
    valid_snr = snr_peaks > cutoff["snr"]
    quality_outcomes_df["snr"] = valid_snr
    return quality_outcomes_df


def test_aub(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict,
    parameter_names: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test percentage area under the baseline.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes for more information.
    """
    if parameter_names["baseline"] not in timeseries.y_data:
        msg = "Baseline not determined, but required for area under the baseline (AUB) evaluaton."
        raise ValueError(msg)
    if "start_idx" not in peak_set.peak_df.columns:
        msg = "start_idxs not determined, but required for area under the baseline (AUB) evaluaton."
        raise ValueError(msg)
    if "end_idx" not in peak_set.peak_df.columns:
        msg = "end_idxs not determined, but required for area under the baseline (AUB) evaluaton."
        raise ValueError(msg)
    outputs = qa.percentage_under_baseline(
        signal=peak_set.signal,
        fs=timeseries.param["fs"],
        peak_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
        start_idxs=peak_set.peak_df["start_idx"].to_numpy(),
        end_idxs=peak_set.peak_df["end_idx"].to_numpy(),
        baseline=timeseries.y_data[parameter_names["baseline"]],
        aub_window_s=None,
        ref_signal=None,
        aub_threshold=cutoff["aub"],
    )

    valid_timeproducts, percentages_aub, y_refs = outputs
    quality_values_df["aub"] = percentages_aub
    quality_values_df["aub_y_refs"] = y_refs
    quality_outcomes_df["aub"] = valid_timeproducts
    return quality_outcomes_df, quality_values_df


def test_curve_fits(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict,
    parameter_names: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, PeaksSet]:
    """Test curve fit.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes for more information.
    """
    if timeseries.baseline is None:
        msg = "Baseline not determined, but required for area under the baseline (AUB) evaluaton."
        raise ValueError(msg)
    if "start_idx" not in peak_set.peak_df.columns:
        msg = "start_idxs not determined, but required for curve fit evaluaton."
        raise ValueError(msg)
    if "end_idx" not in peak_set.peak_df.columns:
        msg = "end_idxs not determined, but required for curve fit evaluaton."
        raise ValueError(msg)
    if parameter_names["time_product"] not in peak_set.peak_df.columns:
        msg = "ETPs not determined, but required for curve fit evaluaton."
        raise ValueError(msg)
    outputs = qa.evaluate_bell_curve_error(
        peak_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
        start_idxs=peak_set.peak_df["start_idx"].to_numpy(),
        end_idxs=peak_set.peak_df["end_idx"].to_numpy(),
        signal=peak_set.signal,
        fs=timeseries.param["fs"],
        time_products=peak_set.peak_df[parameter_names["time_product"]].to_numpy(),
        bell_window_s=cutoff["bell_window_s"],
        bell_threshold=cutoff["curve_fit"],
    )
    valid_bell_shape, _, percentage_bell_error, y_min, parameters = outputs

    peak_set.peak_df["bell_y_min"] = y_min
    for idx, parameter in enumerate(["a", "b", "c"]):
        peak_set.peak_df["bell_" + parameter] = parameters[:, idx]

    quality_values_df["bell"] = percentage_bell_error
    quality_outcomes_df["bell"] = valid_bell_shape
    return quality_outcomes_df, quality_values_df, peak_set


def test_relative_aub(peak_set: PeaksSet, quality_outcomes_df: pd.DataFrame, cutoff: dict) -> pd.DataFrame:
    """Test the relative area under the baseline.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes
    for more information.
    """
    if "AUB" not in peak_set.peak_df.columns:
        msg = "AUB not determined, but required for relative area under the baseline (AUB) evaluaton."
        raise ValueError(msg)
    valid_relative_aubs = qa.detect_local_high_aub(
        aubs=peak_set.peak_df["AUB"].to_numpy(),
        threshold_percentile=cutoff["relative_aub_percentile"],
        threshold_factor=cutoff["relative_aub_factor"],
    )
    quality_outcomes_df["relative_aub"] = valid_relative_aubs
    return quality_outcomes_df


def test_relative_etp(
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    cutoff: dict,
    parameter_names: dict,
) -> pd.DataFrame:
    """Evaluate extremely low and high timeproducts.

    See TimeSeries.test_emg_quality method in resurfemg.data_connector.data_classes
    for more information.
    """
    if parameter_names["time_product"] not in peak_set.peak_df.columns:
        msg = "ETPs not determined, but required for curve fit evaluaton."
        raise ValueError(msg)
    valid_etps = qa.detect_extreme_time_products(
        time_products=peak_set.peak_df[parameter_names["time_product"]].to_numpy(),
        upper_percentile=cutoff["relative_etp_upper_percentile"],
        upper_factor=cutoff["relative_etp_upper_factor"],
        lower_percentile=cutoff["relative_etp_lower_percentile"],
        lower_factor=cutoff["relative_etp_lower_factor"],
    )
    quality_outcomes_df["relative_etp"] = valid_etps
    return quality_outcomes_df


def initialize_pocc_tests(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: dict | None,
    skip_tests: list | None,
    parameter_names: dict | None,
) -> tuple[list, dict, PeaksSet, dict, int, pd.DataFrame, pd.DataFrame]:
    """Initialize local parameters.

    See TimeSeries.test_pocc_quality method in resurfemg.data_connector.data_classes for more information.
    """
    peak_set, skip_tests = _resolve_peaks(timeseries, peak_set_name, skip_tests)

    if cutoff is None or (isinstance(cutoff, str) and cutoff in ["tolerant", "strict"]):
        cutoff = {
            "consecutive_poccs": 0,
            "dP_up_10": 0.0,
            "dP_up_90": 2.0,
            "dP_up_90_norm": 0.8,
        }
    elif isinstance(cutoff, dict):
        tests_crit = {
            "consecutive_poccs": ["consecutive_poccs"],
            "pocc_upslope": ["dP_up_10", "dP_up_90", "dP_up_90_norm"],
        }
        for test, criteria in tests_crit.items():
            if test not in skip_tests:
                for test_crit in criteria:
                    if test_crit not in cutoff:
                        raise KeyError("No cut-off value provided for: " + test)

    if parameter_names is None:
        parameter_names = {}

    for parameter in ["ventilator_breaths", "time_product", "AUB"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df["peak_idx"].to_numpy())
    quality_values_df = pd.DataFrame(data=peak_set.peak_df["peak_idx"], columns=["peak_idx"])
    quality_outcomes_df = pd.DataFrame(data=peak_set.peak_df["peak_idx"], columns=["peak_idx"])
    return (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    )


def test_consecutive_poccs(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    parameter_names: dict,
) -> pd.DataFrame:
    """Test for consecutive Pocc manoeuvres.

    See TimeSeries.test_pocc_quality method in resurfemg.data_connector.data_classes for more information.
    """
    if parameter_names["ventilator_breaths"] not in timeseries.peaks:
        msg = "Ventilator breaths not determined, but required for consecutive Pocc evaluation."
        raise ValueError(msg)
    vent_breaths = parameter_names["ventilator_breaths"]
    ventilator_breath_idxs = timeseries.peaks[vent_breaths].peak_df["peak_idx"].to_numpy()
    valid_manoeuvres = qa.detect_non_consecutive_manoeuvres(
        ventilator_breath_idxs=ventilator_breath_idxs,
        manoeuvres_idxs=peak_set.peak_df["peak_idx"].to_numpy(),
    )
    quality_outcomes_df["consecutive_poccs"] = valid_manoeuvres
    return quality_outcomes_df


def test_pocc_upslope(
    timeseries: TimeSeries,
    peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict,
    parameter_names: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test for sudden Pocc release.

    See TimeSeries.test_pocc_quality method in resurfemg.data_connector.data_classes for more information.
    """
    if "end_idx" not in peak_set.peak_df.columns:
        msg = "Pocc end_idx not determined, but required for Pocc upslope evaluaton."
        raise ValueError(msg)
    if parameter_names["time_product"] not in peak_set.peak_df.columns:
        msg = "PTPs not determined, but required for Pocc upslope evaluaton."
        raise ValueError(msg)
    valid_poccs, criteria_matrix = qa.pocc_quality(
        p_vent_signal=timeseries.y_data["raw"],
        pocc_peaks=peak_set.peak_df["peak_idx"].to_numpy(),
        pocc_ends=peak_set.peak_df["end_idx"].to_numpy(),
        ptp_occs=peak_set.peak_df[parameter_names["time_product"]].to_numpy(),
        dp_up_10_threshold=cutoff["dP_up_10"],
        dp_up_90_threshold=cutoff["dP_up_90"],
        dp_up_90_norm_threshold=cutoff["dP_up_90_norm"],
    )
    quality_values_df["dP_up_10"] = criteria_matrix[0, :]
    quality_values_df["dP_up_90"] = criteria_matrix[1, :]
    quality_values_df["dP_up_90_norm"] = criteria_matrix[2, :]

    quality_outcomes_df["pocc_upslope"] = valid_poccs
    return quality_outcomes_df, quality_values_df


def _resolve_linked_peaks_cutoffs(cutoff: dict | str | None, skip_tests: list) -> dict[str, float]:
    if (cutoff is None) or (isinstance(cutoff, str) and cutoff == "tolerant"):
        return {"fraction_emg_breaths": 0.1, "delta_min": -0.5, "delta_max": 2.0}
    if isinstance(cutoff, str) and cutoff == "strict":
        return {"fraction_emg_breaths": 0.5, "delta_min": -0.5, "delta_max": 2.0}
    if isinstance(cutoff, dict):
        tests_crit = {
            "fraction_emg_breaths": ["fraction_emg_breaths"],
            "event_timing": ["delta_min", "delta_max"],
        }
        for test, criteria in tests_crit.items():
            if test not in skip_tests:
                for crit in criteria:
                    if crit not in cutoff:
                        raise KeyError("No cut-off value provided for: " + test)
        return cutoff
    return {}


def initialize_linked_peaks_tests(
    timeseries: TimeSeries,
    peak_set_name: str,
    linked_timeseries: TimeSeries,
    linked_peak_set_name: str,
    cutoff: dict | str | None,
    skip_tests: list | None,
    parameter_names: dict | None,
) -> tuple[list, dict, PeaksSet, PeaksSet, dict, int, pd.DataFrame, pd.DataFrame]:
    """Initialize local parameter.

    See TimeSeries.test_linked_peak_sets method in resurfemg.data_connector.data_classes for more information.
    """
    peak_set, skip_tests = _resolve_peaks(timeseries, peak_set_name, skip_tests)
    linked_peak_set, skip_tests = _resolve_peaks(linked_timeseries, linked_peak_set_name, skip_tests)

    cutoff = _resolve_linked_peaks_cutoffs(cutoff, skip_tests)

    if not isinstance(cutoff, dict):
        msg = "Cutoff needs to be a dict."
        raise TypeError(msg)

    if parameter_names is None:
        parameter_names = {}

    for parameter in ["ventilator_breaths", "rr"]:
        if parameter not in parameter_names:
            parameter_names[parameter] = parameter

    n_peaks = len(peak_set.peak_df["peak_idx"].to_numpy())
    quality_values_df = pd.DataFrame(data=peak_set.peak_df["peak_idx"], columns=["peak_idx"])
    quality_outcomes_df = pd.DataFrame(data=peak_set.peak_df["peak_idx"], columns=["peak_idx"])
    return (
        skip_tests,
        cutoff,
        peak_set,
        linked_peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    )


def test_fraction_detected_breaths(
    native_peak_set: PeaksSet,
    linked_time_series: TimeSeries,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    n_peaks: int,
    cutoff: dict,
    parameter_names: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test detected peak fraction.

    See TimeSeries.test_linked_peak_sets method in resurfemg.data_connector.data_classes for more information.
    """
    native_peak_idxs = native_peak_set.peak_df["peak_idx"].to_numpy()

    fraction_emg_breaths, _ = qa.evaluate_respiratory_rates(
        emg_breath_idxs=native_peak_idxs,
        t_emg=max(native_peak_set.t_data),
        rr_vent=linked_time_series.param[parameter_names["rr"]],
    )
    valid_fraction = fraction_emg_breaths > cutoff["fraction_emg_breaths"]
    fraction_emg_breaths_vec = np.array(n_peaks * [fraction_emg_breaths])
    valid_fraction_vec = np.array(n_peaks * [valid_fraction])
    quality_outcomes_df["detected_fraction"] = valid_fraction_vec
    quality_values_df["detected_fraction"] = fraction_emg_breaths_vec
    return quality_outcomes_df, quality_values_df


def test_event_timing(
    timeseries: TimeSeries,
    native_peak_set: PeaksSet,
    linked_timeseries: TimeSeries,
    linked_peak_set: PeaksSet,
    quality_outcomes_df: pd.DataFrame,
    quality_values_df: pd.DataFrame,
    cutoff: dict,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Test event timing relative to other peak_set.

    See TimeSeries.test_linked_peak_sets method in resurfemg.data_connector.data_classes for more information.
    """
    native_peak_idxs = native_peak_set.peak_df["peak_idx"].to_numpy()
    linked_peak_idxs = linked_peak_set.peak_df["peak_idx"].to_numpy()

    t_events_1 = native_peak_idxs / timeseries.param["fs"]
    t_events_2 = linked_peak_idxs / linked_timeseries.param["fs"]

    correct_timing, delta_time = qa.evaluate_event_timing(
        t_events_1,
        t_events_2,
        delta_min=cutoff["delta_min"],
        delta_max=cutoff["delta_max"],
    )
    quality_outcomes_df["event_timing"] = correct_timing
    quality_values_df["event_delta_time"] = delta_time
    return quality_outcomes_df, quality_values_df


def test_emg_quality(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: dict | None = None,  # noqa: PT028
    skip_tests: list[str] | None = None,  # noqa: PT028
    parameter_names: dict[str, str] | None = None,  # noqa: PT028
    verbose: bool = True,  # noqa: PT028
) -> None:
    """Test EMG PeaksSet according to quality criteria in Warnaar et al. (2024).

    interpeak_distance, snr, aub, curve_fit, and extended with
    relative area under the baseline (relative_aub) and relative ETP
    (relative_ETP). Peak validity is updated in the PeaksSet object.
    See postprocessing.quality_assessment submodule for details.
    -----------------------------------------------------------------------
    :param peak_set_name: PeaksSet name in self.peaks dict
    :type peak_set_name: str
    :param cutoff: Cut-off criteria for passing the tests 'tolerant' and
    'strict' can also be providedto use the values from Warnaar et al.
    :type cutoff: dict
    :param skip_tests: List of tests to skip.
    :type skip_tests: list
    :param parameter_names: Optionally refer to custom parameter names for
    default PeaksSet (ecg)
    :type parameter_names: dict
    :param verbose: Output the test values, and pass/fail to console.
    :type verbose: bool

    :returns: None
    :rtype: None
    """
    output = initialize_emg_tests(timeseries, peak_set_name, cutoff, skip_tests, parameter_names)
    (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    ) = output

    if "interpeak_dist" not in skip_tests:
        quality_outcomes_df = test_interpeak_distance(
            timeseries, peak_set, quality_outcomes_df, n_peaks, cutoff, parameter_names
        )

    if "snr" not in skip_tests:
        quality_outcomes_df = test_snr(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    if "aub" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_aub(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    if "curve_fit" not in skip_tests:
        quality_outcomes_df, quality_values_df, peak_set = test_curve_fits(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    if "relative_aub" not in skip_tests:
        quality_outcomes_df = test_relative_aub(peak_set, quality_outcomes_df, cutoff)

    if "relative_etp" not in skip_tests:
        quality_outcomes_df = test_relative_etp(peak_set, quality_outcomes_df, cutoff, parameter_names)

    peak_set.update_test_outcomes(quality_values_df)
    peak_set.evaluate_validity(quality_outcomes_df)
    if verbose:
        logger.info("Test values:\n%s", peak_set.quality_values_df)
        logger.info("Test outcomes:\n%s", peak_set.quality_outcomes_df)


def test_pocc_quality(
    timeseries: TimeSeries,
    peak_set_name: str,
    cutoff: dict | None = None,  # noqa: PT028
    skip_tests: list[str] | None = None,  # noqa: PT028
    parameter_names: dict[str, str] | None = None,  # noqa: PT028
    verbose: bool = True,  # noqa: PT028
) -> None:
    """Test EMG PeaksSet according to quality criteria in Warnaar et al. (2024).

    consecutive_poccs, and pocc_upslope. Peak validity is updated
    in the PeaksSet object.
    -----------------------------------------------------------------------
    :param peak_set_name: PeaksSet name in self.peaks dict
    :type peak_set_name: str
    :param cutoff: Cut-off criteria for passing the tests. 'tolerant' and
    'strict' can also be provided  to use the values from Warnaar et al.
    :type cutoff: dict
    :param skip_tests: List of tests to skip.
    :type skip_tests: list
    :param parameter_names: Refer to custom parameter names for default
    PeaksSet and parameters (ventilator_breaths, time_product, AUB)
    :type parameter_names: dict
    :param verbose: Output the test values, and pass/fail to console.
    :type verbose: bool

    :returns: None
    :rtype: None
    """
    output = initialize_pocc_tests(timeseries, peak_set_name, cutoff, skip_tests, parameter_names)
    (
        skip_tests,
        cutoff,
        peak_set,
        parameter_names,
        _,
        quality_values_df,
        quality_outcomes_df,
    ) = output

    if "consecutive_poccs" not in skip_tests:
        quality_outcomes_df = test_consecutive_poccs(timeseries, peak_set, quality_outcomes_df, parameter_names)

    if "pocc_upslope" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_pocc_upslope(
            timeseries,
            peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
            parameter_names,
        )

    peak_set.update_test_outcomes(quality_values_df)
    peak_set.evaluate_validity(quality_outcomes_df)
    if verbose:
        logger.info("Test values:\n%s", peak_set.quality_values_df)
        logger.info("Test outcomes:\n%s", peak_set.quality_outcomes_df)


def test_linked_peak_sets(
    timeseries: TimeSeries,
    peak_set_name: str,
    linked_timeseries: TimeSeries,
    linked_peak_set_name: str,
    parameter_names: dict[str, str] | None = None,  # noqa: PT028
    cutoff: dict | None = None,  # noqa: PT028
    skip_tests: list[str] | None = None,  # noqa: PT028
    verbose: bool = True,  # noqa: PT028
) -> None:
    """Test the number of detected linked breaths.

    Test number of detected breaths in the native PeaksSet compared to
    number (fraction_emg_breaths) of and timing peaks (event_timing) in the
    linked PeaksSet. Peak validity is updated in the PeaksSet object.

    :param peak_set_name: PeaksSet name in self.peaks dict
    :type peak_set_name: str
    :param linked_timeseries: TimeSeries object with linked signal
    :type linked_timeseries: TimeSeries
    :param linked_peak_set_name: PeaksSet name in linked_timeseries.peaks
    :type linked_peak_set_name: str
    :param parameter_names: Optionally refer to custom parameter names for
    default PeaksSet and parameter names ('rr', )
    :type parameter_names: dict
    :param cutoff: Cut-off criteria for passing the tests. 'tolerant' and
    'strict' can also be provided use the pre-defined values.
    :type cutoff: dict
    :param skip_tests: List of tests to skip.
    :type skip_tests: list
    :param verbose: Output the test values, and pass/fail to console.
    :type verbose: bool

    :returns: None
    :rtype: None
    """
    output = initialize_linked_peaks_tests(
        timeseries,
        peak_set_name,
        linked_timeseries,
        linked_peak_set_name,
        cutoff,
        skip_tests,
        parameter_names,
    )
    (
        skip_tests,
        cutoff,
        native_peak_set,
        linked_peak_set,
        parameter_names,
        n_peaks,
        quality_values_df,
        quality_outcomes_df,
    ) = output

    if "fraction_emg_breaths" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_fraction_detected_breaths(
            native_peak_set,
            linked_timeseries,
            quality_outcomes_df,
            quality_values_df,
            n_peaks,
            cutoff,
            parameter_names,
        )

    if "event_timing" not in skip_tests:
        quality_outcomes_df, quality_values_df = test_event_timing(
            timeseries,
            native_peak_set,
            linked_timeseries,
            linked_peak_set,
            quality_outcomes_df,
            quality_values_df,
            cutoff,
        )

    native_peak_set.update_test_outcomes(quality_values_df)
    native_peak_set.evaluate_validity(quality_outcomes_df)
    if verbose:
        logger.info("Test values:\n%s", native_peak_set.quality_values_df)
        logger.info("Test outcomes:\n%s", native_peak_set.quality_outcomes_df)
