"""Data classes for standardized data storage and automation.

This file contains data classes for standardized data storage and method
automation.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

import functools
import inspect
import logging
import warnings
from textwrap import dedent, wrap
from typing import TYPE_CHECKING, Any, ClassVar, cast

import matplotlib.pyplot as plt
import numpy as np
import scipy

from resurfemg.data_connector.peakset_class import PeaksSet
from resurfemg.helper_functions import data_classes_quality_assessment as data_qa
from resurfemg.helper_functions import math_operations as mo
from resurfemg.pipelines.processing import ecg_removal_gating
from resurfemg.postprocessing import baseline as bl
from resurfemg.postprocessing import event_detection as evt
from resurfemg.postprocessing import features as feat
from resurfemg.preprocessing import ecg_removal as ecg_rm
from resurfemg.preprocessing import envelope as evl
from resurfemg.preprocessing import filtering as filt

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes

# Descriptive constants to avoid magic-number comparisons for array ndim checks
# `_NDIM_VECTOR` means a 1-dimensional (single-channel) input, `_NDIM_MATRIX`
# means a 2-dimensional (multi-channel or shaped) input.
_NDIM_VECTOR = 1
_NDIM_MATRIX = 2

logger = logging.getLogger(__name__)


class TimeSeries:
    """Data class to store, process, and plot single channel time series data.

    Data is stored in the _y_data dictionary, which can be accessed using the
    TimeSeries[`signal_type`]. The default signal types are:
    "raw", "filt", "clean", "env", "env_ci", and "baseline".

    Defined properties:
    - t_data: time axis data
    - label: channel label
    - y_units: channel signal units
    - param: dictionary of channel parameters (fs, n_samp)
    - peaks: dictionary of PeaksSet objects.
    """

    _data_fields: ClassVar[tuple[str, ...]] = (
        "raw",
        "filt",
        "clean",
        "env",
        "env_ci",
        "baseline",
    )

    def __init__(
        self,
        y_raw: np.ndarray,
        t_data: np.ndarray | None = None,
        fs: int | None = None,
        label: str | None = None,
        units: str | None = None,
    ):
        """Initialize a TimeSeries object with raw signal data and optional time axis.

        Args:
                y_raw (numpy.ndarray): 1-dimensional raw signal data.
                t_data (numpy.ndarray, optional): Time axis data. If None,
                    generated from fs.
                fs (int, optional): Sampling rate. If None, calculated from
                    t_data.
                label (str, optional): Label of the channel.
                units (str, optional): Channel signal units.
        """
        self.param = {}
        self.param["fs"] = fs
        y_raw = np.array(y_raw)
        data_dims = y_raw.ndim
        if data_dims == _NDIM_VECTOR:
            self.param["n_samp"] = len(y_raw)
        elif data_dims == _NDIM_MATRIX:
            if y_raw.shape[0] < y_raw.shape[1]:
                y_raw = y_raw.T
            self.param["n_samp"] = y_raw.shape[0]
        else:
            msg = "Invalid y_raw dimensions"
            raise ValueError(msg)
        # Signal storage: values can be a single ndarray (signal) or a
        # tuple of two ndarrays for confidence-interval data (lower, upper).
        self._y_data: dict[str, np.ndarray | tuple[np.ndarray, np.ndarray]] = {
            "raw": y_raw,
        }

        self.peaks = {}

        if t_data is None and fs is None:
            self.t_data = np.arange(self.param["n_samp"])
        elif t_data is not None:
            if len(np.array(t_data).shape) > 1:
                raise ValueError
            self.t_data = np.array(t_data)
            if fs is None:
                self.param["fs"] = int(1 / (t_data[1:] - t_data[:-1]))
        else:
            # At this point t_data is None and `fs` must be provided. Use an
            # explicit runtime check (raise) instead of `assert` so linters
            # and optimized Python runs don't skip the check.
            if fs is None:
                msg = "Sampling rate (fs) must be provided when t_data is None."
                raise ValueError(msg)
            # Use numpy arithmetic for efficiency and to produce float times
            self.t_data = np.arange(self.param["n_samp"]) / float(fs)

        self.label = label or ""
        self.y_units = units or "?"

    def __getattr__(self, item: str):
        default_y_data = ["raw", "filt", "clean", "env", "env_ci", "baseline"]
        if item.replace("y_", "") in default_y_data:
            key = item.replace("y_", "")
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(f"""
                The attribute y_{key} is deprecated to allow for more, and
                custom, signal types. Use "{self.__class__.__name__}"[{key}]
                instead.""")
                    )
                ),
                FutureWarning,
            )
            return self._y_data.get(key)
        if item in self.peaks:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(f"""
                The attribute {item} is deprecated to allow for multiple peak
                sets. Use self.peaks["{item}"] instead.""")
                    )
                ),
                FutureWarning,
            )
            return self.peaks[item]
        if item in self.param:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(f"""
                The attribute {item} is deprecated to allow for more parameters.
                Use self.param["{item}"] instead.""")
                    )
                ),
                FutureWarning,
            )
            return self.param[item]
        # `__getattr__` is only called when normal attribute lookup fails.
        # The correct behavior is to raise `AttributeError` to signal the
        # attribute does not exist (rather than calling `object.__getattr__`,
        # which doesn't exist and confuses static analyzers).
        raise AttributeError(item)

    def __getitem__(self, key: str) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Get the signal data of the requested type.

            The default signal types are: "raw", "filt", "clean", "env", "env_ci",
            and "baseline".


        Args:
                key (str): One of "raw", "filt", "clean", "env", "env_ci",
                    "baseline", or a custom signal type.

        Returns:
                numpy.ndarray or tuple: Data of the selected signal type. May be a
                    numpy array or a tuple of two arrays for confidence interval
                    data (lower, upper).
        """
        return self._y_data[key]

    def __setitem__(self, key: str, value: np.ndarray | tuple[np.ndarray, np.ndarray]):
        """Set the signal data of the requested type.

            The default signal types are: "raw", "filt", "clean", "env", "env_ci",
            and "baseline".


        Args:
                key (str): One of "raw", "filt", "clean", "env", "env_ci",
                    "baseline", or a custom signal type.
                value (numpy.ndarray or tuple): Data of the selected signal type.
                    Can be a numpy array or a tuple of two arrays
                    (lower_ci, upper_ci).
        """
        self._y_data[key] = value

    def __iter__(self):
        """Iterate over the signal data types in the _y_data dictionary.

        Returns:
            iterator: Iterator over the signal data types.
        """
        return iter(self._y_data)

    @property
    def y_data(self) -> dict[str, np.ndarray]:
        """Return the dictionary of all signal data types."""
        return {
            field: np.asarray(self._y_data[field])
            for field in self._data_fields
            if field in self._y_data and isinstance(self._y_data[field], np.ndarray)
        }

    def signal_type_data(self, signal_type: str | None = None) -> np.ndarray:
        """Automatic data type selection.

            Automatically select the most advanced data type eligible for a
            subprocess ("env" {=envelope} > "clean" > "filt" > "raw").


        Args:
                signal_type (str, optional): One of "env", "clean", "filt", or
                    "raw".

        Returns:
                numpy.ndarray: Data of the selected signal type.
        """
        # Use direct dict access for the raw signal since `self.__getitem__`
        # can return `None` (static checkers complain about `.shape` on None).

        y_data = np.zeros(self._y_data["raw"].shape if isinstance(self._y_data["raw"], np.ndarray) else (0,))
        res_order = ["env", "clean", "filt", "raw"]
        if signal_type == "env":
            if "env" in self._y_data:
                y_data = self._y_data["env"]
            else:
                msg = "No envelope defined for this signal."
                raise KeyError(msg)
        elif signal_type is None or signal_type in res_order:
            # If the signal type is one of the resolution order, find the data
            # or its next best option
            search_signal_type = signal_type or "env"
            start_idx = res_order.index(search_signal_type)
            _res_order = res_order[start_idx:]
            for idx, option in enumerate(_res_order):
                if option in self._y_data:
                    y_data = self._y_data[option]
                    find_idx = idx
                    break
            else:
                msg = f"Invalid signal type: {signal_type} or no data available for {signal_type}."
                raise KeyError(msg)
            if signal_type is not None and find_idx > 0:
                warnings.warn(
                    "\n".join(
                        wrap(
                            dedent(
                                f"""Warning: No {signal_type} data available, using raw
                     data instead."""
                            )
                        )
                    )
                )
        # If the signal type is not in the resolution order, find the data
        # or its next best option
        elif signal_type in self._y_data:
            y_data = self._y_data[signal_type]
        else:
            msg_0 = f"Invalid signal type: {signal_type}"
            raise KeyError(msg_0)
        return np.asarray(y_data)

    def filter_emg(
        self,
        signal_io: tuple[str, str] = ("raw", "filt"),
        hp_cf: float = 20.0,
        lp_cf: float = 500.0,
        order: int = 3,
        **kwargs,
    ) -> None:
        """Filter raw EMG signal.

            Filter raw EMG signal to remove baseline wander and high frequency
            components. See preprocessing.emg_bandpass_butter submodule.
            The filtered signal is stored in self[signal_io[1]].


        Args:
                signal_io (tuple, optional): Input/output signal type names.
                    Default is ("raw", "filt").
                hp_cf (float, optional): High-pass cutoff frequency in Hz.
                    Default is 20.0.
                lp_cf (float, optional): Low-pass cutoff frequency in Hz.
                    Default is 500.0.
                order (int, optional): Filter order. Default is 3.
                **kwargs: Accepts deprecated ``signal_type`` argument.

        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`, `output_name`)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"], "filt")
        y_data = self.signal_type_data(signal_type=signal_io[0])
        # Eliminate the baseline wander from the data using a band-pass filter
        self[signal_io[1]] = filt.emg_bandpass_butter(
            y_data,
            high_pass=hp_cf,
            low_pass=lp_cf,
            fs_emg=self.param["fs"],
            order=order,
        )

    def get_ecg_peaks(
        self,
        ecg_raw: np.ndarray | None = None,
        bp_filter: bool = True,
        overwrite: bool = True,
        name: str = "ecg",
    ) -> None:
        """Detect ECG peaks in the provided signal.

            See preprocessing.ecg_removal submodule.
            ECG peaks are stored in the self.peaks dict under the key
            `name`. When no ECG channel is provided, the raw signal of the current
            TimeSeries object is used.
            When running get_ecg_peaks on a EmgDataGroup and no ECG channel or
            ecg_raw is provided, EmgDataGroup.ecg_idx is used to detect the QRS
            peak locations. ecg_idx is auto-detected from the labels on
            EmgDataGroup initialization, or can be set with the set_ecg_idx method.


        Args:
                ecg_raw (numpy.ndarray, optional): ECG signal. If None, the raw
                    signal is used.
                bp_filter (bool): Apply band-pass filter to the ECG signal.
                overwrite (bool): Overwrite existing peaks.
                name (str): Name of the peak set in the self.peaks dict.
        """
        if name in self.peaks and not overwrite:
            msg = "ECG peaks already detected. Use overwrite=True"
            raise UserWarning(msg)

        if ecg_raw is None:
            lp_cf = min(500.0, 0.95 * self.param["fs"] / 2)
            ecg_raw = cast(
                "np.ndarray",
                filt.emg_bandpass_butter(self["raw"], high_pass=1, low_pass=lp_cf, fs_emg=self.param["fs"]),
            )

        ecg_peak_idxs = ecg_rm.detect_ecg_peaks(
            ecg_raw=ecg_raw,
            fs=self.param["fs"],
            bp_filter=bp_filter,
        )

        self.set_peaks(
            signal=ecg_raw,
            peak_idxs=ecg_peak_idxs,
            peak_set_name=name,
            overwrite=overwrite,
        )

    def gating(
        self,
        signal_io: tuple[str, str] = ("filt", "clean"),
        ecg_peakset_name: str = "ecg",
        gate_width_samples: int | None = None,
        fill_method: int = 3,
        **kwargs,
    ) -> None:
        """Eliminate ECG artifacts from the provided signal.

            Eliminate ECG artifacts from the provided signal based on the peak_idx
            of the provided PeakSet with `ecg_peakset_name`. See
            preprocessing.ecg_removal and pipelines.ecg_removal_gating submodules.
            The cleaned signal is stored in self["clean"].


        Args:
                signal_io (tuple, optional): Input/output signal type names.
                    Default is ("filt", "clean").
                ecg_peakset_name (str, optional): Key of the ECG PeaksSet in
                    self.peaks. Default is "ecg".
                gate_width_samples (int, optional): Width of the gating window in
                    samples. Defaults to fs // 10.
                fill_method (int, optional): Fill method for gating. Default is 3.
                **kwargs: Accepts deprecated arguments ``signal_type``,
                    ``ecg_peak_idxs``, ``ecg_raw``, ``bp_filter``, ``overwrite``.

        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`, `output_name`)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"], "filt")

        if any(key in kwargs for key in ["ecg_peak_idxs", "ecg_raw", "bp_filter", "overwrite"]):
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent("""
                The kwargs "ecg_peak_idxs", "ecg_raw", "bp_filter", "overwrite"
                will be removed in future versions of ReSurfEMG:
                ECG removal will be split in QRS detection with
                `get_ecg_peaks` and ECG elimination (`gating`/
                `wavelet_denoising`). Alternatively, create an ECG peakset with
                `set_peaks` and use `gating` or `wavelet_denoising` directly.
                """)
                    )
                ),
                FutureWarning,
            )
            ecg_peak_idxs = kwargs.get("ecg_peak_idxs")
            bp_filter = kwargs.get("bp_filter", True)
            overwrite = kwargs.get("overwrite", False)
            if ecg_peak_idxs is None:
                ecg_raw = kwargs.get("ecg_raw")
                self.get_ecg_peaks(
                    ecg_raw=ecg_raw,
                    bp_filter=bp_filter,
                    overwrite=overwrite,
                    name=ecg_peakset_name,
                )
            else:
                if ecg_peakset_name not in self.peaks and not overwrite:
                    msg = "ECG peaks already detected. Use overwrite=True"
                    raise UserWarning(msg)
                ecg_raw = kwargs.get("ecg_raw", self.signal_type_data(signal_type="raw"))
                self.set_peaks(
                    signal=ecg_raw,
                    peak_idxs=ecg_peak_idxs,
                    peak_set_name=ecg_peakset_name,
                    overwrite=overwrite,
                )
        if ecg_peakset_name not in self.peaks:
            msg = f"""Peakset {ecg_peakset_name} does not yet exist. First detect
                 ECG peaks before using gating."""
            raise KeyError(msg)
        y_data = self.signal_type_data(signal_type=signal_io[0])
        ecg_peak_idxs = self.peaks[ecg_peakset_name]["peak_idx"]
        if gate_width_samples is None:
            gate_width_samples = self.param["fs"] // 10
        self[signal_io[1]] = ecg_removal_gating(
            y_data,
            ecg_peak_idxs,
            cast("int", gate_width_samples),
            ecg_shift=10,
            method=fill_method,
        )

    def wavelet_denoising(
        self,
        signal_io: tuple[str, str] = ("filt", "clean"),
        ecg_peakset_name: str = "ecg",
        n: int | None = None,
        fixed_threshold: float | None = None,
        **kwargs,
    ) -> None:
        """Eliminate ECG artifacts from the provided signal.

            See preprocessing.wavelet_denoising submodules. The cleaned signal is
            by default stored in self["clean"].


        Args:
                signal_io (tuple, optional): Input/output signal type names.
                    Default is ("filt", "clean").
                ecg_peakset_name (str, optional): Key of the ECG PeaksSet in
                    self.peaks. Default is "ecg".
                n (int, optional): Number of wavelet decomposition levels.
                    Defaults to ``int(log(fs/20) // log(2))``.
                fixed_threshold (float, optional): Threshold for wavelet
                    denoising. Default is 4.5.
                **kwargs: Accepts deprecated arguments ``signal_type``,
                    ``ecg_peak_idxs``, ``ecg_raw``, ``bp_filter``, ``overwrite``.

        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`, `output_name`)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"], "filt")

        if any(key in kwargs for key in ["ecg_peak_idxs", "ecg_raw", "bp_filter", "overwrite"]):
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent("""
                The kwargs "ecg_peak_idxs", "ecg_raw", "bp_filter", "overwrite"
                will be removed in future versions of ReSurfEMG.
                ECG removal will be split in QRS detection with
                `get_ecg_peaks` and ECG elimination (`gating`/
                `wavelet_denoising`). Alternatively, create an ECG peakset with
                `set_peaks` and use `gating` or `wavelet_denoising` directly.
                """)
                    )
                ),
                FutureWarning,
            )
            ecg_peak_idxs = kwargs.get("ecg_peak_idxs")
            bp_filter = kwargs.get("bp_filter", True)
            overwrite = kwargs.get("overwrite", False)
            if ecg_peak_idxs is None:
                ecg_raw = kwargs.get("ecg_raw")
                self.get_ecg_peaks(
                    ecg_raw=ecg_raw,
                    bp_filter=bp_filter,
                    overwrite=overwrite,
                    name=ecg_peakset_name,
                )
            else:
                if ecg_peakset_name not in self.peaks and not overwrite:
                    msg = "ECG peaks already detected. Use overwrite=True"
                    raise UserWarning(msg)
                ecg_raw = kwargs.get("ecg_raw", self.signal_type_data(signal_type="raw"))
                self.set_peaks(
                    signal=ecg_raw,
                    peak_idxs=ecg_peak_idxs,
                    peak_set_name=ecg_peakset_name,
                    overwrite=overwrite,
                )
        if ecg_peakset_name not in self.peaks:
            msg = f"""ECG peakset {ecg_peakset_name}. First detect ECG peaks
                before using wavelet denoising."""
            raise KeyError(msg)
        y_data = self.signal_type_data(signal_type=signal_io[0])
        ecg_peak_idxs = self.peaks["ecg"]["peak_idx"]
        if n is None:
            n = int(np.log(self.param["fs"] / 20) // np.log(2))

        if fixed_threshold is None:
            fixed_threshold = 4.5

        self[signal_io[1]], *_ = ecg_rm.wavelet_denoising(
            y_data,
            ecg_peak_idxs,
            fs=self.param["fs"],
            hard_thresholding=True,
            n=n,
            fixed_threshold=fixed_threshold,
            wavelet_type="db2",
        )

    def envelope(
        self,
        env_window: int | None = None,
        env_type: str | None = None,
        signal_io: tuple[str, str] = ("clean", "env"),
        ci_alpha: float | None = None,
        **kwargs,
    ) -> None:
        """Derive the moving envelope of the provided signal.

            See preprocessing.envelope submodule. The envelope is by default stored
            in self["env"]. If ci_alpha is not None, the confidence interval is
            stored in self["env_ci"].


        Args:
                env_window (int, optional): Envelope window length in samples.
                    Defaults to fs // 4.
                env_type (str, optional): Envelope type, "rms" or "arv".
                    Defaults to "rms".
                signal_io (tuple, optional): Input/output signal type names.
                    Default is ("clean", "env").
                ci_alpha (float, optional): Confidence interval alpha level. If
                    None, no confidence interval is computed.
                **kwargs: Accepts deprecated ``signal_type`` argument.
        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`, `output_name`)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"], "env")
        if env_window is None:
            fs = self.param.get("fs")
            if fs is None:
                msg = "Envelope window and sampling rate are not defined."
                raise ValueError(msg)
            env_window = int(fs) // 4

        y_data = self.signal_type_data(signal_type=signal_io[0])
        if env_type == "rms" or env_type is None:
            self[signal_io[1]] = evl.full_rolling_rms(y_data, env_window)
            if ci_alpha is not None:
                self[signal_io[1] + "_ci"] = evl.rolling_rms_ci(y_data, env_window, alpha=ci_alpha)
        elif env_type == "arv":
            self[signal_io[1]] = evl.full_rolling_arv(y_data, env_window)
            if ci_alpha is not None:
                self[signal_io[1] + "_ci"] = evl.rolling_arv_ci(y_data, env_window, alpha=ci_alpha)
        else:
            msg = "Invalid envelope type"
            raise ValueError(msg)

    def baseline(
        self,
        percentile: int = 33,
        window_s: int | None = None,
        step_s: int | None = None,
        base_method: str = "default",
        signal_io: tuple[str | None, str] = (None, "baseline"),
        augm_percentile: int = 25,
        ma_window: int | None = None,
        perc_window: int | None = None,
        **kwargs,
    ) -> None:
        """Derive the moving baseline of the provided signal.

            See postprocessing.baseline submodule. The baseline is stored in
            self["baseline"].


        Args:
                percentile (int, optional): Percentile used for the moving
                    baseline. Default is 33.
                window_s (int, optional): Window length in samples. Defaults to
                    7.5 * fs.
                step_s (int, optional): Step size in samples. Defaults to
                    fs // 5.
                base_method (str, optional): Baseline method, "default"/
                    "moving_baseline" or "slopesum_baseline". Default is
                    "default".
                signal_io (tuple, optional): Input/output signal type names.
                    Default is (None, "baseline"), where None resolves to "env".
                augm_percentile (int, optional): Augmented percentile for
                    slopesum_baseline. Default is 25.
                ma_window (int, optional): Moving average window for
                    slopesum_baseline.
                perc_window (int, optional): Percentile window for
                    slopesum_baseline.
                **kwargs: Accepts deprecated ``signal_type`` argument.
        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`, `output_name`)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"], "baseline")
        window_s = window_s or int(7.5 * self.param.get("fs", 1))
        step_s = step_s or self.param.get("fs", 5) // 5
        if signal_io[0] is None:
            signal_io = ("env", "baseline")

        y_baseline_data = self.signal_type_data(signal_type=signal_io[0])
        if base_method in ("default", "moving_baseline"):
            self[signal_io[1]] = bl.moving_baseline(
                y_baseline_data,
                window_s=window_s,
                step_s=step_s,
                set_percentile=percentile,
            )
        elif base_method == "slopesum_baseline":
            if "fs" not in self.param:
                msg = "Sampling rate is not defined."
                raise ValueError(msg)
            self[signal_io[1]], _, _, _ = bl.slopesum_baseline(
                y_baseline_data,
                window_s=window_s,
                step_s=step_s,
                fs=self.param["fs"],
                set_percentile=percentile,
                augm_percentile=augm_percentile,
                ma_window=ma_window,
                perc_window=perc_window,
            )
        else:
            msg = "Invalid method"
            raise ValueError(msg)

    def set_peaks(
        self,
        peak_idxs: np.ndarray,
        signal: np.ndarray | None,
        peak_set_name: str,
        overwrite: bool = True,
    ) -> None:
        """Store a new PeaksSet object.

            Store a new PeaksSet object in the self.peaks dict under the key
            peak_set_name.


        Args:
                peak_idxs (list[int] or numpy.ndarray): Peak indices.
                signal (numpy.ndarray or None): Signal underlying the peaks.
                peak_set_name (str): Key under which to store the PeaksSet.
                overwrite (bool): Overwrite an existing PeaksSet. Default is
                    False.
        """
        if peak_set_name in self.peaks and not overwrite:
            msg = "PeaksSet already exists. Use overwrite=True"
            raise UserWarning(msg)
        self.peaks[peak_set_name] = PeaksSet(peak_idxs=peak_idxs, t_data=self.t_data, signal=signal)

    def detect_emg_breaths(
        self,
        threshold: int = 0,
        prominence_factor: float = 0.5,
        min_peak_width_s: int | None = None,
        peak_set_name: str = "breaths",
        start_idx: int = 0,
        end_idx: int | None = None,
        overwrite: bool = True,
        signal_io: tuple[tuple[str | None, str], ...] = (("env", "baseline"),),
    ) -> None:
        """Find breath peaks in provided EMG envelope signal.

            See postprocessing.event_detection submodule. The peaks are stored in
            self.peaks under peak_set_name.


        Args:
                threshold (int, optional): Minimum peak height above baseline.
                    Default is 0.
                prominence_factor (float, optional): Minimum peak prominence as a
                    fraction of signal range. Default is 0.5.
                min_peak_width_s (int, optional): Minimum peak width in samples.
                    Defaults to fs // 5.
                peak_set_name (str, optional): Key under which to store the
                    PeaksSet. Default is "breaths".
                start_idx (int, optional): Start index for peak detection.
                    Default is 0.
                end_idx (int, optional): End index for peak detection. Defaults
                    to the signal length.
                overwrite (bool): Overwrite an existing PeaksSet. Default is
                    False.
                signal_io (tuple, optional): Tuple of (input_signal_key,
                    baseline_key). Default is (("env", "baseline"),).
        """
        if signal_io[0][0] not in self._y_data:
            msg = f"Envelope ({signal_io[0][0]}) not yet defined."
            raise ValueError(msg)

        signal = np.asarray(self[signal_io[0][0]])
        y_baseline = np.asarray(self[signal_io[0][1]]) if signal_io[0][1] in self._y_data else np.zeros(signal.shape)
        if signal_io[0][1] not in self._y_data:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent("""EMG baseline ({signal_io[0][1]}) not yet defined. Peak
                detection relative to zero.""")
                    )
                )
            )

        if (end_idx is not None and end_idx > len(signal)) or start_idx > len(signal):
            msg = "Index out of range."
            raise ValueError(msg)

        end_idx = end_idx or len(signal)
        if end_idx < start_idx:
            msg = "End index smaller than start index."
            raise ValueError(msg)

        min_peak_width_s = min_peak_width_s or self.param["fs"] // 5

        peak_idxs = evt.detect_emg_breaths(
            signal[start_idx:end_idx],
            y_baseline[start_idx:end_idx],
            threshold=threshold,
            prominence_factor=prominence_factor,
            min_peak_width_s=min_peak_width_s,
        )
        peak_idxs = np.asarray(peak_idxs) + start_idx
        self.set_peaks(
            peak_idxs=peak_idxs,
            signal=signal,
            peak_set_name=peak_set_name,
            overwrite=overwrite,
        )

    def link_peak_set(
        self,
        peak_set_name: str,
        t_reference_peaks: np.ndarray,
        linked_peak_set_name: str | None = None,
    ) -> None:
        """Find the peaks closest in time to the provided peaks.

            Find the peaks in the PeaksSet with the peak_set_name closest in time
            to the provided peak timings in t_reference_peaks.
            The results are
            stored in a new PeaksSet object in the self.peaks dict under the key
            linked_peak_set_name. If no linked_peak_set_name is provided, the key
            is set to peak_set_name + "_linked".


        Args:
                peak_set_name (str): PeaksSet name in self.peaks dict.
                t_reference_peaks (numpy.ndarray): Reference peak timings.
                linked_peak_set_name (str, optional): Name of the new PeaksSet.
                    Defaults to peak_set_name + "_linked".
        """
        if peak_set_name not in self.peaks:
            msg = "Non-existent PeaksSet key"
            raise KeyError(msg)

        peak_set = self.peaks[peak_set_name]
        linked_peak_set_name = linked_peak_set_name or peak_set_name + "_linked"
        t_peakset_peaks = peak_set["peak_idx"] / self.param["fs"]
        link_peak_nrs = evt.find_linked_peaks(t_reference_peaks, t_peakset_peaks)

        self.peaks[linked_peak_set_name] = PeaksSet(peak_set.signal, peak_set.t_data, peak_idxs=None)
        for attr in ["peak_df", "quality_values_df", "quality_outcomes_df"]:
            setattr(
                self.peaks[linked_peak_set_name],
                attr,
                getattr(peak_set, attr).loc[link_peak_nrs].reset_index(drop=True),
            )

    def calculate_time_products(
        self,
        peak_set_name: str,
        include_aub: bool = True,
        aub_window_s: int | None = None,
        aub_reference_signal: np.ndarray | tuple[np.ndarray, np.ndarray] | None = None,
        parameter_name: str | None = None,
        signal_io: tuple[str, ...] = ("baseline",),
    ) -> None:
        """Calculate the time product, i.e. area under the curve for a PeaksSet.

            The results are stored as
            self.peaks[peak_set_name].peak_df[parameter_name]. If no parameter_name
            is provided, parameter_name = "time_product".


        Args:
                peak_set_name (str): PeaksSet name in self.peaks dict.
                include_aub (bool): Include the area under the baseline in the
                    time product. Default is True.
                aub_window_s (int, optional): Window length in samples for
                    finding the local extreme. Defaults to 5 * fs.
                aub_reference_signal (numpy.ndarray, optional): Reference signal
                    for the local extreme. If None, the PeaksSet signal is used.
                parameter_name (str, optional): Column name in
                    self.peaks[peak_set_name].peak_df. Defaults to
                    "time_product".
                signal_io (tuple, optional): Tuple where the first element is the
                    baseline signal key. Default is ("baseline",).
        """
        peak_set = self._check_peak_set(self.peaks.get(peak_set_name))
        baseline: np.ndarray

        if signal_io[0] not in self._y_data:
            if include_aub:
                msg_0 = "Baseline in not yet defined, but is required to calculate the area under the baseline."
                raise ValueError(msg_0)
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent("""Baseline in not yet defined. Calculating time-product
                    with reference to 0.""")
                    )
                )
            )
            baseline = np.zeros(peak_set.signal.shape)
        else:
            baseline = np.asarray(self[signal_io[0]])

        if "start_idx" not in peak_set.peak_df or "end_idx" not in peak_set.peak_df:
            msg_1 = "PeakSet does not contain start_idx and end_idx columns."
            raise ValueError(msg_1)

        time_products = feat.time_product(
            signal=peak_set.signal,
            fs=self.param["fs"],
            start_idxs=peak_set["start_idx"],
            end_idxs=peak_set["end_idx"],
            baseline=baseline,
        )

        if include_aub:
            aub_window_s = aub_window_s or 5 * self.param["fs"]
            aub_reference_signal = np.asarray(peak_set.signal if aub_reference_signal is None else aub_reference_signal)
            aub, y_refs = feat.area_under_baseline(
                signal=peak_set.signal,
                fs=self.param["fs"],
                start_idxs=peak_set["start_idx"],
                peak_idxs=peak_set["peak_idx"],
                end_idxs=peak_set["end_idx"],
                aub_window_s=aub_window_s,
                baseline=baseline,
                ref_signal=aub_reference_signal,
            )
            peak_set.peak_df["AUB"] = aub
            peak_set.peak_df["aub_y_ref"] = y_refs
            time_products += aub

        peak_set.peak_df[parameter_name or "time_product"] = time_products

    def test_emg_quality(
        self,
        peak_set_name: str,
        cutoff: dict | None = None,
        skip_tests: list[str] | None = None,
        parameter_names: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> None:
        """Test EMG PeaksSet according to quality criteria in Warnaar et al. (2024).

        See helper_functions.data_classes_quality_assessment submodule. The
        results are stored in the self.peaks[peak_set_name].quality_outcomes_df
        and self.peaks[peak_set_name].quality_values_df DataFrames.
        """
        data_qa.test_emg_quality(self, peak_set_name, cutoff, skip_tests, parameter_names, verbose)

    def test_pocc_quality(
        self,
        peak_set_name: str,
        cutoff: dict | None = None,
        skip_tests: list[str] | None = None,
        parameter_names: dict[str, str] | None = None,
        verbose: bool = True,
    ) -> None:
        """Test EMG PeaksSet according to quality criteria in Warnaar et al. (2024).

        See helper_functions.data_classes_quality_assessment submodule. The
        results are stored in the self.peaks[peak_set_name].quality_outcomes_df
        and self.peaks[peak_set_name].quality_values_df DataFrames.
        """
        data_qa.test_pocc_quality(self, peak_set_name, cutoff, skip_tests, parameter_names, verbose)

    def test_linked_peak_sets(
        self,
        peak_set_name: str,
        linked_timeseries: TimeSeries,
        linked_peak_set_name: str,
        parameter_names: dict[str, str] | None = None,
        cutoff: dict[str, float] | None = None,
        skip_tests: list[str] | None = None,
        verbose: bool = True,
    ) -> None:
        """Test number of detected breaths in the native PeaksSet.

        See helper_functions.data_classes_quality_assessment submodule. The
        results are stored in the self.peaks[peak_set_name].quality_outcomes_df
        and self.peaks[peak_set_name].quality_values_df DataFrames.
        """
        data_qa.test_linked_peak_sets(
            self,
            peak_set_name,
            linked_timeseries,
            linked_peak_set_name,
            parameter_names,
            cutoff,
            skip_tests,
            verbose,
        )

    def plot_full(
        self,
        axes: Axes | None = None,
        signal_io: tuple[str | None, ...] = (None,),
        colors: list[str] | None = None,
        baseline_bool: bool = True,
        plot_ci: bool = False,
        **kwargs,
    ) -> None:
        """Plot the indicated signals in the provided axes.

            By default the most
            advanced signal type (envelope > clean > filt > raw) is plotted in the
            provided colours.


        Args:
                axes (matplotlib.Axes, optional): Matplotlib Axes object. If None,
                    a new figure is created.
                signal_io (tuple, optional): Tuple where the first element is the
                    input signal type. Default is (None,), which resolves to the
                    most advanced available type.
                colors (list, optional): Colors for 1) the signal, 2) the
                    baseline.
                baseline_bool (bool): Plot the baseline. Default is True.
                plot_ci (bool): Plot the confidence interval of the envelope.
                    Default is False.
                **kwargs: Accepts deprecated ``signal_type`` argument.
        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`,)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"], "env")
        axis = axes if axes is not None else plt.subplots()[1]
        colors = colors if colors is not None else ["tab:blue", "tab:orange", "tab:red", "tab:cyan", "tab:green"]

        y_data = self.signal_type_data(signal_type=signal_io[0])
        axis.grid(True)
        axis.plot(self.t_data, y_data, color=colors[0])
        axis.set_ylabel(self.label + " (" + self.y_units + ")")

        baseline = self._y_data.get("baseline")
        env_ci = self._y_data.get("env_ci")
        if baseline_bool is True and baseline is not None and np.any(~np.isnan(baseline), axis=0):
            axis.plot(self.t_data, baseline, color=colors[1])
        if plot_ci and env_ci is not None:
            axis.fill_between(
                self.t_data,
                env_ci[0],
                env_ci[1],
                color=colors[0],
                alpha=0.5,
            )

    def plot_markers(
        self,
        peak_set_name: str,
        axes: Axes | np.ndarray | None = None,
        valid_only: bool = False,
        colors: str | list[str] | None = None,
        markers: str | list[str] | None = None,
    ) -> None:
        """Plot the markers for the peak set.

            Plot the markers for the peak set in the provided axes in the
            provided colours using the provided markers.


        Args:
                peak_set_name (str): PeaksSet name in self.peaks dict.
                axes (matplotlib.Axes or numpy.ndarray, optional): Matplotlib
                    Axes object. If None, a new figure is created.
                valid_only (bool): When True, only valid peaks are plotted.
                    Default is False.
                colors (str or list, optional): One color or list of up to 3
                    colors for peak, start, and end markers. If 2 colors are
                    provided, start and end share the same color.
                markers (str or list, optional): One marker or list of up to 3
                    markers for peak, start, and end. If 2 markers are provided,
                    start and end share the same marker.
        """
        if peak_set_name not in self.peaks:
            msg = "Non-existent PeaksSet key"
            raise KeyError(msg)

        peak_set = self.peaks[peak_set_name]
        valid = (
            peak_set["valid"]
            if valid_only and "valid" in peak_set.peak_df.columns
            else np.ones(len(peak_set["peak_idx"]), dtype=bool)
        )

        def get_values(
            column: str,
        ) -> tuple[np.ndarray, np.ndarray] | tuple[list[None], list[None]]:
            return (
                (
                    peak_set.t_data[peak_set[column]][valid],
                    peak_set.signal[peak_set[column]][valid],
                )
                if column in peak_set.peak_df.columns
                else (len(peak_set) * [None], len(peak_set) * [None])
            )

        x_vals_peak, y_vals_peak = get_values("peak_idx")
        x_vals_start, y_vals_start = get_values("start_idx")
        x_vals_end, y_vals_end = get_values("end_idx")

        def get_color_marker(values: str | list[str] | None, default: str) -> list[str]:
            if values is None:
                values = [default]
            elif isinstance(values, str):
                values = [values]
            return (values + [values[-1]] * 2)[:3]

        peak_color, start_color, end_color = get_color_marker(colors, "tab:red")
        peak_marker, start_marker, end_marker = get_color_marker(markers, "*")
        axes_arr = np.atleast_1d(cast("Any", axes))
        if len(axes_arr) == 1 and len(x_vals_peak) > 1:
            axes = np.matlib.repmat(cast("Any", axes), len(x_vals_peak), 1).flatten()
        for axis, x_peak, y_peak, x_start, y_start, x_end, y_end in zip(
            axes_arr, x_vals_peak, y_vals_peak, x_vals_start, y_vals_start, x_vals_end, y_vals_end, strict=False
        ):
            axis.plot(x_peak, y_peak, marker=peak_marker, color=peak_color, linestyle="None")
            if x_start is not None:
                axis.plot(
                    x_start,
                    y_start,
                    marker=start_marker,
                    color=start_color,
                    linestyle="None",
                )
            if x_end is not None:
                axis.plot(x_end, y_end, marker=end_marker, color=end_color, linestyle="None")

    def plot_peaks(
        self,
        peak_set_name: str,
        axes: Axes | np.ndarray | None = None,
        signal_io: tuple[str | None, ...] = (None,),
        margin_s: int | None = None,
        valid_only: bool = False,
        colors: list[str] | None = None,
        baseline_bool: bool = True,
        plot_ci: bool = False,
        **kwargs,
    ) -> None:
        """Plot the indicated peaks in the provided axes.

            By default the most advanced signal type (envelope > clean > filt > raw)
            is plotted in the provided colours.


        Args:
                peak_set_name (str): The name of the peak set to be plotted.
                axes (matplotlib.Axes or numpy.ndarray, optional): Matplotlib
                    Axes object. If None, a new figure is created.
                signal_io (tuple, optional): Tuple where the first element is the
                    input signal type. Default is (None,).
                margin_s (int, optional): Margins in samples before peak onset
                    and after peak offset. Defaults to fs // 2.
                valid_only (bool): When True, only valid peaks are plotted.
                    Default is False.
                colors (list, optional): Colors for 1) the signal, 2) the
                    baseline.
                baseline_bool (bool): Plot the baseline. Default is True.
                plot_ci (bool): Plot the confidence interval of the envelope.
                    Default is False.
                **kwargs: Accepts deprecated ``signal_type`` argument.
        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`,)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"],)
        peak_set = self._check_peak_set(self.peaks.get(peak_set_name))

        start_idxs, end_idxs = peak_set["start_idx"], peak_set["end_idx"]
        if valid_only and "valid" in peak_set.peak_df.columns:
            valid = peak_set["valid"]
            start_idxs, end_idxs = start_idxs[valid], end_idxs[valid]

        if axes is None:
            _, axes = plt.subplots(nrows=1, ncols=len(start_idxs), sharey=True)
        axes = np.asarray(np.atleast_1d(cast("Any", axes)))
        colors = colors if colors is not None else ["tab:blue", "tab:orange", "tab:red", "tab:cyan", "tab:green"]
        y_data = peak_set.signal if signal_io[0] is None else self.signal_type_data(signal_type=signal_io[0])
        m_s = margin_s if margin_s is not None else self.param["fs"] // 2
        ci = self._y_data.get("env_ci")
        baseline = self._y_data.get("baseline")
        for axis, x_start, x_end in zip(axes, start_idxs, end_idxs, strict=False):
            s_start, s_end = max(0, x_start - m_s), max(0, x_end + m_s)
            axis.grid(True)
            axis.plot(self.t_data[s_start:s_end], y_data[s_start:s_end], color=colors[0])
            if baseline_bool and baseline is not None and np.any(~np.isnan(baseline), axis=0):
                axis.plot(
                    self.t_data[s_start:s_end],
                    baseline[s_start:s_end],
                    color=colors[1],
                )
            if plot_ci and ci is not None:
                axis.fill_between(
                    self.t_data[s_start:s_end],
                    ci[0][s_start:s_end],
                    ci[1][s_start:s_end],
                    color=colors[0],
                    alpha=0.5,
                )

        axes[0].set_ylabel(f"{self.label} ({self.y_units})")

    def plot_curve_fits(
        self,
        peak_set_name: str,
        axes: Axes | np.ndarray | None,
        valid_only: bool = False,
        colors: list[str] | None = None,
    ) -> None:
        """Plot the curve-fits for the peak set.

            Plot the curve-fits for the peak set in the provided axes in the
            provided colours.


        Args:
                peak_set_name (str): PeaksSet name in self.peaks dict.
                axes (matplotlib.Axes or numpy.ndarray): Matplotlib Axes object.
                valid_only (bool): When True, only valid peaks are plotted.
                    Default is False.
                colors (str or list, optional): One color or list of colors for
                    the fitted curve.
        """
        peak_set = self._check_peak_set(self.peaks.get(peak_set_name))

        axes = np.asarray(np.atleast_1d(cast("Any", axes)))
        required_params = ["y_min", "a", "b", "c"]
        missing_params = [param for param in required_params if f"bell_{param}" not in peak_set.peak_df.columns]
        if missing_params:
            msg = f"Missing parameters in PeaksSet: {', '.join(missing_params)}"
            raise KeyError(msg)

        plot_peak_df = (
            peak_set.peak_df.loc[peak_set.peak_df["valid"]]
            if valid_only and "valid" in peak_set.peak_df.columns
            else peak_set.peak_df
        )
        color = colors[0] if isinstance(colors, list) and colors else "tab:green"

        for axis, (_, row) in zip(axes, plot_peak_df.iterrows(), strict=False):
            y_bell = mo.bell_curve(
                peak_set.t_data[row.start_idx : row.end_idx],
                a=row.bell_a,
                b=row.bell_b,
                c=row.bell_c,
            )
            axis.plot(
                peak_set.t_data[row.start_idx : row.end_idx],
                row.bell_y_min + y_bell,
                color=color,
            )

        if len(axes) > 1:
            for _, (axis, (_, row)) in enumerate(zip(axes, plot_peak_df.iterrows(), strict=False)):
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx : row.end_idx],
                    a=row.bell_a,
                    b=row.bell_b,
                    c=row.bell_c,
                )
                axis.plot(
                    peak_set.t_data[row.start_idx : row.end_idx],
                    row.bell_y_min + y_bell,
                    color=color,
                )
        else:
            for _, row in plot_peak_df.iterrows():
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx : row.end_idx],
                    a=row.bell_a,
                    b=row.bell_b,
                    c=row.bell_c,
                )
                axes[0].plot(
                    peak_set.t_data[row.start_idx : row.end_idx],
                    row.bell_y_min + y_bell,
                    color=color,
                )

    def plot_aub(
        self,
        peak_set_name: str,
        axes: Axes | np.ndarray | None,
        signal_io: tuple[str, ...] | None = None,
        valid_only: bool = False,
        colors: list[str] | None = None,
        **kwargs,
    ) -> None:
        """Plot the area under the baseline (AUB) for the peak set.

            Plot the area under the baseline (AUB) for the peak set in the provided
            axes in the provided colours.


        Args:
                peak_set_name (str): PeaksSet name in self.peaks dict.
                axes (matplotlib.Axes or numpy.ndarray): Matplotlib Axes object.
                signal_io (tuple, optional): Tuple where the first element is the
                    input signal type.
                valid_only (bool): When True, only valid peaks are plotted.
                    Default is False.
                colors (str or list, optional): One color or list of up to 3
                    colors for the markers.
                **kwargs: Accepts deprecated ``signal_type`` argument.
        """
        if "signal_type" in kwargs:
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent(
                            """The "signal_type" argument is deprecated. Use "signal_io"
                instead: signal_io = (`input_name`,)."""
                        )
                    )
                ),
                FutureWarning,
            )
            signal_io = (kwargs["signal_type"],)
        elif signal_io is None:
            msg = "Signal type not provided. Use signal_io."
            raise ValueError(msg)
        peak_set = self.peaks.get(peak_set_name)
        peak_set = self._check_peak_set(peak_set)

        axes = np.asarray(np.atleast_1d(cast("Any", axes)))

        if "aub_y_ref" not in peak_set.peak_df.columns:
            msg = "aub_y_ref not included in PeaksSet, area under the baseline is not evaluated yet."
            raise KeyError(msg)

        y_data = peak_set.signal if signal_io[0] is None else self.signal_type_data(signal_type=signal_io[0])

        plot_peak_df = (
            peak_set.peak_df.loc[peak_set.peak_df["valid"]]
            if valid_only and "valid" in peak_set.peak_df.columns
            else peak_set.peak_df
        )

        color = colors if isinstance(colors, str) else colors[0] if isinstance(colors, list) and colors else "tab:cyan"

        if len(axes) > 1:
            for _, (axis, (_, row)) in enumerate(zip(axes, plot_peak_df.iterrows(), strict=False)):
                axis.plot(
                    peak_set.t_data[[row.start_idx, row.end_idx]],
                    [row.aub_y_ref, row.aub_y_ref],
                    color=color,
                )
                axis.plot(
                    peak_set.t_data[[row.start_idx, row.start_idx]],
                    [y_data[row.start_idx], row.aub_y_ref],
                    color=color,
                )
                axis.plot(
                    peak_set.t_data[[row.end_idx, row.end_idx]],
                    [y_data[row.end_idx], row.aub_y_ref],
                    color=color,
                )
        else:
            for _, row in plot_peak_df.iterrows():
                axes[0].plot(
                    peak_set.t_data[[row.start_idx, row.end_idx]],
                    [row.aub_y_ref, row.aub_y_ref],
                    color=color,
                )
                axes[0].plot(
                    peak_set.t_data[[row.start_idx, row.start_idx]],
                    [y_data[row.start_idx], row.aub_y_ref],
                    color=color,
                )
                axes[0].plot(
                    peak_set.t_data[[row.end_idx, row.end_idx]],
                    [y_data[row.end_idx], row.aub_y_ref],
                    color=color,
                )

    def _check_peak_set(self, peak_set: PeaksSet | None) -> PeaksSet:
        """Check if the provided peak set is valid and return it."""
        if peak_set is None:
            msg = "Non-existent PeaksSet key"
            raise KeyError(msg)
        if not isinstance(peak_set, PeaksSet):
            msg = "Object under provided key is not a PeaksSet"
            raise TypeError(msg)
        if peak_set.peak_df.empty:
            msg = "PeaksSet is empty"
            raise ValueError(msg)
        return peak_set


class TimeSeriesGroup:
    """Data class to store, process, and plot time series data.

    TimeSeriesGroup is a collection of TimeSeries objects. Enclosed TimeSeries objects can be
    indexed by index number or channel label.
    TimeSeries methods in TimeSeriesGroup._available_methods can be run on all
    or a subset of the TimeSeries objects through the TimeSeriesGroup.run
    method.
    """

    _available_methods: ClassVar[list[str]] = [
        "envelope",
        "baseline",
        "plot_full",
        "plot_peaks",
        "plot_markers",
        "set_peaks",
    ]

    @staticmethod
    def _resolve_dims(y_raw: np.ndarray) -> tuple[np.ndarray, int, int]:
        """Resolve the dimensions of the raw data array."""
        arr = np.array(y_raw)
        data_shape = list(arr.shape)
        if len(data_shape) == _NDIM_VECTOR:
            n_samp, n_channel = len(arr), 1
            arr = arr.reshape((1, n_samp))
        elif len(data_shape) == _NDIM_MATRIX:
            n_samp = data_shape[int(np.argmax(data_shape))]
            n_channel = data_shape[int(np.argmin(data_shape))]
            arr = arr if np.argmin(data_shape) == 0 else arr.T
        else:
            msg = "Invalid data dimensions"
            raise ValueError(msg)
        return arr, n_samp, n_channel

    @staticmethod
    def _resolve_time(t_data: np.ndarray | None, fs: int | None, n_samp: int) -> tuple[np.ndarray, int]:
        """Resolve the time axis and sampling rate from the provided time data and sampling rate."""
        if t_data is None and fs is None:
            msg = "Either time data (t_data) or sampling rate (fs) must be provided."
            raise ValueError(msg)
        if t_data is not None:
            t_arr = np.array(t_data)
            if t_arr.ndim > 1:
                msg = "Invalid time data dimensions"
                raise ValueError(msg)
            if fs is None:
                fs = int(1 / (t_arr[1:] - t_arr[:-1]))
            return t_arr, fs
        if fs is None:
            msg = "Sampling rate (fs) must be provided when t_data is None."
            raise ValueError(msg)
        return np.arange(n_samp) / float(fs), fs

    @staticmethod
    def _resolve_labels_units(labels: list[str] | None, units: list[str] | None, n_channel: int) -> tuple[list, list]:
        """Resolve the labels and units for the channels from the provided labels and units lists."""
        if labels is None:
            out_labels = n_channel * [None]
        elif len(labels) != n_channel:
            msg = "Number of labels does not match the number of data channels."
            raise ValueError(msg)
        else:
            out_labels = labels
        if units is None:
            out_units = n_channel * ["N/A"]
        elif len(units) != n_channel:
            msg = "Number of units does not match the number of data channels."
            raise ValueError(msg)
        else:
            out_units = units
        return out_labels, out_units

    def _resolve_channels(
        self,
        channels: int | str | list | np.ndarray | None,
        *,
        default: list | None = None,
    ) -> list:
        """Normalize a channel specifier into a list of validated keys.

        Args:
            channels: A single channel key (int or str), a list/array of
                keys, or None. When None, ``default`` is used; if ``default``
                is also None, all channels are selected.
            default: Channel keys to fall back to when ``channels`` is None.

        Returns:
            list: Validated channel keys.

        Raises:
            ValueError: If ``channels`` has an unsupported type or contains a
                key that is not a valid channel.
        """
        if channels is None:
            if default is None:
                # all channels are valid by construction; skip per-key checks
                return list(range(self.param["n_channel"]))
            keys = list(default)
        elif isinstance(channels, (int, np.integer, str)):
            keys = [channels]
        elif isinstance(channels, (list, np.ndarray)):
            keys = list(channels)
        else:
            msg = "channels must be an int, str, list, array, or None"
            raise ValueError(msg)

        key = ""
        try:
            for key in keys:
                self[key]
        except (KeyError, IndexError) as e:
            msg = f"{key!r} is not a valid channel."
            raise ValueError(msg) from e
        return keys

    if TYPE_CHECKING:

        def envelope(
            self,
            *,
            channel_idxs: list[int] | np.ndarray | None = None,
            env_window: int | None = None,
            env_type: str | None = None,
            signal_io: tuple[str, str] = ("clean", "env"),
            ci_alpha: float | None = None,
            **kwargs,
        ) -> None:
            """Calculate the envelope of the indicated channels."""

        ...

        def baseline(
            self,
            *,
            channel_idxs: list[int] | np.ndarray | None = None,
            percentile: int = 33,
            window_s: int | None = None,
            step_s: int | None = None,
            base_method: str = "default",
            signal_io: tuple[str | None, str] = (None, "baseline"),
            augm_percentile: int = 25,
            ma_window: int | None = None,
            perc_window: int | None = None,
            **kwargs,
        ) -> None:
            """Calculate the baseline of the indicated channels."""

        ...

        def plot_full(
            self,
            *,
            channel_idxs: list[int] | np.ndarray | None = None,
            axes: Axes | np.ndarray | None = None,
            signal_io: tuple[str | None, ...] = (None,),
            colors: list[str] | None = None,
            baseline_bool: bool = True,
            plot_ci: bool = False,
            **kwargs,
        ) -> None:
            """Plot the indicated channels in the provided axes."""

        ...

        def plot_peaks(
            self,
            *,
            channel_idxs: list[int] | np.ndarray | None = None,
            peak_set_name: str,
            axes: Axes | np.ndarray | None = None,
            signal_io: tuple[str | None, ...] = (None,),
            margin_s: int | None = None,
            valid_only: bool = False,
            colors: list[str] | None = None,
            baseline_bool: bool = True,
            plot_ci: bool = False,
            **kwargs,
        ) -> None:
            """Plot peak windows for the indicated channels."""

        ...

        def plot_markers(
            self,
            *,
            channel_idxs: list[int] | np.ndarray | None = None,
            peak_set_name: str,
            axes: Axes | np.ndarray | None = None,
            valid_only: bool = False,
            colors: str | list[str] | None = None,
            markers: str | list[str] | None = None,
        ) -> None:
            """Plot peak markers for the indicated channels."""

        ...

        def set_peaks(
            self,
            *,
            channel_idxs: list[int] | np.ndarray | None = None,
            peak_idxs: np.ndarray,
            signal: np.ndarray | None,
            peak_set_name: str,
            overwrite: bool = True,
        ) -> None:
            """Store a PeaksSet on the indicated channels."""

        ...

    def __init__(
        self,
        y_raw: np.ndarray,
        t_data: np.ndarray | None = None,
        fs: int | None = None,
        labels: list[str] | None = None,
        units: list[str] | None = None,
    ) -> None:
        """Initialize the TimeSeriesGroup object.

        Args:
                y_raw (numpy.ndarray): Raw signal data.
                t_data (numpy.ndarray, optional): Time axis data. If None,
                    generated from fs.
                fs (int, optional): Sampling rate. If None, calculated from
                    t_data.
                labels (list, optional): List of labels, one per channel.
                units (list, optional): List of signal units, one per channel.
        """
        self.channels = []
        self.fs: int
        y_raw, n_samp, n_channel = self._resolve_dims(y_raw)
        t_data, self.fs = self._resolve_time(t_data, fs, n_samp)
        self.param = {"fs": self.fs, "n_samp": n_samp, "n_channel": n_channel}
        self.labels, self.y_units = self._resolve_labels_units(labels, units, n_channel)
        for idx in range(n_channel):
            self.channels.append(
                TimeSeries(
                    y_raw=y_raw[idx, :],
                    t_data=t_data,
                    fs=self.fs,
                    label=self.labels[idx],
                    units=self.y_units[idx],
                )
            )

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        for method_name in cls._available_methods:
            if method_name not in cls.__dict__:
                source: Callable[..., Any] | None = getattr(TimeSeries, method_name, None)

                def make_wrapper(name: str, src: Callable[..., Any] | None) -> Callable[..., Any]:
                    def wrapper(
                        self: TimeSeriesGroup,
                        *,
                        channel_idxs: list[int] | np.ndarray | None = None,
                        **kw,
                    ) -> None:
                        return self._run_wrapper(name, channel_idxs=channel_idxs, **kw)

                    if src is not None:
                        functools.update_wrapper(wrapper, src)
                        sig = inspect.signature(src)
                        params = list(sig.parameters.values())
                        channel_idxs_param = inspect.Parameter(
                            "channel_idxs",
                            inspect.Parameter.KEYWORD_ONLY,
                            default=None,
                        )
                        kwarg_index = next(
                            (idx for idx, param in enumerate(params) if param.kind is inspect.Parameter.VAR_KEYWORD),
                            len(params),
                        )
                        params.insert(
                            kwarg_index,
                            channel_idxs_param,
                        )
                        wrapper.__signature__ = sig.replace(  # type: ignore[attr-defined]
                            parameters=params
                        )
                    return wrapper

                setattr(cls, method_name, make_wrapper(method_name, source))

    def __getitem__(self, key: int | str | None) -> TimeSeries:
        if isinstance(key, int):
            return self.channels[key]
        if isinstance(key, str):
            for channel in self.channels:
                if channel.label == key:
                    return channel
            msg = "Channel not found"
            raise KeyError(msg)
        msg = "Invalid key type"
        raise ValueError(msg)

    def __iter__(self):
        return iter(self.channels)

    def to_numpy(self, channel_idxs: int | np.ndarray | None = None, signal_io: tuple = (None,)) -> np.ndarray:
        """Convert the TimeSeriesGroup to a numpy array.

            The output is a 2D
            array with the shape (n_channels, n_samples). The signal type is
            determined by the signal_io parameter. If signal_io is (None,), the
            most advanced signal type (envelope > clean > filt > raw) is used.


        Args:
                channel_idxs (numpy.ndarray or int, optional): Channel indices to
                    include. If None, all channels are used.
                signal_io (tuple, optional): Tuple where the first element is the
                    input signal type. Default is (None,).

        Returns:
                numpy.ndarray: 2D array of shape (n_channels, n_samples).
        """
        if channel_idxs is None:
            channel_idxs = np.arange(self.param["n_channel"])
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])
        return np.array([self.channels[idx].signal_type_data(signal_io[0]) for idx in np.asarray(channel_idxs)])

    @staticmethod
    def _check_plot_kwargs(method: str, channel_idxs: np.ndarray, kwargs: dict) -> None:
        """Validate and auto-create axes for plot_* methods."""
        if "axes" not in kwargs:
            _, kwargs["axes"] = plt.subplots(nrows=len(channel_idxs), ncols=1, figsize=(10, 6), sharex=True)

        if method == "plot_full":
            kwargs["axes"] = np.atleast_1d(kwargs["axes"])
            if len(channel_idxs) > len(kwargs["axes"]):
                msg = "Provided axes have not enough rows for all channels to plot."
                raise ValueError(msg)
            if len(channel_idxs) < len(kwargs["axes"]):
                warnings.warn("\n".join(wrap(dedent("More axes provided than channels to plot."))))

        elif method in ["plot_peaks", "plot_markers"]:
            kwargs["axes"] = np.atleast_2d(kwargs["axes"])
            if kwargs["axes"].shape[0] < len(channel_idxs):
                msg = "Provided axes have not enough rows for all channels to plot."
                raise ValueError(msg)
            if "peak_set_name" not in kwargs:
                msg = "No peak_set_name provided."
                raise ValueError(msg)

    def _check_channel_idxs(self, channel_idxs: list[int] | np.ndarray | None, method: str) -> np.ndarray:
        if method not in self._available_methods:
            msg = "Invalid method"
            raise ValueError(msg)
        return np.asarray(self._resolve_channels(channel_idxs))

    def _run_wrapper(self, method: str, channel_idxs: list[int] | np.ndarray | None = None, **kwargs) -> None:
        channel_idxs = self._check_channel_idxs(channel_idxs, method)

        # only plot checks remain here
        if method.startswith("plot_"):
            self._check_plot_kwargs(method, channel_idxs, kwargs)
        elif isinstance(self, EmgDataGroup):
            if method in ["gating", "wavelet_denoising"]:
                self._check_ecg_future_warning(channel_idxs, kwargs)
            elif method == "get_ecg_peaks":
                self._resolve_ecg_source(kwargs)

        _kwargs = kwargs.copy()
        for idx, channel_idx in enumerate(channel_idxs):
            if method.startswith("plot_"):
                if method in ["plot_peaks", "plot_markers"]:
                    _kwargs["axes"] = kwargs["axes"][idx, :]
                else:
                    _kwargs["axes"] = kwargs["axes"][idx]
            getattr(self.channels[channel_idx], method)(**_kwargs)

    def run(self, method: str, channel_idxs: list[int] | np.ndarray | None = None, **kwargs) -> None:
        """LEGACY - Run the indicated method on the indicated channels with the provided kwargs."""
        if method not in self._available_methods:
            msg = "Invalid method"
            raise ValueError(msg)
        return self._run_wrapper(method, channel_idxs=channel_idxs, **kwargs)


class EmgDataGroup(TimeSeriesGroup):
    """Child-class of TimeSeriesGroup to store and handle emg data in.

    Includes additional methods filter_emg, gating, and wavelet_denoising. Enclosed
    TimeSeries objects can be indexed by index number or channel label.
    TimeSeries methods in TimeSeriesGroup._available_methods can be run on all
    or a subset of the TimeSeries objects through the TimeSeriesGroup.run
    method.
    When running ECG elimination methods (gating, wavelet_denoising) and no ECG
    channel or ecg_raw is provided, EmgDataGroup.ecg_idx is used to detect the
    QRS peak locations. ecg_idx is auto-detected from the labels on
    EmgDataGroup initialization, or can be set with the set_ecg_idx method.
    When no ECG channel is provided, the raw signal per TimeSeries object is
    used.
    """

    _available_methods: ClassVar[list[str]] = [
        *TimeSeriesGroup._available_methods,  # noqa: SLF001
        "filter_emg",
        "get_ecg_peaks",
        "gating",
        "wavelet_denoising",
    ]

    def __init__(
        self,
        y_raw: np.ndarray,
        t_data: np.ndarray | None = None,
        fs: int | None = None,
        labels: list[str] | None = None,
        units: list[str] | None = None,
    ):
        super().__init__(y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        labels_lc = [label.lower() for label in labels] if labels is not None else []
        if "ecg" in labels_lc:
            self.ecg_idx = labels_lc.index("ecg")
            logger.warning("Auto-detected ECG channel from labels.")
        else:
            logger.warning(
                "No ECG channel detected. Set ECG channel index with `EmgDataGroup.set_ecg_idx(arg)` method."
            )
            self.ecg_idx = None

    if TYPE_CHECKING:

        def filter_emg(
            self,
            channel_idxs: list[int] | np.ndarray | None = None,
            *,
            signal_io: tuple[str, str] = ("raw", "filt"),
            hp_cf: float = 20.0,
            lp_cf: float = 500.0,
            order: int = 3,
            **kwargs,
        ) -> None:
            """Apply EMG band-pass filtering to the indicated channels."""

        ...

        def get_ecg_peaks(
            self,
            channel_idxs: list[int] | np.ndarray | None = None,
            *,
            ecg_raw: np.ndarray | None = None,
            bp_filter: bool = True,
            overwrite: bool = True,
            name: str = "ecg",
        ) -> None:
            """Detect and store ECG peaks for the indicated channels."""

        ...

        def gating(
            self,
            channel_idxs: list[int] | np.ndarray | None = None,
            *,
            signal_io: tuple[str, str] = ("filt", "clean"),
            ecg_peakset_name: str = "ecg",
            gate_width_samples: int | None = None,
            fill_method: int = 3,
            **kwargs,
        ) -> None:
            """Remove ECG artefacts from the indicated channels using gating."""

        ...

        def wavelet_denoising(
            self,
            channel_idxs: list[int] | np.ndarray | None = None,
            *,
            signal_io: tuple[str, str] = ("filt", "clean"),
            ecg_peakset_name: str = "ecg",
            n: int | None = None,
            fixed_threshold: float | None = None,
            **kwargs,
        ) -> None:
            """Remove ECG artefacts from the indicated channels with wavelets."""

        ...

    def set_ecg_idx(self, ecg_idx: int | str) -> None:
        """Set the ECG channel index in the group.

        Args:
                ecg_idx (int or str): ECG channel index or label.
        """
        if isinstance(ecg_idx, int):
            self.ecg_idx = ecg_idx
        elif isinstance(ecg_idx, str):
            self.ecg_idx = self.labels.index(ecg_idx)

    def _check_ecg_future_warning(self, channel_idxs: list[int] | np.ndarray | None, kwargs: dict) -> None:
        """Handle FutureWarning and auto-ECG-peak resolution for gating/wavelet_denoising.

        Handle FutureWarning and auto-ECG-peak resolution for gating/wavelet_denoising.
        """
        if all(kwargs.get("ecg_peakset_name", "ecg") not in channel.peaks for channel in self.channels) or any(
            key in kwargs for key in ["ecg_peak_idxs", "ecg_raw", "bp_filter", "overwrite"]
        ):
            warnings.warn(
                "\n".join(
                    wrap(
                        dedent("""
                The kwargs "ecg_peak_idxs", "ecg_raw", "bp_filter",
                "overwrite" will be removed from the gating and wavelet
                denoising methods in future versions of ReSurfEMG: ...""")
                    )
                ),
                FutureWarning,
            )
            ecg_kwargs = {
                "channel_idxs": channel_idxs,
                "overwrite": kwargs.pop("overwrite", False),
            }
            if "ecg_peak_idxs" in kwargs:
                ecg_kwargs["ecg_peak_idxs"] = kwargs.pop("ecg_peak_idxs")
                self._run_wrapper("set_peaks", **ecg_kwargs)
            else:
                ecg_kwargs = {
                    "ecg_raw": kwargs.pop("ecg_raw", None),
                    "bp_filter": kwargs.pop("bp_filter", True),
                }
                self._run_wrapper("get_ecg_peaks", **ecg_kwargs)

    def _resolve_ecg_source(self, kwargs: dict) -> dict:
        """Resolve the ECG source for peak detection, updating kwargs in place."""
        if kwargs.get("ecg_raw") is not None:
            logger.warning("Provided raw ECG used for ECG peak detection.")
        elif self.ecg_idx is not None:
            kwargs["ecg_raw"] = self[self.ecg_idx]["raw"]
            logger.warning("Set ECG channel used for ECG peak detection.")
        else:
            logger.warning("Channel raw signals used for ECG peak detection.")
        return kwargs

    def filter(
        self,
        channel_idxs: list[int] | np.ndarray | None = None,
        *,
        signal_io: tuple[str, str] = ("raw", "filt"),
        hp_cf: float = 20.0,
        lp_cf: float = 500.0,
        order: int = 3,
        **kwargs,
    ) -> None:
        """Apply EMG-specific filtering to the indicated channels. See TimeSeries.filter_emg."""
        return self._run_wrapper(
            "filter_emg",
            channel_idxs=channel_idxs,
            signal_io=signal_io,
            hp_cf=hp_cf,
            lp_cf=lp_cf,
            order=order,
            **kwargs,
        )


class VentilatorDataGroup(TimeSeriesGroup):
    """Child-class of TimeSeriesGroup to store and handle ventilator data in.

    Default channels are "Paw"/ "Pvent", "F", and "Vvent", which are auto-
    detected from the labels. The PEEP-level (VentilatorDataGroup.peep) is
    auto-detected from the pressure channel when a pressure channel is set.
    """

    def __init__(
        self,
        y_raw: np.ndarray,
        t_data: np.ndarray | None = None,
        fs: int | None = None,
        labels: list[str] | None = None,
        units: list[str] | None = None,
    ) -> None:
        super().__init__(y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        if labels is None:
            labels = []

        self.p_vent_idx: int | None = next((labels.index(label) for label in ["Paw", "Pvent"] if label in labels), None)
        self.f_idx: int | None = labels.index("F") if "F" in labels else None
        self.v_vent_idx: int | None = labels.index("Vvent") if "Vvent" in labels else None

        if self.p_vent_idx is not None:
            logger.warning("Auto-detected Pvent channel from labels.")
        if self.f_idx is not None:
            logger.warning("Auto-detected Flow channel from labels.")
        if self.v_vent_idx is not None:
            logger.warning("Auto-detected Volume channel from labels.")

        if self.p_vent_idx is not None and self.v_vent_idx is not None:
            self.find_peep(self.p_vent_idx, self.v_vent_idx)
        else:
            self.peep = None

    def find_peep(self, pressure_idx: int | None, volume_idx: int | None) -> None:
        """Calculate PEEP.

            Calculate PEEP as the median value of p_vent at end-expiration.


        Args:
                pressure_idx (int): Channel index of the ventilator pressure data.
                volume_idx (int): Channel index of the ventilator volume data.
        """
        pressure_idx = pressure_idx or self.p_vent_idx
        if pressure_idx is None:
            msg = "pressure_idx and self.p_vent_idx not defined"
            raise ValueError(msg)

        volume_idx = volume_idx or self.v_vent_idx
        if volume_idx is None:
            msg = "volume_idx and self.v_vent_idx not defined"
            raise ValueError(msg)

        v_ee_pks, _ = scipy.signal.find_peaks(-self.channels[volume_idx]["raw"])
        self.peep = np.round(np.median(self.channels[pressure_idx]["raw"][v_ee_pks]))

    def find_occluded_breaths(
        self,
        pressure_idx: int | None = None,
        peep: float | None = None,
        overwrite: bool = True,
        **kwargs,
    ) -> None:
        """Find occluded breaths.

            Find end-expiratory occlusion manoeuvres in ventilator pressure
            timeseries data. See postprocessing.event_detection submodule.


        Args:
                pressure_idx (int, optional): Channel index of the ventilator
                    pressure data. Defaults to self.p_vent_idx.
                peep (float, optional): PEEP level. Defaults to self.peep.
                overwrite (bool): Overwrite existing peaks. Default is False.
                **kwargs: Additional arguments passed to
                    postprocessing.event_detection submodule.
        """
        pressure_idx = pressure_idx or self.p_vent_idx
        if pressure_idx is None:
            msg = "pressure_idx and self.p_vent_idx are not defined."
            raise ValueError(msg)

        kwargs["p_vent"] = self.channels[pressure_idx]["raw"]
        kwargs["fs"] = self.param["fs"]

        kwargs["peep"] = peep or self.peep
        if kwargs["peep"] is None:
            msg = "PEEP is not defined."
            raise ValueError(msg)

        peak_idxs = evt.find_occluded_breaths(**kwargs)
        peak_idxs = peak_idxs + kwargs["start_idx"]
        self.channels[pressure_idx].set_peaks(
            signal=self.channels[pressure_idx]["raw"],
            peak_idxs=peak_idxs,
            peak_set_name="Pocc",
            overwrite=overwrite,
        )

    def find_ventilator_peaks(
        self, channel_io: tuple[str | int, str | int | list] | None = None, overwrite: bool = False, **kwargs
    ) -> None:
        """Detect breath-related peaks in a specified ventilator signal.

        Peaks are stored in the corresponding TimeSeries under the same signal.

        Args:
            channel_io (tuple(str | int, str | int | list)): Tuple of the input
                and output channels. The first element is the input channel for
                peak detection; the found peaks are stored in the channels listed
                in the second element. If None, the volume or pressure channel is
                used as input; in absence of these, the first channel is used.
            overwrite (bool): Whether to overwrite existing peak set.
            **kwargs: Additional keyword arguments passed to the peak detection
                function.

        Returns:
            None
        """
        channel_key_i = (
            channel_io[0]
            if channel_io is not None and channel_io[0] is not None
            else (self.v_vent_idx or self.p_vent_idx or 0)
        )
        if not isinstance(channel_key_i, (int, str)):
            msg = "channel_io[0] must be int or str"
            raise TypeError(msg)
        try:
            signal_raw = self[channel_key_i]["raw"]
        except KeyError as e:
            msg = "channel_io[0] is not a valid channel."
            raise ValueError(msg) from e

        if channel_io is None:
            default_o = [idx for idx in (self.p_vent_idx, self.v_vent_idx) if idx is not None]
            channel_keys_o = self._resolve_channels(None, default=default_o)
        else:
            try:
                channel_keys_o = self._resolve_channels(channel_io[1])
            except ValueError as e:
                msg_0 = "channel_io[1] contains an invalid channel."
                raise ValueError(msg_0) from e

        kwargs["start_idx"] = kwargs.setdefault("start_idx", 0)
        kwargs["end_idx"] = kwargs.setdefault("end_idx", len(signal_raw) - 1)
        kwargs["width_s"] = kwargs.setdefault("width_s", self.param["fs"] // 4)

        # Detect peaks
        peak_idxs = evt.detect_ventilator_breath(
            np.asarray(signal_raw),
            start_idx=kwargs["start_idx"],
            end_idx=kwargs["end_idx"],
            width_s=kwargs["width_s"],
            threshold=kwargs.get("threshold"),
            prominence=kwargs.get("prominence"),
            threshold_new=kwargs.get("threshold_new"),
            prominence_new=kwargs.get("prominence_new"),
        )

        # Store peaks in the same signal
        for _channel_key in channel_keys_o:
            self.channels[_channel_key].set_peaks(
                signal=signal_raw,
                peak_idxs=peak_idxs,
                peak_set_name="ventilator_breaths",
                overwrite=overwrite,
            )
