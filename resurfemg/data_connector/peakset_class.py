"""Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains data classes for standardized peak data storage and method
automation.
"""

from typing import Literal

import numpy as np
import pandas as pd

from resurfemg.postprocessing.event_detection import (
    onoffpeak_baseline_crossing,
    onoffpeak_slope_extrapolation,
)


class PeaksSet:
    """Data class to store, and process peak information."""

    def __init__(
        self,
        signal: np.ndarray,
        t_data: np.ndarray,
        peak_idxs: np.ndarray | None = None,
    ) -> None:
        """Basic data class to store peak information.

        :param signal: 1-dimensional signal data
        :type signal: ~numpy.ndarray
        :param t_data: time axis data
        :type t_data: ~numpy.ndarray
        :param peak_idxs: Indices of peaks
        :type peak_idxs: ~numpy.ndarray
        """
        if isinstance(signal, np.ndarray):
            self.signal = signal
        else:
            msg = "Invalid signal type: 'signal_type'."
            raise ValueError(msg)

        if isinstance(t_data, np.ndarray):
            self.t_data = t_data
        else:
            msg = "Invalid t_data type: 't_data'."
            raise ValueError(msg)

        if peak_idxs is None:
            peak_idxs = np.array([])
        elif (
            isinstance(peak_idxs, np.ndarray)
            and len(np.array(peak_idxs).shape) == 1
        ):
            pass
        elif isinstance(peak_idxs, list):
            peak_idxs = np.array(peak_idxs)
        else:
            msg = "Invalid peak indices: 'peak_s'."
            raise ValueError(msg)

        self.peak_df = pd.DataFrame(
            data=peak_idxs, columns=pd.Index(["peak_idx"])
        )
        self.quality_values_df = pd.DataFrame(
            data=peak_idxs, columns=pd.Index(["peak_idx"])
        )
        self.quality_outcomes_df = pd.DataFrame(
            data=peak_idxs, columns=pd.Index(["peak_idx"])
        )
        self.time_products = None

    def __len__(self):
        return len(self.peak_df)

    def __getitem__(self, key: str | int):
        return self.peak_df[key].to_numpy()

    def __str__(self):
        return str(self.peak_df)

    def keys(self) -> pd.Index:
        """Return the column names of the peak dataframe."""
        return self.peak_df.keys()

    def to_dict(self, export_time_data: bool = False) -> dict:
        """Convert the PeaksSet object to a dictionary for JSON serialization."""
        export_dict = {
            "signal": self.signal,
            "peak_df": self.peak_df,
            "quality_values_df": self.quality_values_df,
            "quality_outcomes_df": self.quality_outcomes_df,
            "time_products": self.time_products,
        }
        if export_time_data:
            export_dict["t_data"] = self.t_data
        return export_dict

    def detect_on_offset(
        self,
        baseline: np.ndarray | None = None,
        method: Literal[
            "default", "baseline_crossing", "slope_extrapolation"
        ] = "default",
        fs: int | None = None,
        slope_window_s: int | None = None,
    ) -> None:
        """Detect the peak on- and offsets.

        See postprocessing.event_detection submodule.
        """
        if baseline is None:
            baseline = np.zeros(self.signal.shape)

        peak_idxs = self.peak_df["peak_idx"].to_numpy()

        match method:
            case "default" | "baseline_crossing":
                (start_idxs, end_idxs, _, _, valid_list) = (
                    onoffpeak_baseline_crossing(
                        self.signal, baseline, peak_idxs
                    )
                )
            case "slope_extrapolation":
                if fs is None:
                    msg = "Sampling rate is not defined."
                    raise ValueError(msg)

                if slope_window_s is None:
                    # TODO Insert valid default slope window
                    slope_window_s = fs // 5

                (start_idxs, end_idxs, _, _, valid_list) = (
                    onoffpeak_slope_extrapolation(
                        self.signal, fs, peak_idxs, slope_window_s
                    )
                )
            case _:
                msg = "Detection algorithm does not exist."
                raise KeyError(msg)

        self.peak_df["start_idx"] = start_idxs
        self.peak_df["end_idx"] = end_idxs
        self.peak_df["valid"] = valid_list
        quality_outcomes_df = self.quality_outcomes_df
        quality_outcomes_df["baseline_detection"] = valid_list

        self.evaluate_validity(quality_outcomes_df)

    def update_test_outcomes(self, tests_df_new: pd.DataFrame) -> None:
        """Update the quality values dataframe.

        Add new peak quality test to self.quality_outcomes_df, and update
        existing entries.
        -----------------------------------------------------------------------
        :param tests_df_new: Dataframe of test parameters per peak
        :type tests_df_new: pandas.DataFrame

        :returns: None
        :rtype: None
        """
        if self.quality_values_df is not None:
            df_old = self.quality_values_df
            pre_existing_keys = list(
                set(tests_df_new.keys()) & set(df_old.keys())
            )
            df_old = df_old.drop(columns=pre_existing_keys)
            tests_df_merge = df_old.merge(
                tests_df_new, left_index=True, right_index=True
            )
            if not self.quality_values_df["peak_idx"].equals(
                tests_df_merge["peak_idx"]
            ):
                msg = "Mismatched 'peak_idx' between old and new dataframes."
                raise ValueError(msg)
            self.quality_values_df = tests_df_merge
        else:
            self.quality_values_df = tests_df_new

    def evaluate_validity(self, tests_df_new: pd.DataFrame) -> None:
        """Update the quality outcomes dataframe.

        Update peak validity based on previously and newly executed tests
        in self.quality_outcomes_df.
        -----------------------------------------------------------------------
        :param tests_df_new: Dataframe of passed tests per peak
        :type tests_df_new: pandas.DataFrame

        :returns: None
        :rtype: None
        """
        if self.quality_outcomes_df is not None:
            df_old = self.quality_outcomes_df
            pre_existing_keys = list(
                set(tests_df_new.keys()) & set(df_old.keys())
            )
            df_old = df_old.drop(columns=pre_existing_keys)
            tests_df_merge = df_old.merge(
                tests_df_new, left_index=True, right_index=True
            )
            if not self.quality_outcomes_df["peak_idx"].equals(
                tests_df_merge["peak_idx"]
            ):
                msg = "Mismatched 'peak_idx' between old and new dataframes."
                raise ValueError(msg)
            self.quality_outcomes_df = tests_df_merge
        else:
            self.quality_outcomes_df = tests_df_new

        test_keys = list(tests_df_new.keys())
        test_keys.pop(test_keys.index("peak_idx"))
        passed_tests = np.all(
            tests_df_new.loc[:, test_keys].to_numpy(), axis=1
        )
        self.peak_df["valid"] = passed_tests

    def sanitize(self) -> None:
        """Deleve invalid peak entries from the peakset.

        Delete invalid peak entries (self.peak_df['valid'] is False) from
        self.peak_df, self.quality_values_df, and self.quality_outcomes_df.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        valid_idxs = self.peak_df["valid"].to_numpy()
        self.peak_df = self.peak_df.loc[valid_idxs].reset_index(drop=True)
        self.quality_outcomes_df = self.quality_outcomes_df.loc[
            valid_idxs
        ].reset_index(drop=True)
        self.quality_values_df = self.quality_values_df.loc[
            valid_idxs
        ].reset_index(drop=True)

    @classmethod
    def from_dict(cls, data: dict) -> "PeaksSet":
        """Load the PeaksSet object from a dictionary."""
        peak_set = cls(
            signal=np.asarray(data["signal"]),
            t_data=np.asarray(data["t_data"]),
        )
        peak_set.peak_df = data["peak_df"]
        peak_set.quality_values_df = data["quality_values_df"]
        peak_set.quality_outcomes_df = data["quality_outcomes_df"]
        peak_set.time_products = data["time_products"]
        return peak_set
