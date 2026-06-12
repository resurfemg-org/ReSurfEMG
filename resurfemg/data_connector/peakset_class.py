"""This file contains data classes for standardized peak data storage and method automation.

Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.
"""

from __future__ import annotations

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
        signal: np.ndarray | None,
        t_data: np.ndarray,
        peak_idxs: np.ndarray | None = None,
    ):
        """Initialize the PeaksSet.

        Args:
            signal (numpy.ndarray): 1-dimensional signal data.
            t_data (numpy.ndarray): Time axis data.
            peak_idxs (numpy.ndarray): Indices of peaks.

        Raises:
            ValueError: If signal, t_data, or peak_idxs have an invalid type.
        """
        if isinstance(signal, np.ndarray):
            self.signal: np.ndarray = signal
        else:
            msg = "Invalid signal type: 'signal_type'."
            raise TypeError(msg)
        if isinstance(t_data, np.ndarray):
            self.t_data: np.ndarray = t_data
        else:
            msg = "Invalid t_data type: 't_data'."
            raise TypeError(msg)
        if peak_idxs is None:
            peak_idxs = np.array([])
        elif isinstance(peak_idxs, np.ndarray) and len(np.array(peak_idxs).shape) == 1:
            pass
        elif isinstance(peak_idxs, list):
            peak_idxs = np.array(peak_idxs)
        else:
            msg = "Invalid peak indices: 'peak_s'."
            raise TypeError(msg)
        self.peak_df: pd.DataFrame = pd.DataFrame(data=peak_idxs, columns=["peak_idx"])
        self.quality_values_df: pd.DataFrame = pd.DataFrame(
            data=peak_idxs, columns=["peak_idx"]
        )
        self.quality_outcomes_df: pd.DataFrame = pd.DataFrame(
            data=peak_idxs, columns=["peak_idx"]
        )
        self.time_products: np.ndarray | None = None

    def __len__(self):
        return len(self.peak_df)

    def __contains__(self, key: str):
        return key in self.peak_df.columns

    def __getitem__(self, key: str):
        return self.peak_df[key].to_numpy()

    def __str__(self):
        return str(self.peak_df)

    def keys(self) -> pd.Index:
        """Return the column names of the peak dataframe."""
        return self.peak_df.keys()

    def detect_on_offset(
        self,
        baseline: np.ndarray | None = None,
        method: str = "default",
        fs: int | None = None,
        slope_window_s: int | None = None,
    ) -> None:
        """Detect the peak on- and offsets.

        See postprocessing.event_detection submodule.
        """
        if baseline is None:
            baseline = np.zeros(self.signal.shape)

        peak_idxs = self.peak_df["peak_idx"].to_numpy()

        if method in {"default", "baseline_crossing"}:
            start_idxs, end_idxs, _, _, valid_list = onoffpeak_baseline_crossing(
                self.signal, baseline, peak_idxs
            )

        elif method == "slope_extrapolation":
            if fs is None:
                msg = "Sampling rate is not defined."
                raise ValueError(msg)

            if slope_window_s is None:
                # TODO Insert valid default slope window
                slope_window_s = fs // 5

            start_idxs, end_idxs, _, _, valid_list = onoffpeak_slope_extrapolation(
                self.signal, fs, peak_idxs, slope_window_s
            )
        else:
            msg = "Detection algorithm does not exist."
            raise KeyError(msg)

        self.peak_df["start_idx"] = start_idxs
        self.peak_df["end_idx"] = end_idxs
        self.peak_df["valid"] = valid_list
        quality_outcomes_df = self.quality_outcomes_df
        quality_outcomes_df["baseline_detection"] = valid_list

        self.evaluate_validity(quality_outcomes_df)

    def update_test_outcomes(self, tests_df_new: pd.DataFrame) -> None:
        """Add new peak quality test to self.quality_outcomes_df.

        Updates existing entries.

        Args:
            tests_df_new (pandas.DataFrame): Dataframe of test parameters per
                peak.
        """
        if self.quality_values_df is not None:
            df_old = self.quality_values_df
            pre_existing_keys = list(set(tests_df_new.keys()) & set(df_old.keys()))
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
        """Update peak validity based on executed tests.

        Considers previously and newly executed tests in
        self.quality_outcomes_df.

        Args:
            tests_df_new (pandas.DataFrame): Dataframe of passed tests per
                peak.
        """
        if self.quality_outcomes_df is not None:
            df_old = self.quality_outcomes_df
            pre_existing_keys = list(set(tests_df_new.keys()) & set(df_old.keys()))
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
        passed_tests = np.all(tests_df_new.loc[:, test_keys].to_numpy(), axis=1)
        self.peak_df["valid"] = passed_tests

    def sanitize(self) -> None:
        """Delete invalid peak entries.

        Removes peaks where self.peak_df['valid'] is False from self.peak_df,
        self.quality_values_df, and self.quality_outcomes_df.
        """
        valid_idxs = self.peak_df["valid"].to_numpy()
        self.peak_df = self.peak_df.loc[valid_idxs].reset_index(drop=True)
        self.quality_outcomes_df = self.quality_outcomes_df.loc[valid_idxs].reset_index(
            drop=True
        )
        self.quality_values_df = self.quality_values_df.loc[valid_idxs].reset_index(
            drop=True
        )
