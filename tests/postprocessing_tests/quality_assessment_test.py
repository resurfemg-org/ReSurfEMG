"""Sanity tests for the postprocessing module of the resurfemg library."""  # noqa: INP001

import logging
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import scipy
from scipy.integrate import trapezoid

import resurfemg.postprocessing.baseline as bl
import resurfemg.postprocessing.event_detection as evt
import resurfemg.postprocessing.features as feat
import resurfemg.postprocessing.quality_assessment as qa
from resurfemg.preprocessing import envelope as evl

logger = logging.getLogger(__name__)
rng = np.random.default_rng(None)

sample_emg = Path(__file__).resolve().parents[2] / "test_data" / "emg_data_synth_quiet_breathing.Poly5"

# Dummy EMG signal
fs_emg = 2048
rr = 30
t_r = 60 / rr
f_r = 1 / t_r
t_emg = np.array([s_t / fs_emg for s_t in range(10 * fs_emg)])
y_sin = np.cos((f_r * t_emg - 0.5) * 2 * np.pi)
y_sin[y_sin < 0] = 0
y_rand = rng.normal(0, 1, size=len(y_sin))
y_rand_baseline = rng.normal(0, 1, size=len(y_sin)) / 10
y_t_emg = y_sin * y_rand + y_rand_baseline

y_env_emg = evl.full_rolling_rms(y_t_emg, fs_emg // 5)
y_emg_baseline = bl.moving_baseline(y_env_emg, 5 * fs_emg, fs_emg // 2)

peaks_env, _ = scipy.signal.find_peaks(y_env_emg, prominence=0.1)
emg_start_idxs, emg_end_idxs, *_ = evt.onoffpeak_baseline_crossing(y_env_emg, y_emg_baseline, peaks_env)
etps = feat.time_product(signal=y_env_emg, fs=fs_emg, start_idxs=emg_start_idxs, end_idxs=emg_end_idxs)
aubs = feat.area_under_baseline(
    signal=y_env_emg,
    fs=fs_emg,
    peak_idxs=peaks_env,
    start_idxs=emg_start_idxs,
    end_idxs=emg_end_idxs,
    baseline=y_emg_baseline,
    aub_window_s=5 * fs_emg,
)

# Dummy Pocc signal
fs_vent = 100
s_vent = np.arange(10 * fs_vent)
t_vent = (s_vent + 1) / fs_vent
y_sin = np.sin((f_r * t_vent) * 2 * np.pi)
y_sin[y_sin > 0] = 0
y_t_p_vent = 5 * y_sin

pocc_peaks_valid, _ = scipy.signal.find_peaks(-y_t_p_vent, prominence=0.1)
pocc_starts = s_vent[((t_vent + t_r / 2) % t_r == 0)]
pocc_ends = s_vent[(t_vent % t_r == 0)]

PTP_occs = np.zeros(pocc_peaks_valid.shape)
for _idx, _ in enumerate(pocc_peaks_valid):
    PTP_occs[_idx] = trapezoid(-y_t_p_vent[pocc_starts[_idx] : pocc_ends[_idx]], dx=1 / fs_vent)


class TestSnrPseudo(unittest.TestCase):
    """Test Snr Pseudo."""

    fs_emg = 2048
    t_emg = np.array([s_t / fs_emg for s_t in range(15 * fs_emg)])

    y_block = np.array(10 * scipy.signal.square((t_emg - 1.25) / 5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0
    y_baseline = np.ones(y_block.shape)
    peak_idxs: ClassVar[list] = [(5 // 2 + x * 5) * 2048 for x in range(3)]

    snr_values = qa.snr_pseudo(y_block, peak_idxs, y_baseline, fs_emg)

    def test_snr_length(self) -> None:
        """Test that the length of the SNR values matches the number of peaks."""
        assert len(self.snr_values) == len(self.peak_idxs)

    def test_snr_values(self) -> None:
        """Test that the SNR values are approximately 10 dB for the given signal."""
        median_snr = np.median(self.snr_values)
        assert median_snr == pytest.approx(10.0, abs=1e-3)


class TestPoccQuality(unittest.TestCase):
    """Test Pocc Quality."""

    valid_poccs, _ = qa.pocc_quality(y_t_p_vent, pocc_peaks_valid, pocc_ends, PTP_occs)

    def test_valid_pocc(self) -> None:
        """Test that all valid poccs are correctly identified."""
        assert not np.any(~self.valid_poccs)

    def test_negative_upslope(self) -> None:
        """Test that a negative upslope is correctly identified as invalid."""
        y_sin_shifted = np.sin(((f_r - 0.025) * t_vent - 0.11) * 2 * np.pi)
        y_sin_shifted[y_sin_shifted > 0] = 0
        y_sin_shifted[t_vent < 7.0] = 0
        y_t_shifted = 5 * y_sin + 4 * y_sin_shifted

        invalid_upslope, _ = qa.pocc_quality(y_t_shifted, pocc_peaks_valid, pocc_ends, PTP_occs)

        assert not invalid_upslope[-1]

    def test_steep_upslope(self) -> None:
        """Test that a steep upslope is correctly identified as invalid."""
        y_sin_shifted = np.sin((f_r * t_vent - 0.4) * 2 * np.pi)
        y_sin_shifted[y_sin_shifted > 0] = 0
        y_sin_shifted = y_sin_shifted**4
        y_t_steeper = 1000 * y_sin * y_sin_shifted

        peak_idxs_steeper, _ = scipy.signal.find_peaks(-y_t_steeper, prominence=0.1)
        y_baseline = bl.moving_baseline(-y_t_steeper, int(7.5 * fs_vent), fs_vent // 5)

        peak_start_idxsteep, peak_end_idxs_steep, _, _, _ = evt.onoffpeak_baseline_crossing(
            y_t_steeper, y_baseline, peak_idxs_steeper
        )

        ptp_occs_steep = np.zeros(peak_idxs_steeper.shape)
        for idx, _ in enumerate(peak_idxs_steeper):
            ptp_occs_steep[idx] = trapezoid(
                -y_t_steeper[peak_start_idxsteep[idx] : peak_end_idxs_steep[idx]], dx=1 / fs_vent
            )

        steep_upslope, _ = qa.pocc_quality(y_t_steeper, peak_idxs_steeper, peak_end_idxs_steep, ptp_occs_steep)

        assert not steep_upslope[-1]

    def test_consec_manoeuvres(self) -> None:
        """Test that consecutive manoeuvres are correctly identified as invalid."""
        sim_breaths = np.arange(1, 20, 2)
        sim_occ = np.arange(1, 20, 10)
        sim_occ_false = np.array([1, 7, 9])
        valid_manoeuvres = qa.detect_non_consecutive_manoeuvres(
            ventilator_breath_idxs=sim_breaths, manoeuvres_idxs=sim_occ
        )
        assert np.all(valid_manoeuvres)
        valid_manoeuvres_false = qa.detect_non_consecutive_manoeuvres(sim_breaths, sim_occ_false)
        assert not np.all(valid_manoeuvres_false)


class TestAreaUnderBaselineQuality(unittest.TestCase):
    """Test Area Under Baseline Quality."""

    # Define signal
    fs_emg = 2048
    t_emg = np.array([s_t / fs_emg for s_t in range(15 * fs_emg)])

    y_block = np.array(3 * scipy.signal.square((t_emg - 1.25) / 5 * 2 * np.pi, duty=0.5))
    y_block[y_block < 0] = 0

    peak_idxs: ClassVar[list] = [(5 // 2 + x * 5) * 2048 for x in range(3)]
    start_idxs: ClassVar[list] = [(5 + x * 5 * 4) * 2048 // 4 for x in range(3)]
    end_idxs: ClassVar[list] = [(15 + x * 5 * 4) * 2048 // 4 - 1 for x in range(3)]

    def test_percentage_aub_good(self) -> None:
        """Test that the percentage of area under baseline is correctly identified as valid with correct baseline."""
        y_baseline = np.ones(self.y_block.shape)
        valid_timeproducts, _, _ = qa.percentage_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peak_idxs,
            self.start_idxs,
            self.end_idxs,
            y_baseline,
            aub_window_s=None,
            ref_signal=None,
            aub_threshold=40,
        )

        assert np.all(valid_timeproducts)

    def test_percentage_aub_wrong(self) -> None:
        """Test that the percentage of area under baseline is correctly identified as invalid with wrong baseline."""
        y_baseline = 2 * np.ones(self.y_block.shape)
        valid_timeproducts, _, _ = qa.percentage_under_baseline(
            self.y_block,
            self.fs_emg,
            self.peak_idxs,
            self.start_idxs,
            self.end_idxs,
            y_baseline,
            aub_window_s=None,
            ref_signal=None,
            aub_threshold=40,
        )

        assert not np.all(valid_timeproducts)

    def test_detect_local_high_aub(self) -> None:
        """Test that local high area under baseline values are correctly identified."""
        valid_aubs = qa.detect_local_high_aub(
            aubs=aubs[0],
            threshold_percentile=75,
            threshold_factor=4,
        )
        assert np.all(valid_aubs)

    def test_detect_extreme_time_products(self) -> None:
        """Test that extreme time products are correctly identified."""
        valid_etps = qa.detect_extreme_time_products(
            etps,
            upper_percentile=95.0,
            upper_factor=10.0,
            lower_percentile=5.0,
            lower_factor=0.1,
        )
        assert np.all(valid_etps)


class TestBellFit(unittest.TestCase):
    """Test Bell Fit."""

    def test_evaluate_bell_curve_error(self) -> None:
        """Test that the bell curve error is correctly evaluated."""
        output = qa.evaluate_bell_curve_error(
            peak_idxs=peaks_env,
            start_idxs=emg_start_idxs,
            end_idxs=emg_end_idxs,
            signal=y_env_emg,
            fs=fs_emg,
            time_products=etps,
            bell_window_s=None,
            bell_threshold=40,
        )
        (valid_peak, _, percentage_bell_error, *_) = output

        np.testing.assert_equal(valid_peak, np.array([True, True, True, True, True]))
        np.testing.assert_equal(np.abs(percentage_bell_error - 5) < 5, np.array([True, True, True, True, True]))


class TestInterpeakMethods(unittest.TestCase):
    """Test Interpeak Methods."""

    def test_interpeak_dist(self) -> None:
        """Test that the interpeak distance is correctly evaluated."""
        sim_ecg = np.arange(1, 11)
        sim_emg = np.linspace(1, 10, 4)
        valid_interpeak = qa.interpeak_dist(sim_ecg, sim_emg, threshold=1.1)

        assert valid_interpeak, "The interpeak_dist functiondid not return True as expected."


class TestEvaluateEventTiming(unittest.TestCase):
    """Test Evaluate Event Timing."""

    def test_evaluate_event_timing_correct(self) -> None:
        """Test that the event timing is correctly evaluated."""
        t_emg_peaks = (np.arange(int(10 * f_r)) + 0.2) / f_r
        t_vent_peaks = (np.arange(int(10 * f_r)) + 0.25) / f_r
        (correct_timing, delta_time) = qa.evaluate_event_timing(
            t_events_1=t_emg_peaks,
            t_events_2=t_vent_peaks,
            delta_min=-0.5,
            delta_max=0.5 * 60 / rr,
        )
        np.testing.assert_array_almost_equal(delta_time, [0.1, 0.1, 0.1, 0.1, 0.1], 6)
        assert np.all(correct_timing)

    def test_evaluate_event_timing_incorrect_too_late(self) -> None:
        """Test that the event timing is correctly evaluated when events are too late."""
        t_emg_peaks = (np.arange(int(10 * f_r)) + 0.55) / f_r
        t_vent_peaks = (np.arange(int(10 * f_r)) + 0.25) / f_r
        (correct_timing, delta_time) = qa.evaluate_event_timing(
            t_events_1=t_emg_peaks,
            t_events_2=t_vent_peaks,
            delta_min=-0.5,
            delta_max=0.5 * 60 / rr,
        )
        np.testing.assert_array_almost_equal(delta_time, 5 * [-0.6], 6)
        assert np.all(~correct_timing)

    def test_evaluate_event_timing_incorrect_too_early(self) -> None:
        """Test that the event timing is correctly evaluated when events are too early."""
        t_emg_peaks = (np.arange(int(10 * f_r)) + 0.05) / f_r
        t_vent_peaks = (np.arange(int(10 * f_r)) + 0.6) / f_r
        (correct_timing, delta_time) = qa.evaluate_event_timing(
            t_events_1=t_emg_peaks,
            t_events_2=t_vent_peaks,
            delta_min=-0.5,
            delta_max=0.5 * 60 / rr,
        )
        np.testing.assert_array_almost_equal(delta_time, 5 * [1.1], 6)
        assert np.all(~correct_timing)


class TestEvaluateRespiratoryRates(unittest.TestCase):
    """Test Evaluate Respiratory Rates."""

    def test_evaluate_respiratory_rates(self) -> None:
        """Test that the respiratory rates are correctly evaluated."""
        logger.info(rr, peaks_env)
        fraction_emg_breaths, crit_met = qa.evaluate_respiratory_rates(
            emg_breath_idxs=peaks_env,
            t_emg=max(t_emg),
            rr_vent=rr,
        )
        assert abs(fraction_emg_breaths - 1.0) < 0.01
        assert crit_met


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False)
