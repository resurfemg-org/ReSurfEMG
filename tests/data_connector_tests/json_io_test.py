"""Tests for TimeSeriesGroup.export_to_json / import_from_json"""

import gzip
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from resurfemg.data_connector.data_classes import (
    EmgDataGroup,
    TimeSeriesGroup,
    VentilatorDataGroup,
)
from resurfemg.data_connector.peakset_class import PeaksSet
from resurfemg.data_connector.tmsisdk_lite import Poly5Reader
from resurfemg.helper_functions import file_utilities as fu

TEST_DATA_DIR = os.path.join(
    os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "test_data",
)
SYNTH_POCC_EMG = os.path.join(TEST_DATA_DIR, "emg_data_synth_pocc.Poly5")
SYNTH_POCC_VENT = os.path.join(TEST_DATA_DIR, "vent_data_synth_pocc.Poly5")


def _sha256_gz(path):
    """Hash of the decompressed file content, streamed to keep memory low."""
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as fin:
        for chunk in iter(lambda: fin.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_dump(path):
    with gzip.open(path, "rt") as fin:
        return json.load(fin)


def _build_emg_group(crop_s=120):
    """Processed EmgDataGroup with every exported field populated."""
    data = Poly5Reader(SYNTH_POCC_EMG)
    fs = data.sample_rate
    y_emg = data.samples[:, : crop_s * fs]
    emg = EmgDataGroup(y_emg, fs=fs, labels=["ECG", "EMGdi"], units=["uV", "uV"])
    emg.filter_emg()
    emg.get_ecg_peaks(overwrite=True)
    emg.gating()
    # ci_alpha populates env_ci (tuple of lower/upper arrays)
    emg.envelope(env_type="rms", signal_io=("clean", "env"), ci_alpha=0.05)
    emg.baseline()

    emg_di = emg[1]
    emg_di.detect_emg_breaths(peak_set_name="breaths")
    emg_di.peaks["breaths"].detect_on_offset(baseline=emg_di["baseline"])
    emg_di.calculate_time_products(
        peak_set_name="breaths",
        aub_reference_signal=emg_di["baseline"],
        parameter_name="ETPdi",
    )
    emg_di.test_emg_quality(
        "breaths", parameter_names={"time_product": "ETPdi"}, verbose=False
    )
    emg_di.peaks["breaths"].time_products = (
        emg_di.peaks["breaths"].peak_df["ETPdi"].to_numpy()
    )
    return emg


def _build_vent_group():
    """VentilatorDataGroup with Pocc peaks, PTPs and quality tests."""
    data = Poly5Reader(SYNTH_POCC_VENT)
    y_vent = data.samples[: data.num_samples]
    vent = VentilatorDataGroup(
        y_vent,
        fs=data.sample_rate,
        labels=["Pvent", "F", "Vvent"],
        units=["cmH2O", "L/s", "L"],
    )
    vent.baseline(channel_idxs=[0], signal_io=("raw", "baseline"))
    vent.find_occluded_breaths(vent.p_vent_idx, start_idx=360 * vent.param["fs"])
    vent.find_ventilator_peaks()  # required by the consecutive-Pocc test
    p_vent = vent[vent.p_vent_idx]
    p_vent.peaks["Pocc"].detect_on_offset(baseline=p_vent["baseline"])
    p_vent.calculate_time_products(
        peak_set_name="Pocc",
        aub_reference_signal=p_vent["baseline"],
        parameter_name="PTPocc",
    )
    p_vent.test_pocc_quality(
        "Pocc", parameter_names={"time_product": "PTPocc"}, verbose=False
    )
    return vent


def _random_group(cls=EmgDataGroup, labels=None, t_data=None, seed=0):
    """Small raw-only group for edge cases."""
    rng = np.random.default_rng(seed)
    fs = 256
    y_raw = rng.normal(size=(2, 10 * fs))
    if t_data is not None:
        return cls(y_raw, t_data=t_data, fs=fs, labels=labels)
    return cls(y_raw, fs=fs, labels=labels)


class _TmpDirMixin:
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    @classmethod
    def path(cls, name):
        return os.path.join(cls.tmpdir.name, name)


class TestEmgRoundTrip(_TmpDirMixin, unittest.TestCase):
    """Full processed EmgDataGroup: export, import, compare everything."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.original = _build_emg_group()
        cls.dump_file = cls.path("emg.json.gz")
        cls.original.export_to_json(cls.dump_file)
        cls.loaded = EmgDataGroup.import_from_json(cls.dump_file)

    def _channel_pairs(self):
        return zip(self.original.channels, self.loaded.channels, strict=True)

    # --- file format ---
    def test_file_structure(self):
        dump = _read_dump(self.dump_file)
        self.assertEqual(set(dump), {"metadata", "data", "t_data", "object_type"})
        self.assertEqual(dump["object_type"], "EmgDataGroup")
        self.assertEqual(set(dump["data"]), {"0", "1"})

    def test_t_data_stored_once(self):
        dump = _read_dump(self.dump_file)
        for channel in dump["data"].values():
            self.assertNotIn("t_data", channel)
            for peak_set in channel["peaks"].values():
                self.assertNotIn("t_data", peak_set)

    # --- group level ---
    def test_returns_subclass(self):
        self.assertIs(type(self.loaded), EmgDataGroup)

    def test_group_attributes(self):
        self.assertEqual(self.loaded.labels, self.original.labels)
        self.assertEqual(self.loaded.y_units, self.original.y_units)
        self.assertEqual(self.loaded.fs, self.original.fs)
        self.assertEqual(self.loaded.param, self.original.param)

    def test_subclass_init_attributes(self):
        self.assertEqual(self.loaded.ecg_idx, self.original.ecg_idx)
        self.assertEqual(self.loaded.ecg_idx, 0)

    # --- channel level ---
    def test_channel_attributes(self):
        for orig, new in self._channel_pairs():
            self.assertEqual(new.label, orig.label)
            self.assertEqual(new.y_units, orig.y_units)
            self.assertEqual(new.param, orig.param)

    def test_t_data(self):
        for orig, new in self._channel_pairs():
            self.assertIsInstance(new.t_data, np.ndarray)
            np.testing.assert_array_equal(new.t_data, orig.t_data)

    def test_y_data_keys(self):
        for orig, new in self._channel_pairs():
            self.assertEqual(set(new._y_data), set(orig._y_data))

    def test_y_data_exact(self):
        for orig, new in self._channel_pairs():
            for key, value in orig._y_data.items():
                if key == "env_ci":
                    continue
                self.assertIsInstance(new[key], np.ndarray, key)
                # JSON float repr is lossless for float64
                np.testing.assert_array_equal(new[key], value, err_msg=key)

    def test_env_ci_as_2d_array(self):
        """env_ci is a (lower, upper) tuple in memory; loads as (2, N)."""
        for orig, new in self._channel_pairs():
            expected = np.asarray(orig["env_ci"])
            self.assertEqual(new["env_ci"].shape, expected.shape)
            np.testing.assert_array_equal(new["env_ci"], expected)

    # --- peak sets ---
    def test_peak_set_names(self):
        for orig, new in self._channel_pairs():
            self.assertEqual(set(new.peaks), set(orig.peaks))
        self.assertEqual(set(self.loaded[1].peaks), {"ecg", "breaths"})

    def test_peak_signals(self):
        for orig, new in self._channel_pairs():
            for name, ps_orig in orig.peaks.items():
                self.assertIsInstance(new.peaks[name].signal, np.ndarray)
                np.testing.assert_array_equal(
                    new.peaks[name].signal, ps_orig.signal, err_msg=name
                )

    def test_peak_set_shares_channel_t_data(self):
        for new in self.loaded.channels:
            for name, peak_set in new.peaks.items():
                self.assertIs(peak_set.t_data, new.t_data, name)

    def test_peak_dataframes(self):
        for orig, new in self._channel_pairs():
            for name, ps_orig in orig.peaks.items():
                ps_new = new.peaks[name]
                for attr in ("peak_df", "quality_values_df", "quality_outcomes_df"):
                    pd.testing.assert_frame_equal(
                        getattr(ps_new, attr),
                        getattr(ps_orig, attr),
                        obj=f"{name}.{attr}",
                    )

    def test_ecg_peak_df_has_no_valid_column(self):
        """Import must not recompute validity (would add 'valid')."""
        self.assertNotIn("valid", self.loaded[0].peaks["ecg"].peak_df)

    def test_time_products(self):
        ps_orig = self.original[1].peaks["breaths"]
        ps_new = self.loaded[1].peaks["breaths"]
        np.testing.assert_array_almost_equal(
            np.asarray(ps_new.time_products), ps_orig.time_products
        )
        self.assertIsNone(self.loaded[0].peaks["ecg"].time_products)

    # --- behaviour after loading ---
    def test_reexport_identical(self):
        dump_2 = self.path("emg_reexport.json.gz")
        self.loaded.export_to_json(dump_2)
        self.assertEqual(_sha256_gz(self.dump_file), _sha256_gz(dump_2))

    def test_loaded_group_can_be_reprocessed(self):
        """Peak detection on the loaded data reproduces the original peaks."""
        dump = self.path("emg_reprocess.json.gz")
        self.original.export_to_json(dump)
        loaded = EmgDataGroup.import_from_json(dump)
        loaded[1].detect_emg_breaths(peak_set_name="redo")
        np.testing.assert_array_equal(
            loaded[1].peaks["redo"]["peak_idx"],
            self.original[1].peaks["breaths"]["peak_idx"],
        )

    def test_loaded_group_ecg_detection(self):
        """get_ecg_peaks relies on ecg_idx set by EmgDataGroup.__init__."""
        dump = self.path("emg_ecg.json.gz")
        self.original.export_to_json(dump)
        loaded = EmgDataGroup.import_from_json(dump)
        loaded.get_ecg_peaks(name="ecg_redo", overwrite=True)
        np.testing.assert_array_equal(
            loaded[1].peaks["ecg_redo"]["peak_idx"],
            self.original[1].peaks["ecg"]["peak_idx"],
        )


class TestVentilatorRoundTrip(_TmpDirMixin, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.original = _build_vent_group()
        cls.dump_file = cls.path("vent.json.gz")
        cls.original.export_to_json(cls.dump_file)
        cls.loaded = VentilatorDataGroup.import_from_json(cls.dump_file)

    def test_returns_subclass(self):
        self.assertIs(type(self.loaded), VentilatorDataGroup)

    def test_subclass_init_attributes(self):
        for attr in ("p_vent_idx", "f_idx", "v_vent_idx", "peep"):
            self.assertEqual(
                getattr(self.loaded, attr), getattr(self.original, attr), attr
            )

    def test_pocc_peaks(self):
        ps_orig = self.original[self.original.p_vent_idx].peaks["Pocc"]
        ps_new = self.loaded[self.loaded.p_vent_idx].peaks["Pocc"]
        pd.testing.assert_frame_equal(ps_new.peak_df, ps_orig.peak_df)
        pd.testing.assert_frame_equal(
            ps_new.quality_outcomes_df, ps_orig.quality_outcomes_df
        )
        self.assertIn("PTPocc", ps_new.peak_df)

    def test_partial_y_data(self):
        """Only channel 0 has a baseline; others must not gain one."""
        self.assertIn("baseline", self.loaded[0]._y_data)
        for channel in self.loaded.channels[1:]:
            self.assertEqual(set(channel._y_data), {"raw"})


class TestEdgeCases(_TmpDirMixin, unittest.TestCase):
    def test_raw_only_group(self):
        group = _random_group(labels=["EMG1", "EMG2"])
        path = self.path("raw_only.json.gz")
        group.export_to_json(path)
        loaded = EmgDataGroup.import_from_json(path)
        for orig, new in zip(group.channels, loaded.channels, strict=True):
            self.assertEqual(set(new._y_data), {"raw"})
            self.assertEqual(new.peaks, {})
            np.testing.assert_array_equal(new["raw"], orig["raw"])

    def test_labels_none(self):
        group = _random_group(labels=None)
        path = self.path("no_labels.json.gz")
        group.export_to_json(path)
        loaded = EmgDataGroup.import_from_json(path)
        self.assertEqual(loaded.labels, [None, None])
        self.assertIsNone(loaded.ecg_idx)

    def test_custom_t_data_preserved(self):
        """A time axis with an offset must survive, not be rebuilt from fs."""
        fs = 256
        t_data = 10.0 + np.arange(10 * fs) / fs
        group = _random_group(labels=["A", "B"], t_data=t_data)
        path = self.path("offset_t.json.gz")
        group.export_to_json(path)
        loaded = EmgDataGroup.import_from_json(path)
        for channel in loaded.channels:
            np.testing.assert_array_equal(channel.t_data, t_data)

    def test_empty_peak_set(self):
        group = _random_group(labels=["A", "B"])
        group[0].set_peaks(
            peak_idxs=np.array([], dtype=int),
            signal=group[0]["raw"],
            peak_set_name="none",
        )
        path = self.path("empty_peaks.json.gz")
        group.export_to_json(path)
        loaded = EmgDataGroup.import_from_json(path)
        ps_new = loaded[0].peaks["none"]
        self.assertEqual(len(ps_new), 0)
        self.assertEqual(list(ps_new.peak_df.columns), ["peak_idx"])

    def test_object_type_mismatch_warns(self):
        group = _random_group(labels=["A", "B"])
        path = self.path("emg_as_vent.json.gz")
        group.export_to_json(path)
        with self.assertWarns(UserWarning):
            loaded = VentilatorDataGroup.import_from_json(path)
        self.assertIs(type(loaded), VentilatorDataGroup)

    def test_base_class_import(self):
        group = _random_group(labels=["A", "B"])
        path = self.path("base.json.gz")
        group.export_to_json(path)
        with self.assertWarns(UserWarning):
            loaded = TimeSeriesGroup.import_from_json(path)
        self.assertIs(type(loaded), TimeSeriesGroup)

    def test_missing_fs_raises(self):
        group = _random_group(labels=["A", "B"])
        path = self.path("no_fs.json.gz")
        group.export_to_json(path)
        dump = _read_dump(path)
        del dump["metadata"]["fs"]
        with gzip.open(path, "wt") as fout:
            json.dump(dump, fout)
        with self.assertRaises(KeyError):
            EmgDataGroup.import_from_json(path)

    def test_compression_level(self):
        group = _random_group(labels=["A", "B"])
        fast, small = self.path("lvl1.json.gz"), self.path("lvl9.json.gz")
        group.export_to_json(fast, compression_level=1)
        group.export_to_json(small, compression_level=9)
        self.assertGreaterEqual(os.path.getsize(fast), os.path.getsize(small))
        self.assertEqual(_sha256_gz(fast), _sha256_gz(small))


class TestPeaksSetDict(unittest.TestCase):
    def setUp(self):
        t_data = np.arange(100) / 10
        self.peak_set = PeaksSet(np.sin(t_data), t_data, peak_idxs=np.array([10, 50]))

    def test_t_data_excluded_by_default(self):
        self.assertNotIn("t_data", self.peak_set.to_dict())

    def test_t_data_optional(self):
        exported = self.peak_set.to_dict(export_time_data=True)
        np.testing.assert_array_equal(exported["t_data"], self.peak_set.t_data)

    def test_round_trip(self):
        restored = PeaksSet.from_dict(self.peak_set.to_dict(export_time_data=True))
        np.testing.assert_array_equal(restored.signal, self.peak_set.signal)
        pd.testing.assert_frame_equal(restored.peak_df, self.peak_set.peak_df)


class TestJsonHelpers(unittest.TestCase):
    def test_numpy_types(self):
        obj = {
            "i": np.int64(3),
            "f": np.float32(1.5),
            "b": np.bool_(True),
            "a": np.arange(4).reshape(2, 2),
        }
        out = json.loads(json.dumps(obj, default=fu._json_default))
        self.assertEqual(out, {"i": 3, "f": 1.5, "b": True, "a": [[0, 1], [2, 3]]})

    def test_dataframe_round_trip(self):
        df = pd.DataFrame(
            {"peak_idx": [10, 20], "valid": [True, False], "value": [0.1, np.nan]},
            index=[3, 7],
        )
        out = json.loads(
            json.dumps(df, default=fu._json_default), object_hook=fu._json_object_hook
        )
        pd.testing.assert_frame_equal(out, df)

    def test_nan_in_array(self):
        arr = np.array([1.0, np.nan, np.inf])
        out = json.loads(json.dumps(arr, default=fu._json_default))
        np.testing.assert_array_equal(np.asarray(out), arr)

    def test_unsupported_type_raises(self):
        with self.assertRaises(TypeError):
            json.dumps({"t": pd.Timestamp("2026-01-01")}, default=fu._json_default)


if __name__ == "__main__":
    unittest.main()
