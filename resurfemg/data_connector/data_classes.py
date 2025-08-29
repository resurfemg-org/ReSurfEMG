"""
Copyright 2022 Netherlands eScience Center and University of Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.

This file contains data classes for standardized data storage and method
automation.
"""

import warnings
from textwrap import wrap, dedent
import numpy as np
import scipy
import matplotlib.pyplot as plt

from resurfemg.helper_functions import math_operations as mo
from resurfemg.helper_functions import \
    data_classes_quality_assessment as data_qa
from resurfemg.preprocessing import filtering as filt
from resurfemg.preprocessing import ecg_removal as ecg_rm
from resurfemg.pipelines.processing import ecg_removal_gating
from resurfemg.preprocessing import envelope as evl
from resurfemg.postprocessing import baseline as bl
from resurfemg.postprocessing import event_detection as evt
from resurfemg.postprocessing import features as feat

from resurfemg.data_connector.peakset_class import PeaksSet


class TimeSeries:
    """
    Data class to store, process, and plot single channel time series data.
    Data is stored in the _y_data dictionary, which can be accessed using the
    TimeSeries[`signal_type`]. The default signal types are:
    'raw', 'filt', 'clean', 'env', 'env_ci', and 'baseline'.
    Defined properties:
    - t_data: time axis data
    - label: channel label
    - y_units: channel signal units
    - param: dictionary of channel parameters (fs, n_samp)
    - peaks: dictionary of PeaksSet objects
    """

    def __init__(self, y_raw, t_data=None, fs=None, label=None, units=None):
        """
        :param y_raw: 1-dimensional raw signal data
        :type y_raw: ~numpy.ndarray
        :param t_data: time axis data, if None, generated from fs
        :type t_data: ~numpy.ndarray
        :param fs: sampling rate, if None, calculated from t_data
        :type fs: ~int
        :param label: label of the channel
        :type label: str
        :param units: channel signal units
        :type units: str
        """
        self.param = dict()
        self.param['fs'] = fs
        y_raw = np.array(y_raw)
        data_dims = y_raw.ndim
        if data_dims == 1:
            self.param['n_samp'] = len(y_raw)
        elif data_dims == 2:
            if y_raw.shape[0] < y_raw.shape[1]:
                y_raw = y_raw.T
            self.param['n_samp'] = y_raw.shape[0]
        else:
            raise ValueError("Invalid y_raw dimensions")
        self._y_data = {
            'raw': y_raw,
        }
        self.peaks = {}

        if t_data is None and fs is None:
            self.t_data = np.arange(self.param['n_samp'])
        elif t_data is not None:
            if len(np.array(t_data).shape) > 1:
                raise ValueError
            self.t_data = np.array(t_data)
            if fs is None:
                self.param['fs'] = int(1/(t_data[1:]-t_data[:-1]))
        else:
            self.t_data = np.array(
                [x_i/fs for x_i in range(self.param['n_samp'])])

        self.label = label or ''
        self.y_units = units or '?'

    def __getattr__(self, item):
        default_y_data = ['raw', 'filt', 'clean', 'env', 'env_ci', 'baseline']
        if item.replace('y_', '') in default_y_data:
            key = item.replace('y_', '')
            warnings.warn("\n".join(wrap(dedent(
                f"""
                The attribute y_{key} is deprecated to allow for more, and
                custom, signal types. Use '{self.__class__.__name__}'[{key}]
                instead."""))), FutureWarning)
            return self._y_data.get(key, None)
        return object.__getattr__(self, item)

    def __getitem__(self, key):
        """
        Get the signal data of the requested type. The default signal types
        are: 'raw', 'filt', 'clean', 'env', 'env_ci', and 'baseline'.
        -----------------------------------------------------------------------
        :param key: one of 'raw', 'filt', 'clean', 'env', 'env_ci', 'baseline'
        or a custom signal type
        :type key: str

        :returns y_data: data of the selected signal type
        :rtype y_data: ~numpy.ndarray
        """
        return self._y_data.get(key)

    def __setitem__(self, key, value):
        """
        Set the signal data of the requested type. The default signal types
        are: 'raw', 'filt', 'clean', 'env', 'env_ci', and 'baseline'.
        -----------------------------------------------------------------------
        :param key: one of 'raw', 'filt', 'clean', 'env', 'env_ci', 'baseline'
        or a custom signal type
        :type key: str
        :param value: data of the selected signal type
        :type value: ~numpy.ndarray

        :returns: None
        :rtype: None
        """
        self._y_data[key] = value

    def __iter__(self):
        """
        Iterate over the signal data types in the _y_data dictionary.

        :returns: iterator over the signal data types
        :rtype: iterator
        """
        return iter(self._y_data)

    def signal_type_data(self, signal_type=None):
        """
        Automatically select the most advanced data type eligible for a
        subprocess ('env' {=envelope} > 'clean' > 'filt' > 'raw')
        -----------------------------------------------------------------------
        :param signal_type: one of 'env', 'clean', 'filt', or 'raw'
        :type signal_type: str

        :returns y_data: data of the selected signal type
        :rtype y_data: ~numpy.ndarray
        """
        y_data = np.zeros(self['raw'].shape)
        res_order = ['env', 'clean', 'filt', 'raw']
        if signal_type == 'env':
            if 'env' in self._y_data:
                y_data = self['env']
            else:
                raise KeyError('No envelope defined for this signal.')
        elif signal_type is None or signal_type in res_order:
            # If the signal type is one of the resolution order, find the data
            # or its next best option
            search_signal_type = signal_type or 'env'
            start_idx = res_order.index(search_signal_type)
            _res_order = res_order[start_idx:]
            for idx, option in enumerate(_res_order):
                if option in self._y_data:
                    y_data = self[option]
                    find_idx = idx
                    break
            else:
                raise KeyError(
                    f"Invalid signal type: {signal_type}"
                    f" or no data available for {signal_type}.")
            if signal_type is not None and find_idx > 0:
                warnings.warn("\n".join(wrap(dedent(
                     f"""Warning: No {signal_type} data available, using raw
                     data instead."""))))
        else:
            # If the signal type is not in the resolution order, find the data
            # or its next best option
            if signal_type in self._y_data:
                y_data = self[signal_type]
            else:
                raise KeyError(f"Invalid signal type: {signal_type}")
        return y_data

    def filter_emg(self, signal_io=('raw', 'filt'), hp_cf=20.0, lp_cf=500.0,
                   order=3, **kwargs):
        """
        Filter raw EMG signal to remove baseline wander and high frequency
        components. See preprocessing.emg_bandpass_butter submodule.
        The filtered signal is stored in self[signal_io[1]].

        :param signal_io: tuple of strings, the first element is the input
            the second element is the output signal type.
        :type signal_io: tuple

        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`, `output_name`)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'], 'filt')
        y_data = self.signal_type_data(signal_type=signal_io[0])
        # Eliminate the baseline wander from the data using a band-pass filter
        self[signal_io[1]] = filt.emg_bandpass_butter(
            y_data,
            high_pass=hp_cf,
            low_pass=lp_cf,
            fs_emg=self.param['fs'],
            order=order)

    def get_ecg_peaks(
            self, ecg_raw=None, bp_filter=True, overwrite=False, name='ecg'):
        """
        Detect ECG peaks in the provided signal. See preprocessing.ecg_removal
        submodule. ECG peaks are stored in the self.peaks dict under the key
        `name`. When no ECG channel is provided, the raw signal of the current
        TimeSeries object is used.
        When running get_ecg_peaks on a EmgDataGroup and no ECG channel or
        ecg_raw is provided, EmgDataGroup.ecg_idx is used to detect the QRS
        peak locations. ecg_idx is auto-detected from the labels on
        EmgDataGroup initialization, or can be set with the set_ecg_idx method.
        -----------------------------------------------------------------------
        :param ecg_raw: ECG signal, if None, the raw signal is used
        :type ecg_raw: ~numpy.ndarray
        :bp_filter: Apply band-pass filter to the ECG signal
        :type bp_filter: bool
        :overwrite: Overwrite existing peaks
        :type overwrite: bool
        :param name: Name of the peak set in the self.peaks dict
        :type name: str

        :returns: None
        :rtype: None
        """
        if name in self.peaks and not overwrite:
            raise UserWarning('ECG peaks already detected. Use overwrite=True')

        if ecg_raw is None:
            lp_cf = min([500.0, 0.95 * self.param['fs'] / 2])
            ecg_raw = filt.emg_bandpass_butter(
                self['raw'],
                high_pass=1,
                low_pass=lp_cf,
                fs_emg=self.param['fs'])

        ecg_peak_idxs = ecg_rm.detect_ecg_peaks(
            ecg_raw=ecg_raw,
            fs=self.param['fs'],
            bp_filter=bp_filter,
        )

        self.set_peaks(
            signal=ecg_raw,
            peak_idxs=ecg_peak_idxs,
            peak_set_name=name,
            overwrite=overwrite,
        )

    def gating(self, signal_io=('filt', 'clean'), ecg_peakset_name='ecg',
               gate_width_samples=None, fill_method=3, **kwargs):
        """
        Eliminate ECG artifacts from the provided signal based on the peak_idx
        of the provided PeakSet with `ecg_peakset_name`. See
        preprocessing.ecg_removal and pipelines.ecg_removal_gating submodules.
        The cleaned signal is by stored in self['clean'].

        :param signal_io: tuple of strings, the first element is the input
            the second element is the output signal type.
        :type signal_io: tuple

        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`, `output_name`)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'], 'filt')

        if any(key in kwargs for key in ['ecg_peak_idxs', 'ecg_raw',
                                         'bp_filter', 'overwrite']):
            warnings.warn("\n".join(wrap(dedent("""
                The kwargs 'ecg_peak_idxs', 'ecg_raw', 'bp_filter', 'overwrite'
                will be removed in future versions of ReSurfEMG:
                ECG removal will be split in QRS detection with
                `get_ecg_peaks` and ECG elimination (`gating`/
                `wavelet_denoising`). Alternatively, create an ECG peakset with
                `set_peaks` and use `gating` or `wavelet_denoising` directly.
                """))), FutureWarning)
            ecg_peak_idxs = kwargs.get('ecg_peak_idxs', None)
            bp_filter = kwargs.get('bp_filter', True)
            overwrite = kwargs.get('overwrite', False)
            if ecg_peak_idxs is None:
                ecg_raw = kwargs.get('ecg_raw', None)
                self.get_ecg_peaks(
                    ecg_raw=ecg_raw,
                    bp_filter=bp_filter,
                    overwrite=overwrite,
                    name=ecg_peakset_name)
            else:
                if ecg_peakset_name not in self.peaks and not overwrite:
                    raise UserWarning(
                        'ECG peaks already detected. Use overwrite=True')
                ecg_raw = (
                    kwargs.get('ecg_raw',
                               self.signal_type_data(signal_type='raw')))
                self.set_peaks(
                    signal=ecg_raw,
                    peak_idxs=ecg_peak_idxs,
                    peak_set_name=ecg_peakset_name,
                    overwrite=overwrite,
                )
        if ecg_peakset_name not in self.peaks:
            raise KeyError(
                f"""Peakset {ecg_peakset_name} does not yet exist. First detect
                 ECG peaks before using gating.""")
        y_data = self.signal_type_data(signal_type=signal_io[0])
        ecg_peak_idxs = self.peaks[ecg_peakset_name]['peak_idx']
        if gate_width_samples is None:
            gate_width_samples = self.param['fs'] // 10
        self[signal_io[1]] = ecg_removal_gating(
            y_data,
            ecg_peak_idxs,
            gate_width_samples,
            ecg_shift=10,
            method=fill_method,
        )

    def wavelet_denoising(
            self, signal_io=('filt', 'clean'), ecg_peakset_name='ecg', n=None,
            fixed_threshold=None, **kwargs):
        """
        Eliminate ECG artifacts from the provided signal. See
        preprocessing.wavelet_denoising submodules. The cleaned signal is
        by default stored in self['clean'].

        :param signal_io: tuple of strings, the first element is the input
            the second element is the output signal type.
        :type signal_io: tuple

        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`, `output_name`)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'], 'filt')

        if any(key in kwargs for key in ['ecg_peak_idxs', 'ecg_raw',
                                         'bp_filter', 'overwrite']):
            warnings.warn("\n".join(wrap(dedent("""
                The kwargs 'ecg_peak_idxs', 'ecg_raw', 'bp_filter', 'overwrite'
                will be removed in future versions of ReSurfEMG.
                ECG removal will be split in QRS detection with
                `get_ecg_peaks` and ECG elimination (`gating`/
                `wavelet_denoising`). Alternatively, create an ECG peakset with
                `set_peaks` and use `gating` or `wavelet_denoising` directly.
                """))), FutureWarning)
            ecg_peak_idxs = kwargs.get('ecg_peak_idxs', None)
            bp_filter = kwargs.get('bp_filter', True)
            overwrite = kwargs.get('overwrite', False)
            if ecg_peak_idxs is None:
                ecg_raw = kwargs.get('ecg_raw', None)
                self.get_ecg_peaks(
                    ecg_raw=ecg_raw,
                    bp_filter=bp_filter,
                    overwrite=overwrite,
                    name=ecg_peakset_name)
            else:
                if ecg_peakset_name not in self.peaks and not overwrite:
                    raise UserWarning(
                        'ECG peaks already detected. Use overwrite=True')
                ecg_raw = (
                    kwargs.get('ecg_raw',
                               self.signal_type_data(signal_type='raw')))
                self.set_peaks(
                    signal=ecg_raw,
                    peak_idxs=ecg_peak_idxs,
                    peak_set_name=ecg_peakset_name,
                    overwrite=overwrite,
                )
        if ecg_peakset_name not in self.peaks:
            raise KeyError(
                f"""ECG peakset {ecg_peakset_name}. First detect ECG peaks
                before using wavelet denoising.""")
        y_data = self.signal_type_data(signal_type=signal_io[0])
        ecg_peak_idxs = self.peaks['ecg']['peak_idx']
        if n is None:
            n = int(np.log(self.param['fs']/20) // np.log(2))

        if fixed_threshold is None:
            fixed_threshold = 4.5

        self[signal_io[1]], *_ = ecg_rm.wavelet_denoising(
            y_data, ecg_peak_idxs, fs=self.param['fs'], hard_thresholding=True,
            n=n, fixed_threshold=fixed_threshold, wavelet_type='db2')

    def envelope(self, env_window=None, env_type=None,
                 signal_io=('clean', 'env'), ci_alpha=None, **kwargs):
        """
        Derive the moving envelope of the provided signal. See
        preprocessing.envelope submodule. The envelope is by default stored in
        the self['env']. If ci_alpha is not None, the confidence interval is
        stored in self['env_ci'].

        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`, `output_name`)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'], 'env')
        if env_window is None:
            if 'fs' in self.param:
                env_window = self.param['fs'] // 4
            else:
                raise ValueError(
                    'Evelope window and sampling rate are not defined.')

        y_data = self.signal_type_data(signal_type=signal_io[0])
        if env_type == 'rms' or env_type is None:
            self[signal_io[1]] = evl.full_rolling_rms(y_data, env_window)
            if ci_alpha is not None:
                self[signal_io[1] + '_ci'] = evl.rolling_rms_ci(
                    y_data, env_window, alpha=ci_alpha)
        elif env_type == 'arv':
            self[signal_io[1]] = evl.full_rolling_arv(y_data, env_window)
            if ci_alpha is not None:
                self[signal_io[1] + '_ci'] = evl.rolling_arv_ci(
                    y_data, env_window, alpha=ci_alpha)
        else:
            raise ValueError('Invalid envelope type')

    def baseline(
            self, percentile=33, window_s=None, step_s=None,
            base_method='default', signal_io=(None, 'baseline'),
            augm_percentile=25, ma_window=None, perc_window=None, **kwargs):
        """
        Derive the moving baseline of the provided signal. See
        postprocessing.baseline submodule. The baseline is stored in the
        self['baseline'] attribute.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`, `output_name`)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'], 'baseline')
        window_s = window_s or int(7.5 * self.param.get('fs', 1))
        step_s = step_s or self.param.get('fs', 5) // 5
        if signal_io[0] is None:
            signal_io = ('env', 'baseline')

        y_baseline_data = self.signal_type_data(signal_type=signal_io[0])
        if base_method in ('default', 'moving_baseline'):
            self[signal_io[1]] = bl.moving_baseline(
                y_baseline_data, window_s=window_s, step_s=step_s,
                set_percentile=percentile)
        elif base_method == 'slopesum_baseline':
            if 'fs' not in self.param:
                raise ValueError(
                    'Sampling rate is not defined.')
            self[signal_io[1]], _, _, _ = bl.slopesum_baseline(
                    y_baseline_data, window_s=window_s, step_s=step_s,
                    fs=self.param['fs'], set_percentile=percentile,
                    augm_percentile=augm_percentile, ma_window=ma_window,
                    perc_window=perc_window)
        else:
            raise ValueError('Invalid method')

    def set_peaks(self, peak_idxs, signal, peak_set_name, overwrite=False):
        """
        Store a new PeaksSet object in the self.peaks dict under the key
        peak_set_name.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        if peak_set_name in self.peaks and not overwrite:
            raise UserWarning('PeaksSet already exists. Use overwrite=True')
        self.peaks[peak_set_name] = PeaksSet(
            peak_idxs=peak_idxs, t_data=self.t_data, signal=signal)

    def detect_emg_breaths(
        self,
        threshold=0,
        prominence_factor=0.5,
        min_peak_width_s=None,
        peak_set_name='breaths',
        start_idx=0,
        end_idx=None,
        overwrite=False,
        signal_io=(('env', 'baseline'),),
    ):
        """
        Find breath peaks in provided EMG envelope signal. See
        postprocessing.event_detection submodule. The peaks are stored in the
        self.peaks dict under the key peak_set_name.
        -----------------------------------------------------------------------
        :returns: None
        :rtype: None
        """
        if signal_io[0][0] not in self._y_data:
            raise ValueError(f'Envelope ({signal_io[0][0]}) not yet defined.')

        y_baseline = (self[signal_io[0][1]] if signal_io[0][1] in self._y_data
                      else np.zeros(self[signal_io[0][0]].shape))
        if signal_io[0][1] not in self._y_data:
            warnings.warn("\n".join(wrap(dedent(
                """EMG baseline ({signal_io[0][1]}) not yet defined. Peak
                detection relative to zero."""))))

        if ((end_idx is not None and end_idx > len(self[signal_io[0][0]]))
                or start_idx > len(self[signal_io[0][0]])):
            raise ValueError('Index out of range.')

        end_idx = end_idx or len(self[signal_io[0][0]])
        if end_idx < start_idx:
            raise ValueError('End index smaller than start index.')

        min_peak_width_s = min_peak_width_s or self.param['fs'] // 5

        peak_idxs = evt.detect_emg_breaths(
            self[signal_io[0][0]][start_idx:end_idx],
            y_baseline[start_idx:end_idx], threshold=threshold,
            prominence_factor=prominence_factor,
            min_peak_width_s=min_peak_width_s)
        peak_idxs += start_idx
        self.set_peaks(peak_idxs=peak_idxs, signal=self[signal_io[0][0]],
                       peak_set_name=peak_set_name, overwrite=overwrite)

    def link_peak_set(self, peak_set_name, t_reference_peaks,
                      linked_peak_set_name=None):
        """
        Find the peaks in the PeaksSet with the peak_set_name closest in time
        to the provided peak timings in t_reference_peaks. The results are
        stored in a new PeaksSet object in the self.peaks dict under the key
        linked_peak_set_name. If no linked_peak_set_name is provided, the key
        is set to peak_set_name + '_linked'.
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param t_reference_peaks: Refernce peak timings in t_reference_peaks
        :type t_reference_peaks: ~numpy.ndarray
        :param linked_peak_set_name: Name of the new PeaksSet
        :type linked_peak_set_name: str

        :return: None
        :rtype: None
        """
        if peak_set_name not in self.peaks:
            raise KeyError("Non-existent PeaksSet key")

        peak_set = self.peaks[peak_set_name]
        linked_peak_set_name = \
            linked_peak_set_name or peak_set_name + '_linked'
        t_peakset_peaks = peak_set['peak_idx'] / self.param['fs']
        link_peak_nrs = evt.find_linked_peaks(
            t_reference_peaks, t_peakset_peaks)

        self.peaks[linked_peak_set_name] = PeaksSet(
            peak_set.signal, peak_set.t_data, peak_idxs=None
        )
        for attr in ['peak_df', 'quality_values_df', 'quality_outcomes_df']:
            setattr(self.peaks[linked_peak_set_name], attr,
                    getattr(peak_set, attr).loc[link_peak_nrs].reset_index(
                        drop=True))

    def calculate_time_products(
            self, peak_set_name, include_aub=True, aub_window_s=None,
            aub_reference_signal=None, parameter_name=None,
            signal_io=('baseline',)):
        """
        Calculate the time product, i.e. area under the curve for a PeaksSet.
        The results are stored as
        self.peaks[peak_set_name].peak_df[parameter_name]. If no parameter_name
        is provided, parameter_name = 'time_product'
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param include_aub: Include the area under the baseline in the time
        product
        :type include_aub: bool
        :param signal_io: tuple of strings, the first element is the input
        baseline type.
        :type signal_io: tuple
        :param aub_window_s: window length in samples in which the local
        extreme is sought.
        :param aub_window_s: int
        :param aub_reference_signal: Optional reference signal to find the
        local extreme in, else the signal underlying the PeaksSet is taken.
        :type aub_reference_signal: ~numpy.ndarray
        :param parameter_name: parameter name in Dataframe
        self.peaks[peak_set_name].peak_df
        :type parameter_name: str

        :returns: None
        :rtype: None
        """
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
            raise KeyError("Non-existent PeaksSet key")

        if signal_io[0] not in self._y_data:
            if include_aub:
                raise ValueError(
                    'Baseline in not yet defined, but is required to calculate'
                    + ' the area under the baseline.')
            else:
                warnings.warn("\n".join(wrap(dedent(
                    """Baseline in not yet defined. Calculating time-product
                    with reference to 0."""))))
                baseline = np.zeros(peak_set.signal.shape)
        else:
            baseline = self[signal_io[0]]

        time_products = feat.time_product(
            signal=peak_set.signal, fs=self.param['fs'],
            start_idxs=peak_set['start_idx'], end_idxs=peak_set['end_idx'],
            baseline=baseline)

        if include_aub:
            aub_window_s = aub_window_s or 5 * self.param['fs']
            aub_reference_signal = (
                peak_set.signal if aub_reference_signal is None
                else aub_reference_signal)
            aub, y_refs = feat.area_under_baseline(
                signal=peak_set.signal, fs=self.param['fs'],
                start_idxs=peak_set['start_idx'],
                peak_idxs=peak_set['peak_idx'], end_idxs=peak_set['end_idx'],
                aub_window_s=aub_window_s, baseline=baseline,
                ref_signal=aub_reference_signal)
            peak_set.peak_df['AUB'] = aub
            peak_set.peak_df['aub_y_ref'] = y_refs
            time_products += aub

        peak_set.peak_df[parameter_name or 'time_product'] = time_products

    def test_emg_quality(self, peak_set_name, cutoff=None, skip_tests=None,
                         parameter_names=None, verbose=True):
        """See helper_functions.data_classes_quality_assessment submodule. The
        results are stored in the self.peaks[peak_set_name].quality_outcomes_df
        and self.peaks[peak_set_name].quality_values_df DataFrames.
        """
        data_qa.test_emg_quality(
            self, peak_set_name, cutoff, skip_tests, parameter_names, verbose)

    def test_pocc_quality(self, peak_set_name, cutoff=None, skip_tests=None,
                          parameter_names=None, verbose=True):
        """See helper_functions.data_classes_quality_assessment submodule. The
        results are stored in the self.peaks[peak_set_name].quality_outcomes_df
        and self.peaks[peak_set_name].quality_values_df DataFrames."""
        data_qa.test_pocc_quality(
            self, peak_set_name, cutoff, skip_tests, parameter_names, verbose)

    def test_linked_peak_sets(
            self, peak_set_name, linked_timeseries, linked_peak_set_name,
            parameter_names=None, cutoff=None, skip_tests=None, verbose=True):
        """See helper_functions.data_classes_quality_assessment submodule. The
        results are stored in the self.peaks[peak_set_name].quality_outcomes_df
        and self.peaks[peak_set_name].quality_values_df DataFrames."""
        data_qa.test_linked_peak_sets(
            self, peak_set_name, linked_timeseries, linked_peak_set_name,
            parameter_names, cutoff, skip_tests, verbose)

    def plot_full(self, axes=None, signal_io=(None,), colors=None,
                  baseline_bool=True, plot_ci=False, **kwargs):
        """Plot the indicated signals in the provided axes. By default the most
        advanced signal type (envelope > clean > filt > raw) is plotted in the
        provided colours.
        -----------------------------------------------------------------------
        :param axes: matplotlib Axis object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axis
        :param signal_io: tuple of strings, the first element is the input
        signal type.
        :type signal_io: tuple
        :param colors: list of colors to plot the 1) signal, 2) the baseline
        :type colors: list
        :param baseline_bool: plot the baseline
        :type baseline_bool: bool

        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`,)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'], 'env')
        axis = axes if axes is not None else plt.subplots()[1]
        colors = colors if colors is not None else [
            'tab:blue', 'tab:orange', 'tab:red', 'tab:cyan', 'tab:green']

        y_data = self.signal_type_data(signal_type=signal_io[0])
        axis.grid(True)
        axis.plot(self.t_data, y_data, color=colors[0])
        axis.set_ylabel(self.label + ' (' + self.y_units + ')')

        if (baseline_bool is True
                and self['baseline'] is not None
                and np.any(~np.isnan(self['baseline']), axis=0)):
            axis.plot(self.t_data, self['baseline'], color=colors[1])
        if plot_ci and self['env_ci'] is not None:
            axis.fill_between(self.t_data, self['env_ci'][0],
                              self['env_ci'][1], color=colors[0], alpha=0.5)

    def plot_markers(self, peak_set_name, axes, valid_only=False,
                     colors=None, markers=None):
        """Plot the markers for the peak set in the provided axes in the
        provided colours using the provided markers.
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: 1 color of list of up to 3 colors for the markers, peak,
        start, and end markers. If 2 colors are provided, start and end have
        the same colors
        :type colors: str or list
        :param markers: 1 markers or list of up to 3 markers for peak, start,
        and end markers. If 2 markers are provided, start and end have the same
        marker
        :type markers: str or list

        :returns: None
        :rtype: None
        """
        if peak_set_name not in self.peaks:
            raise KeyError("Non-existent PeaksSet key")

        peak_set = self.peaks[peak_set_name]
        valid = (peak_set['valid'] if valid_only and
                 'valid' in peak_set.peak_df.columns
                 else np.ones(len(peak_set['peak_idx']), dtype=bool))

        def get_values(column):
            return ((peak_set.t_data[peak_set[column]][valid],
                     peak_set.signal[peak_set[column]][valid])
                    if column in peak_set.peak_df.columns
                    else (len(peak_set) * [None], len(peak_set) * [None]))

        x_vals_peak, y_vals_peak = get_values('peak_idx')
        x_vals_start, y_vals_start = get_values('start_idx')
        x_vals_end, y_vals_end = get_values('end_idx')

        def get_color_marker(values, default):
            if values is None:
                return [default] * 3
            if isinstance(values, str):
                return [values] * 3
            if isinstance(values, list):
                return values[:3] if len(values) > 2 else [
                    values[0], values[1], values[1]]
            raise ValueError(f'Invalid {default}')

        peak_color, start_color, end_color = get_color_marker(
            colors, 'tab:red')
        peak_marker, start_marker, end_marker = get_color_marker(markers, '*')

        if len(np.atleast_1d(axes)) == 1 and len(x_vals_peak) > 1:
            axes = np.matlib.repmat(axes, len(x_vals_peak), 1).flatten()
        for axis, x_peak, y_peak, x_start, y_start, x_end, y_end in zip(
                np.atleast_1d(axes), x_vals_peak, y_vals_peak, x_vals_start,
                y_vals_start, x_vals_end, y_vals_end):

            axis.plot(x_peak, y_peak, marker=peak_marker, color=peak_color,
                      linestyle='None')
            if x_start is not None:
                axis.plot(x_start, y_start, marker=start_marker,
                          color=start_color, linestyle='None')
            if x_end is not None:
                axis.plot(x_end, y_end, marker=end_marker, color=end_color,
                          linestyle='None')

    def plot_peaks(self, peak_set_name, axes=None, signal_io=(None,),
                   margin_s=None, valid_only=False, colors=None,
                   baseline_bool=True, plot_ci=False, **kwargs):
        """Plot the indicated peaks in the provided axes. By default the most
        advanced signal type (envelope > clean > filt > raw) is plotted in the
        provided colours.
        -----------------------------------------------------------------------
        :param peak_set_name: The name of the peak_set to be plotted.
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param signal_io: tuple of strings, the first element is the input
        signal type.
        :type signal_io: tuple
        :param margin_s: margins in samples plotted before the peak onset and
        after the peak offset
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: list of colors to plot the 1) signal, 2) the baseline
        :type colors: list
        :param baseline_bool: plot the baseline
        :type baseline_bool: bool

        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`,)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'],)
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
            raise KeyError("Non-existent PeaksSet key")

        start_idxs, end_idxs = peak_set['start_idx'], peak_set['end_idx']
        if valid_only and peak_set.valid is not None:
            valid = peak_set.valid
            start_idxs, end_idxs = start_idxs[valid], end_idxs[valid]

        if axes is None:
            _, axes = plt.subplots(nrows=1, ncols=len(start_idxs), sharey=True)
        axes = np.atleast_1d(axes)
        colors = colors if colors is not None else [
            'tab:blue', 'tab:orange', 'tab:red', 'tab:cyan', 'tab:green']
        y_data = (peak_set.signal if signal_io[0] is None
                  else self.signal_type_data(signal_type=signal_io[0]))
        m_s = margin_s if margin_s is not None else self.param['fs'] // 2
        ci = self['env_ci']
        for axis, x_start, x_end in zip(axes, start_idxs, end_idxs):
            s_start, s_end = max(0, x_start - m_s), max(0, x_end + m_s)
            axis.grid(True)
            axis.plot(self.t_data[s_start:s_end], y_data[s_start:s_end],
                      color=colors[0])
            if baseline_bool and self['baseline'] is not None and np.any(
                    ~np.isnan(self['baseline']), axis=0):
                axis.plot(self.t_data[s_start:s_end],
                          self['baseline'][s_start:s_end], color=colors[1])
            if plot_ci and self['env_ci'] is not None:
                axis.fill_between(
                    self.t_data[s_start:s_end], ci[0][s_start:s_end],
                    ci[1][s_start:s_end], color=colors[0], alpha=0.5)

        axes[0].set_ylabel(f"{self.label} ({self.y_units})")

    def plot_curve_fits(self, peak_set_name, axes, valid_only=False,
                        colors=None):
        """Plot the curve-fits for the peak set in the provided axes in the
        provided colours using the provided markers.
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: 1 color or list of colors for the fitted curve peak
        :type colors: str or list

        :returns: None
        :rtype: None
        """
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
            raise KeyError("Non-existent PeaksSet key")

        axes = np.atleast_1d(axes)
        required_params = ['y_min', 'a', 'b', 'c']
        missing_params = [
            param for param in required_params
            if f'bell_{param}' not in peak_set.peak_df.columns]
        if missing_params:
            raise KeyError(
                f"Missing parameters in PeaksSet: {', '.join(missing_params)}")

        plot_peak_df = (peak_set.peak_df.loc[peak_set.peak_df['valid']]
                        if valid_only and 'valid' in peak_set.peak_df.columns
                        else peak_set.peak_df)
        color = \
            colors[0] if isinstance(colors, list) and colors else 'tab:green'

        for axis, (_, row) in zip(axes, plot_peak_df.iterrows()):
            y_bell = mo.bell_curve(
                peak_set.t_data[row.start_idx:row.end_idx], a=row.bell_a,
                b=row.bell_b, c=row.bell_c)
            axis.plot(peak_set.t_data[row.start_idx:row.end_idx],
                      row.bell_y_min + y_bell, color=color)

        if len(axes) > 1:
            for _, (axis, (_, row)) in enumerate(zip(
                        axes, plot_peak_df.iterrows())):
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx:row.end_idx],
                    a=row.bell_a, b=row.bell_b, c=row.bell_c)
                axis.plot(peak_set.t_data[row.start_idx:row.end_idx],
                          row.bell_y_min + y_bell, color=color)
        else:
            for _, row in plot_peak_df.iterrows():
                y_bell = mo.bell_curve(
                    peak_set.t_data[row.start_idx:row.end_idx],
                    a=row.bell_a, b=row.bell_b, c=row.bell_c)
                axes[0].plot(peak_set.t_data[row.start_idx:row.end_idx],
                             row.bell_y_min + y_bell, color=color)

    def plot_aub(self, peak_set_name, axes, signal_io=None,
                 valid_only=False, colors=None, **kwargs):
        """Plot the area under the baseline (AUB) for the peak set in the
        provided axes in the provided colours using the provided markers.
        -----------------------------------------------------------------------
        :param peak_set_name: PeaksSet name in self.peaks dict
        :type peak_set_name: str
        :param axes: matplotlib Axes object. If none provided, a new figure is
        created.
        :type axes: matplotlib.Axes
        :param signal_type: the signal ('env', 'clean', 'filt', 'raw') to plot
        :type signal_type: str
        :param valid_only: when True, only valid peaks are plotted.
        :type valid_only: bool
        :param colors: 1 color or list of up to 3 colors for the peak
        :type colors: str or list

        :returns: None
        :rtype: None
        """
        if 'signal_type' in kwargs:
            warnings.warn("\n".join(wrap(dedent(
                """The 'signal_type' argument is deprecated. Use 'signal_io'
                instead: signal_io = (`input_name`,)."""))),
                FutureWarning)
            signal_io = (kwargs['signal_type'],)
        elif signal_io is None:
            raise ValueError('Signal type not provided. Use signal_io.')
        peak_set = self.peaks.get(peak_set_name)
        if peak_set is None:
            raise KeyError("Non-existent PeaksSet key")

        axes = np.atleast_1d(axes)

        if 'aub_y_ref' not in peak_set.peak_df.columns:
            raise KeyError("aub_y_ref not included in PeaksSet, area under the"
                           " baseline is not evaluated yet.")

        y_data = (peak_set.signal if signal_io[0] is None
                  else self.signal_type_data(signal_type=signal_io[0]))

        plot_peak_df = (peak_set.peak_df.loc[peak_set.peak_df['valid']]
                        if valid_only and 'valid' in peak_set.peak_df.columns
                        else peak_set.peak_df)

        colors = colors[0] if isinstance(colors, list) else colors

        color = (colors if isinstance(colors, str)
                 else colors[0] if isinstance(colors, list) and colors
                 else 'tab:cyan')

        if len(axes) > 1:
            for _, (axis, (_, row)) in enumerate(zip(
                        axes, plot_peak_df.iterrows())):
                axis.plot(peak_set.t_data[[row.start_idx, row.end_idx]],
                          [row.aub_y_ref, row.aub_y_ref], color=color)
                axis.plot(peak_set.t_data[[row.start_idx, row.start_idx]],
                          [y_data[row.start_idx], row.aub_y_ref], color=color)
                axis.plot(peak_set.t_data[[row.end_idx, row.end_idx]],
                          [y_data[row.end_idx], row.aub_y_ref], color=color)
        else:
            for _, row in plot_peak_df.iterrows():
                axes[0].plot(peak_set.t_data[[row.start_idx, row.end_idx]],
                             [row.aub_y_ref, row.aub_y_ref], color=color)
                axes[0].plot(peak_set.t_data[[row.start_idx, row.start_idx]],
                             [y_data[row.start_idx], row.aub_y_ref],
                             color=color)
                axes[0].plot(peak_set.t_data[[row.end_idx, row.end_idx]],
                             [y_data[row.end_idx], row.aub_y_ref],
                             color=color)


class TimeSeriesGroup:
    """
    Data class to store, process, and plot time series data. TimeSeriesGroup is
    a collection of TimeSeries objects. Enclosed TimeSeries objects can be
    indexed by index number or channel label.
    TimeSeries methods in TimeSeriesGroup._available_methods can be run on all
    or a subset of the TimeSeries objects through the TimeSeriesGroup.run
    method."""

    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        """
        :param y_raw: raw signal data
        :type y_raw: ~numpy.ndarray
        :param t_data: time axis data, if None, generated from fs
        :type t_data: ~numpy.ndarray
        :param fs: sampling rate, if None, calculated from t_data
        :type fs: ~int
        :param labels: list of labels, one per provided channel
        :type labels: ~list
        :param units: list of signal units, one per provided channel
        :type units: ~list

        :returns: None
        :rtype: None
        """
        self.channels = []
        self.param = dict()
        self.param['fs'] = fs
        self._available_methods = [
            'envelope', 'baseline', 'plot_full', 'plot_peaks', 'plot_markers',
            'set_peaks'
        ]
        data_shape = list(np.array(y_raw).shape)
        data_dims = len(data_shape)
        if data_dims == 1:
            self.param['n_samp'] = len(y_raw)
            self.param['n_channel'] = 1
            y_raw = np.array(y_raw).reshape((1, self.param['n_samp']))
        elif data_dims == 2:
            self.param['n_samp'] = data_shape[np.argmax(data_shape)]
            self.param['n_channel'] = data_shape[np.argmin(data_shape)]
            if np.argmin(data_shape) == 0:
                y_raw = np.array(y_raw)
            else:
                y_raw = np.array(y_raw).T
        else:
            raise ValueError('Invalid data dimensions')

        if t_data is None and fs is None:
            t_data = np.arange(self.param['n_samp'])
        elif t_data is not None:
            if len(np.array(t_data).shape) > 1:
                raise ValueError('Invalid time data dimensions')
            t_data = np.array(t_data)
            if fs is None:
                fs = int(1/(t_data[1:]-t_data[:-1]))
        else:
            t_data = np.array(
                [x_i/fs for x_i in range(self.param['n_samp'])])

        if labels is None:
            self.labels = self.param['n_channel'] * [None]
        else:
            if len(labels) != self.param['n_channel']:
                raise ValueError('Number of labels does not match the number'
                                 + ' of data channels.')
            self.labels = labels

        if units is None:
            self.y_units = self.param['n_channel'] * ['N/A']
        else:
            if len(labels) != self.param['n_channel']:
                raise ValueError
            self.y_units = units

        for idx in range(self.param['n_channel']):
            new_timeseries = TimeSeries(
                y_raw=y_raw[idx, :],
                t_data=t_data,
                fs=fs,
                label=self.labels[idx],
                units=self.y_units[idx]
            )
            self.channels.append(new_timeseries)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.channels[key]
        elif isinstance(key, str):
            for channel in self.channels:
                if channel.label == key:
                    return channel
            raise KeyError('Channel not found')
        else:
            raise ValueError('Invalid key type')

    def __iter__(self):
        return iter(self.channels)

    def to_numpy(self, channel_idxs=None, signal_io=(None,)):
        """Convert the TimeSeriesGroup to a numpy array. The output is a 2D
        array with the shape (n_channels, n_samples). The signal type is
        determined by the signal_io parameter. If signal_io is (None,), the
        most advanced signal type (envelope > clean > filt > raw) is used.

        :param channel_idxs: channel indices to convert to numpy array
        :type channel_idxs: ~numpy.ndarray or int
        :param signal_io: tuple of strings, the first element is the input
            signal type.
        :type signal_io: tuple

        :returns: numpy array of the TimeSeriesGroup data
        :rtype: ~numpy.ndarray
        """
        if channel_idxs is None:
            channel_idxs = np.arange(self.param['n_channel'])
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])
        return np.array([self.channels[idx].signal_type_data(signal_io[0])
                         for idx in channel_idxs])

    def run(self, method, channel_idxs=None, **kwargs):
        """Run a TimeSeries function on the provided channels in the group. The
        function is run with the provided keyword arguments.
        -----------------------------------------------------------------------
        :param method: method to run on the TimeSeries objects
        :type method: str
        :param channel_idxs: channel indices to run the method on
        :type channel_idxs: ~numpy.ndarray or int
        :param kwargs: keyword arguments for the method
        """
        if channel_idxs is None:
            channel_idxs = np.arange(self.param['n_channel'])
        elif isinstance(channel_idxs, int):
            channel_idxs = np.array([channel_idxs])

        if method in self._available_methods:
            if method == 'get_ecg_peaks':
                if 'ecg_raw' in kwargs and kwargs['ecg_raw'] is not None:
                    print('Provided raw ECG used for ECG peak detection.')
                elif self.ecg_idx is not None:
                    kwargs['ecg_raw'] = self[self.ecg_idx]['raw']
                    print('Set ECG channel used for ECG peak detection.')
                else:
                    print(
                        'Channel raw signals used for ECG peak detection.')

            if method in ['gating', 'wavelet_denoising']:
                if (all(not kwargs.get('ecg_peakset_name', 'ecg') in
                        channel.peaks for channel in self.channels) or
                    any(key in kwargs for key in ['ecg_peak_idxs', 'ecg_raw',
                                                  'bp_filter', 'overwrite'])):
                    warnings.warn("\n".join(wrap(dedent(
                        """
                        The kwargs 'ecg_peak_idxs', 'ecg_raw', 'bp_filter',
                        'overwrite' will be removed from the gating and wavelet
                        denoising methods in future versions of ReSurfEMG:
                        ECG removal will be split in QRS detection with
                        `get_ecg_peaks` and ECG elimination (`gating`/
                        `wavelet_denoising`):
                        TimeSeriesGroup.run('get_ecg_peaks', ...)
                        Before running:
                        TimeSeriesGroup.run('gating', ...) or
                        TimeSeriesGroup.run('wavelet_denoising', ...)\n
                        Alternatively, create an ECG peakset with `set_peaks`
                        and use `gating` or `wavelet_denoising` directly."""
                    ))), FutureWarning)
                    ecg_kwargs = {
                        'channel_idxs': channel_idxs,
                        'overwrite': kwargs.pop('overwrite', False)}
                    if 'ecg_peak_idxs' in kwargs:
                        ecg_kwargs['ecg_peak_idxs'] = kwargs.pop(
                            'ecg_peak_idxs')
                        self.run('set_peaks', **ecg_kwargs)
                    else:
                        ecg_kwargs = {
                            'ecg_raw': kwargs.pop('ecg_raw', None),
                            'bp_filter': kwargs.pop('bp_filter', True)}
                        self.run('get_ecg_peaks', **ecg_kwargs)

            elif method.startswith('plot_'):
                if 'axes' not in kwargs:
                    _, kwargs['axes'] = plt.subplots(
                        nrows=len(channel_idxs), ncols=1, figsize=(10, 6),
                        sharex=True)
                if method == 'plot_full':
                    kwargs['axes'] = np.atleast_1d(kwargs['axes'])

                    if len(channel_idxs) > len(kwargs['axes']):
                        raise ValueError("Provided axes have not enough rows "
                                         "for all channels to plot.")
                    elif len(channel_idxs) < len(kwargs['axes']):
                        warnings.warn("\n".join(wrap(dedent(
                            'More axes provided than channels to plot.'))))
                elif method in ['plot_peaks', 'plot_markers']:
                    kwargs['axes'] = np.atleast_2d(kwargs['axes'])

                    if kwargs['axes'].shape[0] < len(channel_idxs):
                        raise ValueError("Provided axes have not enough rows "
                                         "for all channels to plot.")
                    if 'peak_set_name' not in kwargs:
                        raise ValueError('No peak_set_name provided.')
            _kwargs = kwargs.copy()
            for idx, channel_idx in enumerate(channel_idxs):
                if method.startswith('plot_'):
                    if method in ['plot_peaks', 'plot_markers']:
                        _kwargs['axes'] = kwargs['axes'][idx, :]
                    else:
                        _kwargs['axes'] = kwargs['axes'][idx]
                getattr(self.channels[channel_idx], method)(**_kwargs)
        else:
            raise ValueError('Invalid method')


class EmgDataGroup(TimeSeriesGroup):
    """Child-class of TimeSeriesGroup to store and handle emg data in with the
    additional methods filter_emg, gating, and wavelet_denoising. Enclosed
    TimeSeries objects can be indexed by index number or channel label.
    TimeSeries methods in TimeSeriesGroup._available_methods can be run on all
    or a subset of the TimeSeries objects through the TimeSeriesGroup.run
    method.
    When running ECG elimination methods (gating, wavelet_denoising) and no ECG
    channel or ecg_raw is provided, EmgDataGroup.ecg_idx is used to detect the
    QRS peak locations. ecg_idx is auto-detected from the labels on
    EmgDataGroup initialization, or can be set with the set_ecg_idx method.
    When no ECG channel is provided, the raw signal per TimeSeries object is
    used."""
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)
        self._available_methods.append('filter_emg')
        self._available_methods.append('get_ecg_peaks')
        self._available_methods.append('gating')
        self._available_methods.append('wavelet_denoising')

        labels_lc = [label.lower() for label in labels]
        if 'ecg' in labels_lc:
            self.ecg_idx = labels_lc.index('ecg')
            print('Auto-detected ECG channel from labels.')
        else:
            print("No ECG channel detected. Set ECG channel index with "
                  "`EmgDataGroup.set_ecg_idx(arg)` method.")
            self.ecg_idx = None

    def set_ecg_idx(self, ecg_idx):
        """
        Set the ECG channel index in the group.
        -----------------------------------------------------------------------
        :param ecg_idx: ECG channel index or label
        :type ecg_idx: int

        :returns: None
        :rtype: None
        """
        if isinstance(ecg_idx, int):
            self.ecg_idx = ecg_idx
        elif isinstance(ecg_idx, str):
            self.ecg_idx = self.labels.index(ecg_idx)


class VentilatorDataGroup(TimeSeriesGroup):
    """
    Child-class of TimeSeriesGroup to store and handle ventilator data in.
    Default channels are 'Paw'/ 'Pvent', 'F', and 'Vvent', which are auto-
    detected from the labels. The PEEP-level (VentilatorDataGroup.peep) is
    auto-detected from the pressure channel when a pressure channel is set.
    """
    def __init__(self, y_raw, t_data=None, fs=None, labels=None, units=None):
        super().__init__(
            y_raw, t_data=t_data, fs=fs, labels=labels, units=units)

        self.p_vent_idx = next((labels.index(label)
                                for label in ['Paw', 'Pvent']
                                if label in labels), None)
        self.f_idx = labels.index('F') if 'F' in labels else None
        self.v_vent_idx = labels.index('Vvent') if 'Vvent' in labels else None

        if self.p_vent_idx is not None:
            print('Auto-detected Pvent channel from labels.')
        if self.f_idx is not None:
            print('Auto-detected Flow channel from labels.')
        if self.v_vent_idx is not None:
            print('Auto-detected Volume channel from labels.')

        if self.p_vent_idx is not None and self.v_vent_idx is not None:
            self.find_peep(self.p_vent_idx, self.v_vent_idx)
        else:
            self.peep = None

    def find_peep(self, pressure_idx, volume_idx):
        """
        Calculate PEEP as the median value of p_vent at end-expiration.
        -----------------------------------------------------------------------
        :param pressure_idx: Channel index of the ventilator pressure data
        :type pressure_idx: int
        :param volume_idx: Channel index of the ventilator volume data
        :type volume_idx: int

        :returns: None
        :rtype: None
        """
        pressure_idx = pressure_idx or self.p_vent_idx
        if pressure_idx is None:
            raise ValueError('pressure_idx and self.p_vent_idx not defined')

        volume_idx = volume_idx or self.v_vent_idx
        if volume_idx is None:
            raise ValueError('volume_idx and self.v_vent_idx not defined')

        v_ee_pks, _ = scipy.signal.find_peaks(
            -self.channels[volume_idx]['raw'])
        self.peep = np.round(np.median(
            self.channels[pressure_idx]['raw'][v_ee_pks]))

    def find_occluded_breaths(
            self, pressure_idx=None, peep=None, overwrite=False, **kwargs):
        """
        Find end-expiratory occlusion manoeuvres in ventilator pressure
        timeseries data. See postprocessing.event_detection submodule.
        -----------------------------------------------------------------------
        :param pressure_idx: Channel index of the ventilator pressure data
        :type pressure_idx: int
        For other arguments, see postprocessing.event_detection submodule.

        :returns: None
        :rtype: None
        """
        pressure_idx = pressure_idx or self.p_vent_idx
        kwargs['p_vent'] = self.channels[pressure_idx]['raw']
        kwargs['fs'] = self.param['fs']

        kwargs['peep'] = peep or self.peep
        if kwargs['peep'] is None:
            raise ValueError('PEEP is not defined.')

        peak_idxs = evt.find_occluded_breaths(**kwargs)
        peak_idxs = peak_idxs + kwargs['start_idx']
        self.channels[pressure_idx].set_peaks(
            signal=self.channels[pressure_idx]['raw'], peak_idxs=peak_idxs,
            peak_set_name='Pocc', overwrite=overwrite)

    def find_ventilator_peaks(
            self, channel_io=None, overwrite=False, **kwargs):
        """
        Detect breath-related peaks in a specified ventilator signal
        Peaks are stored in the corresponding TimeSeries under the same signal.
        -----------------------------------------------------------------------
        :param channel_io: tuple of the input and output channels,
            the first element is the input channel for peak detection,
            the found peaks are stored in the channels listed in the
            second element. If none, the volume or pressure channel is used as
            input. In absence of these, the first channel is used.
        :type channel_io: tuple(str/int,str/int/list)
        :param overwrite: Whether to overwrite existing peak set
        :type overwrite: bool
        Other kwargs are passed to the peak detection function.

        :returns: None
        :rtype: None
        """
        if channel_io is None:
            channel_idx_i = self.v_vent_idx or self.p_vent_idx or 0
            channel_idx_list_o = []
            if self.p_vent_idx is not None:
                channel_idx_list_o.append(self.p_vent_idx)
            if self.v_vent_idx is not None:
                channel_idx_list_o.append(self.v_vent_idx)

        elif isinstance(channel_io, tuple) and len(channel_io) == 2:
            if isinstance(channel_io[0], int):
                channel_idx_i = channel_io[0]
            elif isinstance(channel_io[0], str):
                channel_idx_i = self.labels.index(channel_io[0])
            else:
                raise ValueError("channel_io[0] must be int or str")
            if isinstance(channel_io[1], int):
                channel_idx_list_o = [channel_io[1]]
            elif isinstance(channel_io[1], str):
                channel_idx_list_o = [self.labels.index(channel_io[1])]
            elif isinstance(channel_io[1], list):
                channel_idx_list_o = [
                    self.labels.index(ch) for ch in channel_io[1]]
            else:
                raise ValueError("channel_io[1] must be int, str or list")

        else:
            raise ValueError("channel_io must be a tuple of (input_channel,"
                             " output_channel)")

        signal_raw = self.channels[channel_idx_i]['raw']

        kwargs['start_idx'] = kwargs.setdefault('start_idx', 0)
        kwargs['end_idx'] = kwargs.setdefault('end_idx', len(signal_raw) - 1)
        kwargs['width_s'] = kwargs.setdefault('width_s', self.param['fs'] // 4)

        # Detect peaks
        peak_idxs = evt.detect_ventilator_breath(
            signal_raw,
            start_idx=kwargs['start_idx'],
            end_idx=kwargs['end_idx'],
            width_s=kwargs['width_s'],
            threshold=kwargs.get('threshold'),
            prominence=kwargs.get('prominence'),
            threshold_new=kwargs.get('threshold_new'),
            prominence_new=kwargs.get('prominence_new'))

        # Store peaks in the same signal
        for _channel_idx in channel_idx_list_o:
            self.channels[_channel_idx].set_peaks(
                signal=signal_raw,
                peak_idxs=peak_idxs,
                peak_set_name='ventilator_breaths',
                overwrite=True
            )
