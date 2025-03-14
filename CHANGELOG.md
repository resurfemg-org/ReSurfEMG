# Change Log

## Release 1.1.0
* Minor revision
	- Add .adi* file reader (LabChart, BIOPAC). NB Windows only
	- Revise basic_emg_pipeline.ipynb for (ventilator supported) breathing activity
	- Minor bug fixes and performance updates

## Release 1.0.2
* Patch
	- Move ReSurfEMG repo from https://github.com/ReSurfEMG/ReSurfEMG/ to https://github.com/resurfemg-org/ReSurfEMG/

## Release 1.0.1
* Patch
	- Fix: config file generation from template showed unexpected behaviour in overwriting existing config files and failing when not_pushed did not exist.
	- postprocessing.features methods time_to_peak and pseudo_slope methods smoothed by default. Changed to smoothing on smooth=True
	- Add subpackages and modules to init files for smoother imports

## Release 1.0.0
* Major revision
	- Refactor ReSurfEMG library
		- cli
		- data_connector
			- config
			- file_discovery
			- converter_functions
			- tmsisdk_lite
			- synthetic_data
			- data_classes
			- peakset_classes
		- preprocessing
			- filtering
			- ecg_removal
			- envelope
		- postprocesing
			- baseline
			- event_detection
			- features
			- quality_assessment
		- pipelines
			- processing
			- ipy_widgets
			- synthetic_data
		- helper_functions
			- math_operations
			- visualization
			- data_classes_quality_assessment
	- Added new functions:
		- data_connector.converter_functions
			- load_file
			- load_poly5
			- load_csv
			- load_mat
			- load_npy
		- data_connector.file_discovery
			- find_files
			- find_folders
		- data_connector.data_classes
			- TimeSeries
			- TimeSeriesGroup
			- EmgDataGroup
			- VentilatorDataGroup
		- data_connector.peakset_class
			- PeakSet
		- helper_functions.math_operations
			- bell_curve
		- helper_functions.visualization
			- show_psd_welch
			- show_periodogram
		- pipelines.ipy_widgets
			- file_select
		- pipelines.processing
			- quick_look
		- pipelines.synthetic_data
			- synthetic_ventilator_data_cli
		- pipelines
			- ipywidgets
		- postprocessing.event_detection
			- find_occluded_breaths
			- detect_ventilator_breath
			- detect_emg_breaths
			- find_linked_peaks
		- postprocessing.features
			- amplitude
			- respiratory_rate
		- postprocessing.quality_assessment
			- snr_pseudo
			- pocc_quality
			- interpeak_dist
			- percentage_under_baseline
			- detect_local_high_aub
			- detect_extreme_time_products
			- detect_non_consecutive_manoeuvres
			- evaluate_bell_curve_error
			- evaluate_event_timing
			- evaluate_respiratory_rates
		- preprocessing.ecg_removal
			- wavelet_denoising
		- preprocessing.envelope
			- full_rolling_arv
	- Renamed functions for clarity:
		- times_under_curve --> time_to_peak
		- simulate_ventilator_with_occlusions --> simulate_ventilator_data
		- simulate_emg_with_occlusions --> simulate_emg
		- find_peaks_in_ecg_signal --> detect_ecg_peaks
		- show_my_power_spectrum --> show_power_spectrum
	- Moved functions:
		- config.config --> data_connector.synthetic_data
			- simulate_ventilator_data
			- simulate_emg
		- preprocessing.envelope --> helper_functions.math_operations
		- visualization.visualization --> helper_functions.visualization
			- show_power_spectrum
		- pipelines.pipelines --> pipelines.processing
			- ecg_removal
	- Rudimentary functions are discontinued:
		- config.config
			- make_synth_emg
			- config.make_realistic_syn_emg
			- make_realistic_syn_emg_cli
		- data_connector.converter_functions
			- save_j_as_np
			- save_j_as_np_single
		- helper_functions.helper_functions
			- count_decision_array
			- relative_levenshtein
			- distance_matrix
			- preprocess
		- postprocessing.features
			- simple_area_under_curve
			- area_under_curve
			- find_peak_in_breath
			- variability_maker
		- postprocessing.envelope
			- smooth_for_baseline
			- smooth_for_baseline_with_overlay
			- vect_naive_rolling_rms
		- preprocessing.ecg_removal
			- compute_ica_two_comp
			- compute_ica_two_comp_multi
			- compute_ICA_two_comp_selective
			- compute_ICA_n_comp
			- pick_more_peaks_array
			- pick_highest_correlation_array_multi
			- compute_ICA_n_comp_selective_zeroing
			- pick_lowest_correlation_array
			- pick_highest_correlation_array
		- preprocessing.envelope
			- hi_envelope
			- smooth_for_baseline
			- smooth_for_baseline_with_overlay
			- vect_naive_rolling_rms
		- preprocessing.filtering
			- emg_bandpass_butter_sample
			- emg_highpass_butter_sample
			- bad_end_cutter
			- bad_end_cutter_for_samples
			- bad_end_cutter_better
	- Discontinue machine learning (ML) functionality
		- machine_learning.ml
			- save_ml_output
			- applu_model
		- pipelines.pipelines
			- working_pipe_multi
			- alternative_a_pipeline_multi
			- alternative_b_pipeline_multi
			- working_pipeline_pre_ml_multi
			- working_pipeline_exp
			- working_pipeline_pre_ml
	- Entropy functionality is moved to a legacy submodule (legacy.entropy), which is not included in the package:
		- --> legacy.entropical
		- --> legacy.entropy_scipy
		- --> legacy.rowwise_chebyshev
		- --> legacy.sampen
		- --> legacy.calc_closed_sampent
		- --> legacy.calc_open_sampent
		- --> legacy.entropy_maker

## Release 0.3.1

* Minor revision
	- Bug fixes in:
		- postprocessing.event_detection: On-/Offset detection using baseline crossing, and maximum slope extrapolation.


## Release 0.3.0

* Major revision

	- Conversion from setup.py build to pyproject.toml build
	- Discontinue Conda package releases
	- Include Python 3.10 and 3.11 support
	- Introduction of new functions:
		- postprocessing.baseline: Moving baseline, Slopesum baseline
		- postprocessing.event_detection: On-/Offset detection using baseline crossing, and maximum slope extrapolation.

## Release 0.2.1

* Fix release 0.2.0

	- Release 0.2.0 was a refactoring of the code base, but the newly
	created submodules were not included in the builds. This is now fixed.
	- Small adaptations to config.simulate_ventilator_with_occlusions to
	simulate more realistic Pocc manoeuvres.


## Release 0.2.0

* Eleventh version, second major revision

	- Refactoring of helper_functions in submodules organized per function
		-   config
		-   data_connector
		-   helper_functions
		-   machine_learning
		-   postprocessing
		-   preprocessing
	- Refactor tests accordingly

## Release 0.1.3

* Version 0.1.3

	- changes in setup.py to accomodate conda package version format
	- changes to getting started notebook and instructions
	- required release now (11 October 2023) for JOSS paper 

## Release 0.1.2


* Identical to tenth version, first major revision/version:

	- created due to technical problems with pypi
	
## Release 0.1.0


* Tenth version, first major revision/version:

	- Switch of all environments and functionality to run on Python 3.9
	- Installation possibility via mamba added (preffered)
	- Revision of some notebooks to allow processing of numpy array files directly
	- Fixed lack of tkinter in environment yaml
	
## Release 0.0.9

### Added

* Ninth version of this Python project, containing (added to this version):

	- Improved function for entropy in helper_functions module
	- Variability over array function (variability_maker) in helper_functions module
	- Additional command line functionality to convert certain csv files into files that can be processing in the ReSurfEMG-Dashboard.
	- Testing matrix to include Python 3.9 




## Release 0.0.8

### Added

* Eight version of this Python project, containing (added to this version):

	- Improved function at time to peak in a curve absolute and relative i.e. times_under_curve()
	- Function for psuedoslope of take-off i.e. helper_functions.pseudoslope()
	- Upgrade of setup.py to pin mne/mne-base, skikit-learn version, other setup file changes


## Release 0.0.7

### Added 

* Seventh version of this Python project, containing (added to this version):

	- Function for looking at time to peak in a curve absolute and relative i.e. times_under_curve()
	- Upgrade of CI to include a newer setup-conda action (v1.1.1)

## Release 0.0.6

### Added 

* Sixth version of this Python project, containing (added to this version):

	- Reading function Poly5Reader in new tmsisdk_lite module which reads Poly5-files and can produce an array in the format our library uses
	- Command line synthetic data creation, data pre-processing and machine learning (new module cli)
	- New synthetic data function in config module 
	- New independent component analysis (ICA) functions which allow processing of any number of leads
	- More functions for signal analysis including clinically relevant variations of area under curve and a function (distance_matrix) that produces various mathematical distances which can be used to compare EMG and other signals e.g. ventilator or EMG

## Release 0.0.5

### Added

* Fifth version of this Python project, containing (added to this version):

	- Converter functions added to converter_functions module to take other lab formats into an array in the format our library uses
	- Preprocessing pipelines for any number of leads, which any subset can be chosen added to multi_lead_type module
	- config module added so outside users can easily point towards thier own datasets, includes a new function to make synthetic EMG data, and hash file validation function (moved from converter_functions)
	- In repository (technically not part of version): legacy dashboard files removed, notebooks guide added
	
## Release 0.0.4

### Added

* Fourth version of this Python project, containing (added to this version):

	- Gating function for ECG removal,
	- Improve ICA for heart lead detection,
	- Power spectrum function in helper_functions,
	- Converter function for Biopac acquired data in converter_functions,
	- High envelope function hi_envelope in helper_functions,
	- Working pipeline improved (working_pipeline_pre_ml) in helper_functions,
	- Slices_jump_slider function produces continous sequential slices over an
    array of a certain legnth spaced out by a 'jump' in helper_functions.

## Release 0.0.0

### Added

* First version of this Python project to follow the Netherlands eScience Center software development guide, containing (added to this version):

	- Code style checking,
	- Editorconfig,
	- Code of Conduct,
	- Contributing guidelines,
	- Setup configuration,
	- files for installation/building release (pypi, condabuild)

## Unreleased

### Added
* First unreleased version of ReSurfEMG, containing:
	- Tests,
	- Documentation,
	- Change log,
	- License,
	- README,
* 

### Changed

*

