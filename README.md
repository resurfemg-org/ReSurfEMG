<p align="center">
    <img style="width: 35%; height: 35%" src="https://raw.githubusercontent.com/resurfemg-org/ReSurfEMG/8ef19f7013553bd3dbdabf994781f0d612114336/Logo_rond_tekst.svg">
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6811553.svg)](https://doi.org/10.5281/zenodo.6811553)
[![PyPI](https://img.shields.io/pypi/v/resurfemg.svg)](https://pypi.python.org/pypi/resurfemg/)
[![Sanity](https://github.com/resurfemg-org/ReSurfEMG/actions/workflows/on-commit.yml/badge.svg)](https://github.com/resurfemg-org/ReSurfEMG/actions/workflows/on-commit.yml)
[![Sanity](https://github.com/resurfemg-org/ReSurfEMG/actions/workflows/on-tag.yml/badge.svg)](https://github.com/resurfemg-org/ReSurfEMG/actions/workflows/on-tag.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/10985/badge)](https://bestpractices.coreinfrastructure.org/projects/10985)
[![fair-software.eu](https://img.shields.io/badge/fair--software.eu-%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F%20%20%E2%97%8F-green)](https://fair-software.eu)
[![status](https://joss.theoj.org/papers/5f08d1f2bb717b7d05762296e37ded3d/status.svg)](https://joss.theoj.org/papers/5f08d1f2bb717b7d05762296e37ded3d)
[![OpenSSF Scorecard](https://api.scorecard.dev/projects/github.com/resurfemg-org/ReSurfEMG/badge)](https://scorecard.dev/viewer/?uri=github.com/resurfemg-org/ReSurfEMG)
[![codecov](https://codecov.io/gh/resurfemg-org/ReSurfEMG/graph/badge.svg?token=38G927R24C)](https://codecov.io/gh/resurfemg-org/ReSurfEMG)

**ReSurfEMG** is an open source collaborative python library for analysis
of respiratory electromyography (EMG). On the same site as 
the repository for this library we keep [related resources](https://github.com/ReSurfEMG?tab=repositories). 

ReSurfEMG includes a [main code library](https://github.com/resurfemg-org/ReSurfEMG) where the user can access the code to change various filter and analysis settings directly and/or in our [researcher interface notebooks](https://github.com/resurfemg-org/ReSurfEMG/tree/main/notebooks/researcher_interface).
In addition, ReSurfEMG has a [dashboard interface](https://github.com/resurfemg-org/ReSurfEMG-dashboard) which contains default settings for preprocessing and analysis which can be changed through a graphical (no code) interface. We have some functionality available through a [command line interface](#command-line-interface) as well.

The library was initially built for surface EMG, however many functions will also work for
invasively measured respiratory EMG.  This library supports the ongoing research at University of Twente on respiratory EMG.


## Getting Started

### Supported Platforms

ReSurfEMG is a pure Python package. Below is the list of
platforms that should work. Other platforms may work, but have had less extensive testing.
Please note that where python.org Python stated as supported, it means
that versions 3.10 - 3.13 are supported.

#### AMD64 (x86)

|                             | Linux     | Win       | OSX       |
|:---------------------------:|:---------:|:---------:|:---------:|
| ![p](etc/python-logo.png)   | 3.10 - 3.13 | 3.10 - 3.13* | 3.10 - 3.13 |
| ![a](etc/anaconda-logo.png) | Discontinued | Discontinued | Discontinued |

* For Python 3.13 on Windows there is a bug in the adi-reader package. For a succesfull install, please run the following command:
``` sh
pip install git+https://github.com/rspwarnaarUT/adinstruments_sdk_python.git@e15eb62862d3a8054eb3438c2a8f8d6480bf93b4
```
before running:
``` sh
pip install resurfemg
```

### Installation for all supported platforms

Installation in a virtual environment is the preffered method. They are covered in the ["Virtual environment setup"](### Virtual environment setup) section. 
If you wish to install with pip:

``` sh
pip install resurfemg
```

### Virtual environment setup (recommended)

How to get the notebooks running? Assuming the raw data set and
metadata is available.

### 0a. Create a virtual environment using Python

  #### On Linux/OSX:
``` sh
python3 -m venv .venv
```

  This might **require** the python3-venv.
    
  #### On Windows:
  ``` sh
  python -m venv .venv
  ```

### 0b. Activate the virtual environment and install ReSurfEMG

  #### On Linux/OSX:
  ``` sh
  source .venv/bin/activate
  ```
    
  #### On Windows:
  ``` sh
  .venv\Scripts\activate.bat
  ```
### 1. Install ReSurfEMG

  ``` sh
  pip install resurfemg[dev]
  ```

### 2. Open a notebook
   Start a local Jupyter Notebook server by running the `jupyter notebook` 
   command in your terminal. This opens a browser window where you can browse, 
   open and run the notebooks. (We use [Jupyter notebooks](https://jupyter.org/try-jupyter/retro/notebooks/?path=notebooks/Intro.ipynb))
   The ReSurfEMG notebooks are located in the notebooks folder. Navigate there 
   and open a notebook of interest. The [basic_emg_analysis](https://github.com/resurfemg-org/ReSurfEMG/blob/main/researcher_interface/basic_emg_analysis.ipynb) 
   notebook can be used to understand how to use the package.


## Advanced contributor's setup / "Developer's setup"

We distinguish between people who want to use this library in their own code 
and/or analysis, and people who also want to develop this library who we call 
developers, be it as members of our team or independent contributors. People 
who simply want to use our library need to install the packaged version from 
one of the package indexes to which we publish released versions (eg. PyPI). 
This section of the readme is for advanced developers who want to modify the 
library code (and possibly contribute their changes back or eventually publish 
thier own modified fork). NB: You can accomplish modifications of the code, 
submit pull-requests (PRs) and soforth without a 'developer's setup' but we 
feel this setup will make advanced contributions easier.

We have transitioned to a fully Python 3.9+ environment. 
(For older instructions with `venv` please see versions below 0.2.0, and
adapt them if using Windows and/or a different Python version than
Python.org Python. e.g. you may need to use `.venv\Scripts\activate.bat` in
place of `.venv/bin/activate`) 
The instructions below are for our newer versions above 1.0.0. This will create 
a distributable package from the source code, then install it in the currently
active environment.  This will also install development tools we use
s.a. `pytest` and `codestyle` and will also install tools we use for
working with the library, such as `jupyter`.

### On Linux/OSX:
``` sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
``` 
  
### On Windows:
``` sh
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .[dev]
```

These installs differ in two ways from the regular install: 1) The `.[dev]` (as compared to `resurfemg[dev]`) 
installs the library as it currently is including all your local changes, 
instead of pulling it from the PyPI repository.  2) The `-e` flag ensures an 
editable install, such that any changes to the library are automatically 
applied to your environment.

Now you should have everything necessary to start working on the
source code.

## Repository structure

### Program files

The core functions of ReSurfEMG are in the folder [resurfemg](https://github.com/resurfemg-org/ReSurfEMG/tree/main/resurfemg):

-   **cli:** Scripts for the command line interface
-   **data_connector:**  Converter functions for discovering, loading, simulating and handling data.
-   **helper_functions:** General functions to support the functions in this repository
-   **preprocessing:** Process the raw respiratory EMG signal
      - filtering: low-, high- and bandpass filters, notch filter, computer power loss
      - ecg-removal: gating and wavelet denoising
      - envelope: root-mean-square (RMS), average rectified (ARV)
-   **postprocessing:** Aspects of pre-processed the respiratory EMG data:
      - moving baselines
      - event detection: find pneumatic and EMG breaths, on- and offset detection
      - features: amplitude, area under the curve, slope, area under the baseline, respiratory rate
      - quality assessment: signal-to-noise ratio, end-expiratory occlussion manoeuvre quality, interpeak distance, area under the baseline, consecutive manoeuvres, bell-curve error, relative peak timing, relative area under the baseline, relative ETP


### Folders and Notebooks

Our [guide to notebooks](https://github.com/resurfemg-org/ReSurfEMG/blob/main/notebooks_guide.md) is under construction. To look around keep in mind the following distinction on folders:

dev:
- These notebooks are used in feature development and debugging by core members of the ReSurfEMG team. They can provide a basic example how to use some of the functionality.

open_work:
- This folder contains experimental work by core members of the ReSurfEMG
  team that is not deployed yet.

researcher_interface:
- These are a growing series of interactive notebooks that allow
  researchers to investigate questions about their own EMG data

## Data sets

The notebooks are configured to run on various datasets.  Contact
Dr. Eline Mos-Oppersma( 📫 e.mos-oppersma@utwente.nl) to discuss any
questions on data configuration for your datasets.

If you want to use a standardized dataset for any purpose we recommend
the data in the [ReSurfEMG/test_data](https://github.com/resurfemg-org/ReSurfEMG/blob/main/test_data), which is also used in testing the ReSurfEMG functions. 
Data there can be used with any respiratory EMG algorithms in any program. Thus that data can function as a benchmarking set to compare algorithms across different programs.

Alternatively, the data in the ReSurfEMG/synthetic_data repository:

[![DOI](https://zenodo.org/badge/635680008.svg)](https://zenodo.org/badge/latestdoi/635680008)


### Configuring (to work with your data)

In order to preprocess and/or to train  models the code needs to be
able to locate the raw data you want it to find.

There are several ways to specify the location of the following
directories:

-   **root_data:** Special directory. The rest of the directory layout can be derived from its location.
-   **preprocessed_data:** The directory that will be used by preprocessing
    code to output to.
-   **test_data:** The directory for running sanity tests.

You can store this information persistently in several locations.

1.  In the same directory where you run the script (or the notebook).
    e.g. `./config.json`.
2.  In home directory, e.g. `~/.resurfemg/config.json`.
3.  In global directory, e.g. `/etc/resurfemg/config.json`.

However, we highly recommend using the home directory.
This file can have this or similar contents:
```
{
    "root_data": "/mnt/data",
    "patient_data": "/mnt/patient_data",
    "simulated_data": "/mnt/simulated_data",
    "preprocessed_data": "/mnt/data/preprocessed",
    "output_data": "/mnt/data/output",
}
```
The file is read as follows: if the files specifies `root_data`
directory, then the missing entries are assumed to be relative to
the root.  You do not need to specify all entries.

### Test data

Test data is provided in the repository in the test_data folder.

## Generating documentation

Online documentation can be found at https://resurfemg-org.github.io/ReSurfEMG/
or on https://readthedocs.org/ by searching for ReSurfEMG. Up-to-date 
documentation can be generated in command-line as follows in terminal:

### On Linux/OSX:
``` sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[docs]
python setup.py apidoc
python setup.py build_sphinx
``` 
  
### On Windows:
``` sh
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .[docs]
python setup.py apidoc
python setup.py build_sphinx
```


## Automation

The project comes with several modifications to the typical default `setup.py`.

The project has a sub-project of a related dashboard. Dashboard is a GUI that
exposes some of the project's functionality. In the past, we kept a a legacy 
dashboard in the same repository with ReSurfEMG code but we have deleted it. 
The current version of the dashboard into it's own repository:
https://github.com/resurfemg-org/ReSurfEMG-dashboard


### New commands

* `isort resurfemg --check --diff` checks that the imports are properly 
  formatted and sorted.
* `python setup.py apidoc` generates RST outlines necessary to generate
  documentation.

### Testing

The project includes testing data. This test data is synthetic and generated by the ReSurfEMG `data_connector.synthetic_data` methods.

#### On Linux/OSX:
``` sh
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[tests]
pytest
``` 
  
#### On Windows:
``` sh
python -m venv .venv
.venv\Scripts\activate.bat
pip install -e .[tests]
pytest
```

Test coverage can be obtained via:

``` sh
coverage run -m pytest
coverage report
```


## Command-Line Interface

You will be able to preprocess data using command-line interface. You can also,
in some cases, create files in the correct format for our Dashboard in a per 
folder batch process.

You can make synthetic data. To explore this start with

    `python -m resurfemg simulate_emg --help`

    `python -m resurfemg simulate_ventilator --help`


✨Copyright 2022 Netherlands eScience Center and U. Twente
Licensed under the Apache License, version 2.0. See LICENSE for details.✨
