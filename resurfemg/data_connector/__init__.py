# -*- coding: utf-8 -*-
import platform
# Import the submodules
from . import config
from . import converter_functions
from . import data_classes
from . import file_discovery
from . import peakset_class
from . import synthetic_data
from . import tmsisdk_lite
if platform.system() == 'Windows':
    from . import adicht_reader
