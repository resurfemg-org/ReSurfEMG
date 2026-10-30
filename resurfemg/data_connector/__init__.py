import platform

# Import the submodules
from resurfemg.data_connector import (
    config,
    converter_functions,
    data_classes,
    file_discovery,
    peakset_class,
    synthetic_data,
    tmsisdk_lite,
)

if platform.system() == "Windows":
    from resurfemg.data_connector import adicht_reader
