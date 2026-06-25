import platform

# Import the submodules
from resurfemg.data_connector import (
    config,  # noqa: F401
    converter_functions,  # noqa: F401
    data_classes,  # noqa: F401
    file_discovery,  # noqa: F401
    peakset_class,  # noqa: F401
    synthetic_data,  # noqa: F401
    tmsisdk_lite,  # noqa: F401
)

if platform.system() == "Windows":
    from resurfemg.data_connector import adicht_reader  # noqa: F401
