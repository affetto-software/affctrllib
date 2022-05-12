from importlib import metadata

__version__ = metadata.version("affctrllib")
__all__ = ["__version__"]

from .affcomm import (
    AffComm,
    convert_array_to_bytes,
    convert_array_to_string,
    split_received_msg,
    unzip_array,
    unzip_array_as_ndarray,
    zip_arrays,
    zip_arrays_as_ndarray,
)

__all__.extend(
    [
        "AffComm",
        "convert_array_to_bytes",
        "convert_array_to_string",
        "unzip_array_as_ndarray",
        "split_received_msg",
        "unzip_array",
        "zip_arrays",
        "zip_arrays_as_ndarray",
    ]
)

from .affctrl import AffCtrl, AffCtrlThread

__all__.extend(["AffCtrl", "AffCtrlThread"])

from .affmock import AffMock

__all__.extend(["AffMock"])

from .affstate import AffState, AffStateThread

__all__.extend(["AffState", "AffStateThread"])

from .filter import Filter

__all__.extend(["Filter"])

from .logger import Logger

__all__.extend(["Logger"])

from .ptp import PTP

__all__.extend(["PTP"])

from .timer import Timer

__all__.extend(["Timer"])
