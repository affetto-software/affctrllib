from importlib import metadata

__version__ = metadata.version("affctrllib")
__all__ = ["__version__"]

from .affcomm import (
    AffComm,
    convert_array_to_bytes,
    convert_array_to_string,
    reshape_array_for_unzip,
    split_received_msg,
    unzip_array,
    zip_arrays,
    zip_arrays_as_ndarray,
)

__all__.extend(
    [
        "AffComm",
        "convert_array_to_bytes",
        "convert_array_to_string",
        "reshape_array_for_unzip",
        "split_received_msg",
        "unzip_array",
        "zip_arrays",
        "zip_arrays_as_ndarray",
    ]
)

from .affmock import AffettoMock

__all__.extend(["AffettoMock"])

from .logger import Logger

__all__.extend(["Logger"])

from .timer import Timer

__all__.extend(["Timer"])
