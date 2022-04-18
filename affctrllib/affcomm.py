import socket
from pathlib import Path
from typing import Any, Callable, TypeVar, overload

import numpy as np

from ._sockutil import SockAddr
from .affetto import Affetto

R = TypeVar("R")


def split_received_msg(
    data: bytes | str,
    function: Callable[[str], R] = float,
    sep: str | None = None,
    strip: bool = True,
) -> list[R]:
    """Returns a list of values converted from received bytes."""
    if isinstance(data, bytes):
        decoded_data = data.decode()
    elif isinstance(data, str):
        decoded_data = data
    else:
        raise TypeError(f"unsupported type: {type(data)}")
    if strip:
        decoded_data = decoded_data.strip(sep)
    return list(map(function, decoded_data.split(sep)))


def convert_array_to_string(
    array: list[float] | list[int] | np.ndarray,
    sep: str = " ",
    f_spec: str = ".0f",
    precision: int | None = None,
) -> str:
    """Returns a string of array joined with specific format."""
    if precision is None:
        formatted_array = [f"{x:{f_spec}}" for x in array]
    else:
        formatted_array = [f"{x:.{precision}f}" for x in array]
    return sep.join(formatted_array)


def convert_array_to_bytes(
    array: list[float] | list[int] | np.ndarray,
    sep: str = " ",
    f_spec: str = ".0f",
    precision: int | None = None,
) -> bytes:
    """Returns bytes encoded array joined with specific format."""
    return convert_array_to_string(array, sep, f_spec, precision).encode()


def unzip_array_as_ndarray(
    array: list[float] | list[int] | np.ndarray, ncol: int = 3
) -> np.ndarray:
    ret = np.array(array).reshape((int(len(array) / ncol), ncol))
    return ret.T


def unzip_array(array: list[float] | list[int] | np.ndarray, n: int = 3) -> list[Any]:
    arr = unzip_array_as_ndarray(array, ncol=n)
    return arr.tolist()


def zip_arrays_as_ndarray(
    *arrays: list[float] | list[int] | np.ndarray,
) -> np.ndarray:
    arr = np.stack(arrays, axis=1)
    return arr.flatten()


@overload
def zip_arrays(*arrays: list[float]) -> list[float]:
    ...


@overload
def zip_arrays(*arrays: list[int]) -> list[int]:
    ...


@overload
def zip_arrays(*arrays: np.ndarray) -> list[float]:
    ...


def zip_arrays(
    *arrays: list[float] | list[int] | np.ndarray,
) -> list[float] | list[int]:
    return list(zip_arrays_as_ndarray(*arrays))


class AffComm(Affetto):
    comm_config: dict[str, Any]
    remote_addr: SockAddr
    local_addr: SockAddr
    sensory_socket: socket.socket
    command_socket: socket.socket

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.remote_addr = SockAddr()
        self.local_addr = SockAddr()
        super().__init__(config_path)

    def __repr__(self) -> str:
        return "%s.%s()" % (self.__class__.__module__, self.__class__.__qualname__)

    def __str__(self) -> str:
        try:
            cpath = self._config_path
        except AttributeError:
            cpath = None
        return f"""\
AffComm configuration:
  Config file: {str(cpath)}
   Receive at: {str(self.local_addr)}
      Send to: {str(self.remote_addr)}
"""

    def load_config(self, config: dict[str, Any]) -> None:
        super().load_config(config)
        self.load_comm_config()

    def load_comm_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.comm_config = c["comm"]
        self.remote_addr.set(self.comm_config["remote"])
        self.local_addr.set(self.comm_config["local"])

    def create_sensory_socket(
        self, addr: tuple[str, int] | None = None
    ) -> socket.socket:
        self.sensory_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        if addr is not None:
            self.sensory_socket.bind(addr)
        else:
            self.sensory_socket.bind(self.local_addr.addr)
        return self.sensory_socket

    def create_command_socket(self) -> socket.socket:
        self.command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        return self.command_socket
