import socket
from pathlib import Path
from typing import Any, Callable, TypeVar, overload

import numpy as np
import numpy.typing as npt
import tomli

from ._sockutil import SockAddr

R = TypeVar("R")


def split_received_msg(
    data: bytes | str,
    function: Callable[[str], R] = int,
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
    array: list[float] | list[int],
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
    array: list[float] | list[int],
    sep: str = " ",
    f_spec: str = ".0f",
    precision: int | None = None,
) -> bytes:
    """Returns bytes encoded array joined with specific format."""
    return convert_array_to_string(array, sep, f_spec, precision).encode()


def reshape_array_for_unzip(
    array: list[float] | list[int], ncol: int = 3
) -> npt.ArrayLike:
    ret = np.array(array).reshape((int(len(array) / ncol), ncol))
    return ret.T


def unzip_array(array: list[float] | list[int], n: int = 3) -> list[Any]:
    reshaped = reshape_array_for_unzip(array, ncol=n)
    return reshaped.tolist()  # type: ignore


def zip_arrays_as_ndarray(*arrays: list[float] | list[int]) -> npt.ArrayLike:
    stacked = np.stack(arrays, axis=1)
    return stacked.flatten()


@overload
def zip_arrays(*arrays: list[float]) -> list[float]:
    ...


@overload
def zip_arrays(*arrays: list[int]) -> list[int]:
    ...


def zip_arrays(*arrays: list[float] | list[int]) -> list[float] | list[int]:
    return list(zip_arrays_as_ndarray(*arrays))  # type: ignore


class AffComm(object):
    config_path: Path | None
    remote_addr: SockAddr
    local_addr: SockAddr
    sensory_socket: socket.socket
    command_socket: socket.socket

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.config_path = None
        if config_path is not None:
            self.config_path = Path(config_path)
        self.remote_addr = SockAddr()
        self.local_addr = SockAddr()

        if self.config_path:
            self.load_config(self.config_path)

    def __repr__(self) -> str:
        return "%s.%s()" % (self.__class__.__module__, self.__class__.__qualname__)

    def __str__(self) -> str:
        return f"""\
AffComm configuration:
  Config file: {str(self.config_path)}
   Receive at: {str(self.local_addr)}
      Send to: {str(self.remote_addr)}
"""

    def load_config(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "rb") as f:
            config_dict = tomli.load(f)
        comm_config_dict = config_dict["affetto"]["comm"]
        self.remote_addr.set(comm_config_dict["remote"])
        self.local_addr.set(comm_config_dict["local"])

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

    def listen(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(self.local_addr.addr)
        bufsz = 1024
        while True:
            data, addr = sock.recvfrom(bufsz)
            print(f"Recv {data} from {addr}")
