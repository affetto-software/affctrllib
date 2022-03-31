import socket
from pathlib import Path
from typing import Callable

import tomli

from ._sockutil import SockAddr


def split_received_msg(
    data: bytes | str,
    function: Callable = float,
    sep: str | None = None,
    strip: bool = True,
) -> list[float]:
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
