import socket
from pathlib import Path
from typing import Any

import tomli

from ._sockutil import SockAddr


class AffComm(object):
    config_path: Path | None
    config_dict: dict[str, Any]
    remote_addr: SockAddr
    local_addr: SockAddr

    def __init__(self) -> None:
        self.config_path = None
        self.config_dict = {}

    def __repr__(self) -> str:
        return "%s.%s()" % (self.__class__.__module__, self.__class__.__qualname__)

    def load_config(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "rb") as f:
            self.config_dict = tomli.load(f)

        self.remote_addr = SockAddr()
        self.remote_addr.host = "192.168.1.1"
        self.remote_addr.port = 50010

        self.local_addr = SockAddr()
        self.local_addr.host = "localhost"
        self.local_addr.port = 50000

    def listen(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(self.local_addr.addr)
        bufsz = 1024
        while True:
            data, addr = sock.recvfrom(bufsz)
            print(f"Recv {data} from {addr}")
