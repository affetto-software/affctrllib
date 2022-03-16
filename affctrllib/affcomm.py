from pathlib import Path
from typing import Any

import tomli


class Node(object):
    _ip: str
    _port: int

    @property
    def ip(self) -> str:
        return self._ip

    @ip.setter
    def ip(self, ip: str) -> None:
        self._ip = ip

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, port) -> None:
        self._port = port


class AffComm(object):
    config_path: Path | None
    config_dict: dict[str, Any]
    remote_node: Node

    def __init__(self) -> None:
        self.config_path = None
        self.config_dict = {}

    def __repr__(self) -> str:
        return "%s.%s()" % (self.__class__.__module__, self.__class__.__qualname__)

    def load_config(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "rb") as f:
            self.config_dict = tomli.load(f)

        self.remote_node = Node()
        self.remote_node.ip = "192.168.1.1"
        self.remote_node.port = 50010
