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


class AffettoMock(object):
    config_path: Path | None
    config_dict: dict[str, Any]
    local_node: Node
    remote_node: Node
    sensor_rate: float

    def __init__(self) -> None:
        self.config_path = None
        self.config_dict = {}

    def __repr__(self) -> str:
        return "affctrllib.affmock.AffettoMock()"

    def load_config(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "rb") as f:
            self.config_dict = tomli.load(f)

        self.local_node = Node()
        self.local_node.ip = "localhost"
        self.local_node.port = 50010
        self.remote_node = Node()
        self.remote_node.ip = "localhost"
        self.remote_node.port = 50000

        self.sensor_rate = 100
