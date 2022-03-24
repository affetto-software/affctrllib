import socket
import time
from pathlib import Path
from typing import Any

import tomli

from ._sockutil import SockAddr


class AffettoMock(object):
    config_path: Path | None
    remote_addr: SockAddr
    local_addr: SockAddr
    sensor_rate: float

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
        mock_config_dict = config_dict["affetto"]["mock"]
        self.remote_addr.set(mock_config_dict["remote"])
        self.local_addr.set(mock_config_dict["local"])
        self.sensor_rate = mock_config_dict["sensor"]["rate"]

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dt = 1.0 / self.sensor_rate
        while True:
            msg = str(time.time())
            sz = sock.sendto(msg.encode(), self.remote_addr.addr)
            print(f"Sent '{msg}' to {self.remote_addr.addr} ({sz} bytes)")
            time.sleep(dt)
