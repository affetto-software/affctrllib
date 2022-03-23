import socket
import time
from pathlib import Path
from typing import Any

import tomli

from ._sockutil import SockAddr


class AffettoMock(object):
    config_path: Path | None
    config_dict: dict[str, Any]
    remote_addr: SockAddr
    local_addr: SockAddr
    sensor_rate: float

    def __init__(self) -> None:
        self.config_path = None
        self.config_dict = {}
        self.remote_addr = SockAddr()
        self.local_addr = SockAddr()

    def __repr__(self) -> str:
        return "affctrllib.affmock.AffettoMock()"

    def load_config(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path)
        with open(self.config_path, "rb") as f:
            self.config_dict = tomli.load(f)
        self.remote_addr.set(self.config_dict["affetto"]["mock"]["remote"])
        self.local_addr.set(self.config_dict["affetto"]["mock"]["local"])

        self.sensor_rate = 100

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        dt = 1.0 / self.sensor_rate
        while True:
            msg = str(time.time())
            sz = sock.sendto(msg.encode(), self.remote_addr.addr)
            print(f"Sent '{msg}' to {self.remote_addr.addr} ({sz} bytes)")
            time.sleep(dt)
