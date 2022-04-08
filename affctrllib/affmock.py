import socket
from pathlib import Path

import numpy as np
import tomli

from ._sockutil import SockAddr
from .affcomm import convert_array_to_string
from .timer import Timer


class AffettoMock(object):
    config_path: Path | None
    dof: int
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
        self.dof = mock_config_dict["dof"]
        self.remote_addr.set(mock_config_dict["remote"])
        self.local_addr.set(mock_config_dict["local"])
        self.sensor_rate = mock_config_dict["sensor"]["rate"]

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        timer = Timer(rate=self.sensor_rate)
        timer.start()
        while True:
            t = timer.elapsed_time()
            sarr = list(np.random.randint(0, 256, size=self.dof * 3))
            msg = convert_array_to_string(sarr)
            sz = sock.sendto(msg.encode(), self.remote_addr.addr)
            print(f"t={t:.2f}: sent <{msg}> to {self.remote_addr.addr} ({sz} bytes)")
            timer.block()
