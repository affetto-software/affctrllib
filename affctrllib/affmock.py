from pathlib import Path
from typing import Any

import numpy as np

from ._sockutil import Socket
from .affcomm import convert_array_to_string
from .affetto import Affetto
from .timer import Timer


class AffMock(Affetto):
    config_path: Path | None
    dof: int
    command_socket: Socket  # local
    sensory_socket: Socket  # remote
    sensor_rate: float

    def __init__(self, config_path: Path | str | None = None) -> None:
        self.command_socket = Socket()
        self.sensory_socket = Socket()
        super().__init__(config_path)

    def __repr__(self) -> str:
        return "%s.%s()" % (self.__class__.__module__, self.__class__.__qualname__)

    def load_config(self, config: dict[str, Any]):
        super().load_config(config)
        self.load_mock_config()

    def load_mock_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.mock_config = c["mock"]
        self.command_socket.addr = self.mock_config["local"]
        self.sensory_socket.addr = self.mock_config["remote"]
        self.sensor_rate = self.mock_config["sensor"]["rate"]

    def start(self, rate=None, quiet=False) -> None:
        self.sensory_socket.create()
        if rate is None:
            rate = self.sensor_rate
        timer = Timer(rate=rate)
        timer.start()
        while True:
            t = timer.elapsed_time()
            sarr = list(np.random.randint(0, 256, size=self.dof * 3))
            msg = convert_array_to_string(sarr)
            sz = self.sensory_socket.sendto(msg.encode())
            if not quiet:
                print(
                    f"t={t:.2f}: sent <{msg}> to {self.sensory_socket.addr} ({sz} bytes)"
                )
            timer.block()
