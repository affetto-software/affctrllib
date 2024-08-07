# ruff: noqa: PLR2004

from __future__ import annotations

from pathlib import Path

import pytest
from affctrllib._sockutil import Socket
from affctrllib.affmock import AffMock

CONFIG_DIR_PATH = Path(__file__).parent / "config"


class TestAffettoMock:
    def test_init(self) -> None:
        mock = AffMock()
        assert isinstance(mock.command_socket, Socket)
        assert isinstance(mock.sensory_socket, Socket)

    @pytest.mark.parametrize(
        ("config_file", "remote_host"),
        [("mock.toml", "localhost"), ("altmock.toml", "192.168.11.1")],
    )
    def test_init_load_config(self, config_file: str, remote_host: str) -> None:
        mock = AffMock(CONFIG_DIR_PATH / config_file)
        assert mock.sensory_socket.host == remote_host

    def test_repr(self) -> None:
        mock = AffMock()
        assert repr(mock) == "affctrllib.affmock.AffMock()"

    def test_load_config_default(self) -> None:
        mock = AffMock()
        mock.load_config_path(CONFIG_DIR_PATH / "mock.toml")

        # DOF
        assert mock.dof == 13

        # sensory_socket
        assert mock.sensory_socket.host == "localhost"
        assert mock.sensory_socket.port == 50000

        # command_socket
        assert mock.command_socket.host == "localhost"
        assert mock.command_socket.port == 50010

        # sensor
        assert mock.sensor_rate == 100

    def test_load_config_alternative(self) -> None:
        mock = AffMock()
        mock.load_config_path(CONFIG_DIR_PATH / "altmock.toml")

        # DOF
        assert mock.dof == 14

        # sensory_socket
        assert mock.sensory_socket.host == "192.168.11.1"
        assert mock.sensory_socket.port == 70000

        # command_socket
        assert mock.command_socket.host == "192.168.11.2"
        assert mock.command_socket.port == 70010

        # sensor
        assert mock.sensor_rate == 60

    def test_load_config_sensor(self) -> None:
        mock = AffMock()
        mock.load_config_path(CONFIG_DIR_PATH / "mock.toml")
        assert mock.sensor_rate == 100


# Local Variables:
# jinx-local-words: "Aff affmock altmock localhost noqa"
# End:
