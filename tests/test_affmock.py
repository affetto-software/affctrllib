import os

import pytest
from affctrllib._sockutil import SockAddr
from affctrllib.affmock import AffettoMock

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


class TestAffettoMock:
    def test_init(self) -> None:
        mock = AffettoMock()
        assert mock.config_path == None
        assert mock.config_dict == {}
        assert isinstance(mock.remote_addr, SockAddr)
        assert isinstance(mock.local_addr, SockAddr)

    def test_repr(self) -> None:
        mock = AffettoMock()
        assert repr(mock) == "affctrllib.affmock.AffettoMock()"

    def test_load_config(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(CONFIG_DIR_PATH, "mock.toml"))
        assert isinstance(mock.config_dict, dict)

    def test_load_config_default(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(CONFIG_DIR_PATH, "mock.toml"))

        # remote_addr
        assert mock.remote_addr.host == "localhost"
        assert mock.remote_addr.port == 50000

        # local_addr
        assert mock.local_addr.host == "localhost"
        assert mock.local_addr.port == 50010

    def test_load_config_alternative(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(CONFIG_DIR_PATH, "altmock.toml"))

        # remote_addr
        assert mock.remote_addr.host == "192.168.11.1"
        assert mock.remote_addr.port == 70000

        # local_addr
        assert mock.local_addr.host == "192.168.11.2"
        assert mock.local_addr.port == 70010

    def test_load_config_sensor(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(CONFIG_DIR_PATH, "mock.toml"))
        assert mock.sensor_rate == 100
