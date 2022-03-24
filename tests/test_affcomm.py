import os

import pytest
from affctrllib._sockutil import SockAddr
from affctrllib.affcomm import AffComm

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


class TestAffComm:
    def test_init(self) -> None:
        acom = AffComm()
        assert acom.config_path is None
        assert isinstance(acom.remote_addr, SockAddr)
        assert isinstance(acom.local_addr, SockAddr)

    @pytest.mark.parametrize(
        "config_file,remote_host",
        [("default.toml", "192.168.1.1"), ("alternative.toml", "192.168.5.10")],
    )
    def test_init_load_config(self, config_file, remote_host) -> None:
        acom = AffComm(os.path.join(CONFIG_DIR_PATH, config_file))
        assert acom.remote_addr.host == remote_host

    def test_repr(self) -> None:
        acom = AffComm()
        assert repr(acom) == "affctrllib.affcomm.AffComm()"

    def test_load_config_default(self) -> None:
        acom = AffComm()
        acom.load_config(os.path.join(CONFIG_DIR_PATH, "default.toml"))

        # remote_addr
        assert acom.remote_addr.host == "192.168.1.1"
        assert acom.remote_addr.port == 50010

        # local_addr
        assert acom.local_addr.host == "localhost"
        assert acom.local_addr.port == 50000

    def test_load_config_alternative(self) -> None:
        acom = AffComm()
        acom.load_config(os.path.join(CONFIG_DIR_PATH, "alternative.toml"))

        # remote_addr
        assert acom.remote_addr.host == "192.168.5.10"
        assert acom.remote_addr.port == 60010

        # local_addr
        assert acom.local_addr.host == "192.168.5.123"
        assert acom.local_addr.port == 60000
