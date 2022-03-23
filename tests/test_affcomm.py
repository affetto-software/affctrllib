import os

from affctrllib._sockutil import SockAddr
from affctrllib.affcomm import AffComm

config_dir_path = os.path.join(os.path.dirname(__file__), "config")


class TestAffComm:
    def test_init(self) -> None:
        acom = AffComm()
        assert acom.config_path is None
        assert acom.config_dict == {}
        assert isinstance(acom.remote_addr, SockAddr)
        assert isinstance(acom.local_addr, SockAddr)

    def test_repr(self) -> None:
        acom = AffComm()
        assert repr(acom) == "affctrllib.affcomm.AffComm()"

    def test_load_config(self) -> None:
        acom = AffComm()
        acom.load_config(os.path.join(config_dir_path, "default.toml"))
        assert isinstance(acom.config_dict, dict)

    def test_load_config_remote(self) -> None:
        acom = AffComm()
        acom.load_config(os.path.join(config_dir_path, "default.toml"))
        assert acom.remote_addr.host == "192.168.1.1"
        assert acom.remote_addr.port == 50010

    def test_load_config_local(self) -> None:
        acom = AffComm()
        acom.load_config(os.path.join(config_dir_path, "default.toml"))
        assert acom.local_addr.host == "localhost"
        assert acom.local_addr.port == 50000
