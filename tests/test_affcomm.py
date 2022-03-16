import os

from affctrllib.affcomm import AffComm

config_dir_path = os.path.join(os.path.dirname(__file__), "config")


class TestAffComm:
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
        assert acom.remote_node.ip == "192.168.1.1"
        assert acom.remote_node.port == 50010
