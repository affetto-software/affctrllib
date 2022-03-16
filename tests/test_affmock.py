import os

from affctrllib.affmock import AffettoMock

config_dir_path = os.path.join(os.path.dirname(__file__), "config")


class TestAffComm:
    def test_init(self) -> None:
        mock = AffettoMock()
        assert mock.config_path == None
        assert mock.config_dict == {}

    def test_repr(self) -> None:
        mock = AffettoMock()
        assert repr(mock) == "affctrllib.affmock.AffettoMock()"

    def test_load_config(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(config_dir_path, "mock.toml"))
        assert isinstance(mock.config_dict, dict)

    def test_load_config_local(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(config_dir_path, "mock.toml"))
        assert mock.local_node.ip == "localhost"
        assert mock.local_node.port == 50010

    def test_load_config_remote(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(config_dir_path, "mock.toml"))
        assert mock.remote_node.ip == "localhost"
        assert mock.remote_node.port == 50000

    def test_load_config_sensor(self) -> None:
        mock = AffettoMock()
        mock.load_config(os.path.join(config_dir_path, "mock.toml"))
        assert mock.sensor_rate == 100
