import pytest
from affctrllib._sockutil import SockAddr


class TestSockAddr:
    def test_init(self) -> None:
        sa = SockAddr()
        assert sa.family == "AF_INET"
        assert sa.host is None
        assert sa.port is None

    def test_init_with_tuple(self) -> None:
        sa = SockAddr(("localhost", 10101))
        assert sa.host == "localhost"
        assert sa.port == 10101

    def test_init_with_two_args(self) -> None:
        sa = SockAddr("172.16.11.1", 5555)
        assert sa.host == "172.16.11.1"
        assert sa.port == 5555

    def test_init_with_two_args_error(self) -> None:
        sa = SockAddr("172.16.11.2")
        assert sa.host == "172.16.11.2"
        assert sa.port is None
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert f"SockAddr: no port is provided for '172.16.11.2'" in str(excinfo.value)

    def test_init_with_dict(self) -> None:
        confdict = {"host": "10.0.0.1", "port": 11111}
        sa = SockAddr(confdict)
        assert sa.host == "10.0.0.1"
        assert sa.port == 11111

    def test_init_with_dict_error_no_host(self) -> None:
        confdict = {}
        sa = SockAddr(confdict)
        assert sa.host is None
        assert sa.port is None
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert f"SockAddr: no host is provided" in str(excinfo.value)

    def test_init_with_dict_error_no_port(self) -> None:
        confdict = {"host": "192.168.1.4"}
        sa = SockAddr(confdict)
        assert sa.host == "192.168.1.4"
        assert sa.port is None
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert f"SockAddr: no port is provided for '192.168.1.4'" in str(excinfo.value)

    @pytest.mark.parametrize(
        "host,port",
        [
            ("192.168.1.1", 1000),
            ("localhost", None),
            (None, None),
        ],
    )
    def test_repr(self, host, port) -> None:
        sa = SockAddr(host, port)
        assert repr(sa) == f"SockAddr({str(host)}, {str(port)})"

    @pytest.mark.parametrize(
        "host,port",
        [
            ("192.168.5.1", 1010),
            ("172.16.11.1", None),
            (None, None),
        ],
    )
    def test_str(self, host, port) -> None:
        sa = SockAddr(host, port)
        assert str(sa) == f"{str(host)}:{str(port)}"

    def test_set_host(self) -> None:
        sa = SockAddr()
        sa.host = "192.168.1.1"
        assert sa.host == "192.168.1.1"

    def test_set_port(self) -> None:
        sa = SockAddr()
        sa.port = 2002
        assert sa.port == 2002

    def test_set_with_tuple(self) -> None:
        sa = SockAddr()
        sa.set(("localhost", 10101))
        assert sa.host == "localhost"
        assert sa.port == 10101

    def test_set_with_two_args(self) -> None:
        sa = SockAddr()
        sa.set("172.16.11.1", 5555)
        assert sa.host == "172.16.11.1"
        assert sa.port == 5555

    def test_set_with_two_args_error(self) -> None:
        sa = SockAddr()
        sa.set("172.16.11.2")
        assert sa.host == "172.16.11.2"
        assert sa.port is None
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert f"SockAddr: no port is provided for '172.16.11.2'" in str(excinfo.value)

    def test_set_with_dict(self) -> None:
        confdict = {"host": "10.0.0.1", "port": 11111}
        sa = SockAddr()
        sa.set(confdict)
        assert sa.host == "10.0.0.1"
        assert sa.port == 11111

    def test_set_pwith_dict_error_no_host(self) -> None:
        confdict = {}
        sa = SockAddr()
        sa.set(confdict)
        assert sa.host is None
        assert sa.port is None
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert f"SockAddr: no host is provided" in str(excinfo.value)

    def test_set_with_dict_error_no_port(self) -> None:
        confdict = {"host": "192.168.1.4"}
        sa = SockAddr()
        sa.set(confdict)
        assert sa.host == "192.168.1.4"
        assert sa.port is None
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert f"SockAddr: no port is provided for '192.168.1.4'" in str(excinfo.value)

    @pytest.mark.parametrize(
        "host,port", [("192.168.11.1", 80808), ("192.168.11.2", 80810)]
    )
    def test_get_addr(self, host, port) -> None:
        sa = SockAddr((host, port))
        assert sa.addr == (host, port)

    def test_get_addr_with_no_host(self) -> None:
        sa = SockAddr()
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert "SockAddr: no host is provided" in str(excinfo.value)

    @pytest.mark.parametrize("host", ["192.168.11.3", "192.168.11.4"])
    def test_get_addr_with_no_port(self, host) -> None:
        sa = SockAddr()
        sa.host = host
        with pytest.raises(RuntimeError) as excinfo:
            _ = sa.addr
        assert f"SockAddr: no port is provided for '{host}'" in str(excinfo.value)
