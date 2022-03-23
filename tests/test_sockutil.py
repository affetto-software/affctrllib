import pytest
from affctrllib._sockutil import SockAddr


class TestSockAddr:
    def test_init(self) -> None:
        sa = SockAddr()
        assert sa.family == "AF_INET"
        assert sa.host is None
        assert sa.port is None

    def test_init_args(self) -> None:
        sa = SockAddr(("localhost", 10101))
        assert sa.host == "localhost"
        assert sa.port == 10101

    def test_set_host(self) -> None:
        sa = SockAddr()
        sa.host = "192.168.1.1"
        assert sa.host == "192.168.1.1"

    def test_set_port(self) -> None:
        sa = SockAddr()
        sa.port = 2002
        assert sa.port == 2002

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
