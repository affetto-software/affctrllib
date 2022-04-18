import socket

import pytest
from affctrllib._sockutil import SockAddr, Socket


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


class TestSocket:
    def test_init(self) -> None:
        s = Socket()
        assert s.family == "AF_INET"

    def test_init_with_addr(self) -> None:
        addr = ("192.168.10.10", 10)
        s = Socket(addr)
        assert s.family == "AF_INET"
        assert s.host == addr[0]
        assert s.port == addr[1]
        assert s.addr == addr

    @pytest.mark.parametrize("host", ["localhost", "192.168.11.1", "192.168.11.3"])
    def test_init_host(self, host: str) -> None:
        s = Socket(host=host)
        assert s.host == host

    @pytest.mark.parametrize("port", [0, 80, 1111])
    def test_init_port(self, port: int) -> None:
        s = Socket(port=port)
        assert s.port == port

    @pytest.mark.parametrize(
        "addr",
        [
            ("localhost", 80),
            ("192.168.1.1", 1000),
            ("192.168.5.10", 3000),
        ],
    )
    def test_init_addr(self, addr: tuple[str, int]) -> None:
        s = Socket(addr=addr)
        assert s.host == addr[0]
        assert s.port == addr[1]
        assert s.addr == addr

    @pytest.mark.parametrize(
        "host,port",
        [
            ("localhost", 1000),
            ("192.168.1.1", 2000),
            ("192.168.12.34", 3000),
        ],
    )
    def test_repr(self, host: str, port: int) -> None:
        s = Socket((host, port))
        assert repr(s) == f"Socket(({host}, {str(port)}))"

    def test_repr_no_address(self) -> None:
        s = Socket()
        assert repr(s) == f"Socket()"

    @pytest.mark.parametrize(
        "host,port",
        [
            ("localhost", 1010),
            # ("192.168.2.22", 2020),
            # ("192.168.3.33", 3030),
        ],
    )
    def test_str(self, host: str, port: int) -> None:
        s = Socket((host, port))
        assert str(s) == f"UDP: {host}:{str(port)}"

    def test_str_no_address(self) -> None:
        s = Socket()
        assert str(s) == f"UDP: No address is provided"

    @pytest.mark.parametrize("host", ["localhost", "192.168.11.4", "192.168.11.5"])
    def test_host_setter(self, host: str) -> None:
        s = Socket()
        s.host = host
        assert s.host == host

    @pytest.mark.parametrize("port", [10, 20, 30])
    def test_port_setter(self, port: int) -> None:
        s = Socket()
        s.port = port
        assert s.port == port

    @pytest.mark.parametrize(
        "addr",
        [
            ("localhost", 8080),
            ("192.168.5.1", 1010),
            ("192.168.5.2", 3030),
        ],
    )
    def test_addr_setter(self, addr: tuple[str, int]) -> None:
        s = Socket()
        s.addr = addr
        assert s.host == addr[0]
        assert s.port == addr[1]
        assert s.addr == addr

    @pytest.mark.parametrize(
        "addr_dict",
        [
            {"host": "localhost", "port": 800},
            {"host": "192.168.1.1", "port": 700},
            {"host": "192.168.1.3", "port": 500},
        ],
    )
    def test_set_addr_dict(self, addr_dict: dict[str, int | str]) -> None:
        s = Socket()
        s.addr = addr_dict
        assert s.host == addr_dict["host"]
        assert s.port == addr_dict["port"]

    def test_create(self) -> None:
        s = Socket(("localhost", 10000))
        s.create()
        assert s.socket.family == socket.AF_INET
        assert s.socket.type == socket.SOCK_DGRAM

    def test_socket_error(self) -> None:
        s = Socket()
        with pytest.raises(RuntimeError) as excinfo:
            _ = s.socket
        assert str(excinfo.value) == "No socket is created yet"

    def test_bind_error_no_address(self) -> None:
        s = Socket()
        s.create()
        with pytest.raises(RuntimeError) as excinfo:
            s.bind()
        assert str(excinfo.value) == "No address is provided to bind socket"

    def test_bind_error_not_created(self) -> None:
        s = Socket(("localhost", 10000))
        with pytest.raises(RuntimeError) as excinfo:
            s.bind()
        assert str(excinfo.value) == "No socket is created yet"

    def test_recvfrom_error_not_created(self) -> None:
        s = Socket()
        with pytest.raises(RuntimeError) as excinfo:
            _ = s.recvfrom()
        assert str(excinfo.value) == "No socket is created yet"

    def test_sendto_error_no_address(self) -> None:
        s = Socket()
        s.create()
        with pytest.raises(RuntimeError) as excinfo:
            s.sendto(b"")
        assert str(excinfo.value) == "No address is provided to send to"

    def test_sendto_error_not_created(self) -> None:
        s = Socket(("192.168.1.1", 10000))
        with pytest.raises(RuntimeError) as excinfo:
            s.sendto(b"")
        assert str(excinfo.value) == "No socket is created yet"

    def test_close_error_not_created(self) -> None:
        s = Socket()
        with pytest.raises(RuntimeError) as excinfo:
            s.close()
        assert str(excinfo.value) == "No socket is created yet"
