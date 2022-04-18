import socket

import pytest
from affctrllib._sockutil import Socket


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
            ("192.168.2.22", 2020),
            ("192.168.3.33", 3030),
        ],
    )
    def test_str(self, host: str, port: int) -> None:
        s = Socket((host, port))
        assert str(s) == f"{host}:{str(port)}"

    @pytest.mark.parametrize(
        "host,port",
        [
            ("localhost", 1010),
            ("192.168.2.22", 2020),
            ("192.168.3.33", 3030),
        ],
    )
    def test_str_after_create(self, host: str, port: int) -> None:
        s = Socket((host, port))
        s.create()
        assert str(s) == f"{host}:{str(port)} (UDP)"

    def test_str_no_address(self) -> None:
        s = Socket()
        assert str(s) == f"No address is provided"

    def test_str_no_address_after_create(self) -> None:
        s = Socket()
        s.create()
        assert str(s) == f"No address is provided (UDP)"

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
