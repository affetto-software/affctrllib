import socket as sock
from typing import Any


class SockAddr(object):
    """Represents a socket address for internet communication.

        This class stores a socket address structure for network
    communication. Currently, only IPv4 is supported.
    """

    family: str
    host: str | None
    port: int | None

    def __init__(
        self,
        addr: tuple[str, int] | dict[str, Any] | str | None = None,
        port: int | None = None,
    ) -> None:
        self.family = "AF_INET"
        self.host = None
        self.port = None
        if addr is not None:
            self.set(addr, port)

    def __repr__(self) -> str:
        return f"SockAddr({str(self.host)}, {str(self.port)})"

    def __str__(self) -> str:
        return f"{str(self.host)}:{str(self.port)}"

    def set(
        self, addr: tuple[str, int] | dict[str, Any] | str, port: int | None = None
    ) -> None:
        if isinstance(addr, tuple):
            self.host = addr[0]
            self.port = addr[1]
        elif isinstance(addr, dict):
            self.host = addr.get("host", None)
            self.port = addr.get("port", None)
        elif isinstance(addr, str):
            self.host = addr
            self.port = port
        else:
            raise TypeError(f"unsupported type: {type(addr)}")

    @property
    def addr(self) -> tuple[str, int]:
        """Returns a socket address as a tuple of host and port."""
        if self.host is None:
            raise RuntimeError("SockAddr: no host is provided")
        if self.port is None:
            raise RuntimeError(f"SockAddr: no port is provided for '{self.host}'")
        return (self.host, self.port)


class Socket(object):

    _family: str
    _host: str
    _port: int
    _socket: sock.socket

    def __init__(
        self,
        addr: tuple[str, int] | dict[str, str | int] | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> None:
        self._family = "AF_INET"
        if addr is not None:
            self.addr = addr
        if host is not None:
            self.host = host
        if port is not None:
            self.port = port

    def __repr__(self) -> str:
        try:
            return f"Socket(({self.host}, {str(self.port)}))"
        except AttributeError:
            return f"Socket()"

    def __str__(self) -> str:
        try:
            socket_type = self._socket.type
        except AttributeError:
            socket_type = sock.SOCK_DGRAM
        socket_type_name = ""
        if socket_type == sock.SOCK_DGRAM:
            socket_type_name = "UDP"
        elif socket_type == sock.SOCK_STREAM:
            socket_type_name = "TCP"
        else:
            socket_type_name = str(socket_type).split(".")[1]
        try:
            return f"{socket_type_name}: {self.host}:{str(self.port)}"
        except AttributeError:
            return f"{socket_type_name}: No address is provided"

    @property
    def family(self) -> str:
        return self._family

    @property
    def host(self) -> str:
        return self._host

    @host.setter
    def host(self, host: str) -> None:
        self._host = host

    @property
    def port(self) -> int:
        return self._port

    @port.setter
    def port(self, port: int) -> None:
        self._port = port

    @property
    def addr(self) -> tuple[str, int]:
        return (self.host, self.port)

    @addr.setter
    def addr(self, addr: tuple[str, int] | dict[str, int | str]) -> None:
        if isinstance(addr, tuple):
            self.host = addr[0]
            self.port = addr[1]
        elif isinstance(addr, dict):
            self.host = str(addr["host"])
            self.port = int(addr["port"])
        else:
            raise TypeError(f"unsupported type for addr: {type(addr)}")

    @property
    def socket(self) -> sock.socket:
        try:
            return self._socket
        except AttributeError:
            raise RuntimeError("No socket is created yet")

    def create(self, socket_type=sock.SOCK_DGRAM) -> sock.socket:
        self._socket = sock.socket(getattr(sock, self.family), socket_type)
        return self._socket

    def bind(self, addr: tuple[str, int] | None = None) -> None:
        if addr is None:
            try:
                addr = self.addr
            except AttributeError:
                raise RuntimeError("No address is provided to bind socket")
        self.socket.bind(addr)

    def recvfrom(self, bufsize=1024) -> bytes:
        recv_bytes, _ = self.socket.recvfrom(bufsize)
        return recv_bytes

    def sendto(self, send_bytes: bytes, addr: tuple[str, int] | None = None) -> int:
        if addr is None:
            try:
                addr = self.addr
            except AttributeError:
                raise RuntimeError("No address is provided to send to")
        return self.socket.sendto(send_bytes, addr)

    def close(self) -> None:
        self.socket.close()
