import socket as sock


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
            socket_type = None
        socket_type_append = ""
        if socket_type == sock.SOCK_DGRAM:
            socket_type_append = " (UDP)"
        elif socket_type == sock.SOCK_STREAM:
            socket_type_append = " (TCP)"
        elif isinstance(socket_type, sock.SocketKind):
            socket_type_append = f' ({str(socket_type).split(".")[1]})'
        try:
            return f"{self.host}:{str(self.port)}" + socket_type_append
        except AttributeError:
            return f"No address is provided" + socket_type_append

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

    def is_created(self) -> bool:
        if hasattr(self, "_socket"):
            return True
        else:
            return False

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
