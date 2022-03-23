from __future__ import annotations


class SockAddr(object):
    """Represents a socket address for internet communication.

        This class stores a socket address structure for network
    communication. Currently, only IPv4 is supported.
    """

    family: str
    host: str | None
    port: int | None

    def __init__(
        self, addr: tuple[str, int] | str | None = None, port: int | None = None
    ) -> None:
        self.family = "AF_INET"
        self.host = None
        self.port = None
        if isinstance(addr, tuple):
            self.host = addr[0]
            self.port = addr[1]
        elif isinstance(addr, str):
            self.host = addr
            self.port = port

    @property
    def addr(self) -> tuple[str, int]:
        """Returns a socket address as a tuple of host and port."""
        if self.host is None:
            raise RuntimeError("SockAddr: no host is provided")
        if self.port is None:
            raise RuntimeError(f"SockAddr: no port is provided for '{self.host}'")
        return (self.host, self.port)
