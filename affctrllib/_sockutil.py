from __future__ import annotations

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
