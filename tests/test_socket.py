"""Tests for `socket` module."""

from __future__ import annotations

import socket
from socket import AF_INET, SOCK_DGRAM, SOCK_STREAM

import pytest

from affctrllib.socket import IPv4Socket


def default_address() -> tuple[str, int]:
    return ("localhost", 123456)


@pytest.fixture
def default_socket() -> IPv4Socket:
    return IPv4Socket(default_address())


def test_check_repr(default_socket: IPv4Socket) -> None:
    assert repr(default_socket) == "IPv4Socket(('localhost', 123456), SOCK_DGRAM, nonblock=False)"


def test_check_str(default_socket: IPv4Socket) -> None:
    assert str(default_socket) == "IPv4Socket<localhost:123456> (UDP)"
    s = IPv4Socket(default_address(), SOCK_STREAM)
    assert str(s) == "IPv4Socket<localhost:123456> (TCP)"
    s = IPv4Socket(default_address(), nonblock=True)
    assert str(s) == "IPv4Socket<localhost:123456> (UDP,nonblock)"


@pytest.mark.parametrize("host,port", [("localhost", 123456), ("192.168.5.11", 222222)])
def test_check_socket_attributes(host: str, port: int) -> None:
    """Check if a IPv4 socket object is created based on the given address.

    Parameters
    ----------
    host : str
        The host address.
    port : int
        The port number.
    """

    s = IPv4Socket((host, port))
    assert type(s.socket) == socket.socket
    assert s.address == (host, port)
    assert s.host == host
    assert s.port == port
    assert s.family == AF_INET
    assert s.type == SOCK_DGRAM
    assert s.proto == 0


@pytest.mark.parametrize("socket_type", [SOCK_DGRAM, SOCK_STREAM])
def test_specify_socket_type(socket_type: int) -> None:
    """Check if a created socket has the given socket type.

    Parameters
    ----------
    socket_type : int
        The socket type.
    """
    s = IPv4Socket(("localhost", 123456), socket_type=socket_type)
    assert s.type == socket_type


def test_enable_blocking_mode_by_default(default_socket: IPv4Socket) -> None:
    """Test that the blocking mode is enabled by default."""
    assert default_socket.is_nonblocking() == False


@pytest.mark.parametrize("nonblock", [False, True])
def test_check_nonblocking_mode(nonblock: bool) -> None:
    """Check if `is_nonblocking` returns False in blocking mode."""
    s = IPv4Socket(("localhost", 123456), nonblock=nonblock)
    assert s.is_nonblocking() == nonblock
