"""Tests for `socket` module."""

from __future__ import annotations

import socket
from socket import AF_INET, SOCK_DGRAM, SOCK_STREAM

import pytest

from affctrllib.socket import IPv4Socket, split_data


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


@pytest.mark.parametrize(
    "data,expected",
    [("1 2 3", [1, 2, 3]), ("11 22 33 44", [11, 22, 33, 44]), (" 1  2    3  ", [1, 2, 3])],
)
def test_split_string_data(data: str, expected: list[int]) -> None:
    """Test the given string data composed of numbers is split into a list of ints."""
    arr = split_data(data)
    assert arr == expected


@pytest.mark.parametrize(
    "data,expected",
    [(b"1 2 3", [1, 2, 3]), (b"11 22 33 44", [11, 22, 33, 44]), (b" 1  2    3  ", [1, 2, 3])],
)
def test_split_binary_data(data: bytes, expected: list[int]) -> None:
    """Test the given binary data composed of numbers is split into a list of ints."""
    arr = split_data(data, converter=int)
    assert arr == expected


@pytest.mark.parametrize(
    "data,converter,expected",
    [("1 2 3", int, [1, 2, 3]), (b"1 2 3", float, [1.0, 2.0, 3.0]), (b"1 2 3", str, ["1", "2", "3"])],
)
def test_convert_types_when_split_data(data: bytes | str, converter, expected) -> None:
    """Test a specified converter is applied when splitting data."""
    arr = split_data(data, converter=converter)
    assert arr == expected
    assert type(arr[0]) is converter


@pytest.mark.parametrize(
    "data,sep,expected",
    [("1,2,3", ",", [1, 2, 3]), (b"1<>2<>3", "<>", [1, 2, 3]), (b"1 2 3", " ", [1, 2, 3])],
)
def test_separator_to_split_data(data: bytes | str, sep: str, expected: list[int]) -> None:
    """Test the given data composed of numbers is split into a list of ints."""
    arr = split_data(data, sep=sep)
    assert arr == expected


@pytest.mark.parametrize(
    "data,sep,expected",
    [(b" 1 2 3 ", " ", ["", "1", "2", "3", ""])],
)
def test_not_strip_when_split_data(data: bytes | str, sep: str, expected: list[str]) -> None:
    """Test that stripping the given data is not occur."""
    arr = split_data(data, converter=str, sep=sep, strip=False)
    assert arr == expected


@pytest.mark.parametrize(
    "data,converter,sep,strip,expected",
    [
        (b" 1 2 3 ", int, None, True, [1, 2, 3]),
        (b" 1 2 3 ", str, " ", True, ["1", "2", "3"]),
        (b"| 1 2 3 |", int, " ", "| ", [1, 2, 3]),
    ],
)
def test_complicated_cases_when_split_data(
    data: bytes | str, converter, sep: str, strip: bool | str, expected: list
) -> None:
    """Test complicated cases to split data."""
    arr = split_data(data, converter=converter, sep=sep, strip=strip)
    assert arr == expected
