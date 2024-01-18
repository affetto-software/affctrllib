"""Tests for `socket` module."""

from __future__ import annotations

import socket
from socket import AF_INET, SOCK_DGRAM, SOCK_STREAM

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from affctrllib.comm import IPv4Socket, split_data, unzip_items, unzip_items_as_array


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


@pytest.mark.parametrize(
    "items,n,expected",
    [
        ([1, 2, 3, 4, 5, 6], 2, [[1, 3, 5], [2, 4, 6]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 1, [[1, 2, 3, 4, 5, 6, 7, 8, 9]]),
    ],
)
def test_unzip_items_as_array_typical_usage(items: list[int], n: int, expected: list[list[int]]) -> None:
    """Test typical usages of `unzip_items_as_array`."""
    arr = unzip_items_as_array(items, n)
    assert_array_equal(arr, expected)


def test_unzip_items_as_array_fails_if_num_of_items_is_not_factor_of_n() -> None:
    """Test if an exception is raised when the number of items is not a factor of n."""
    with pytest.raises(ValueError) as e:
        unzip_items_as_array([1, 2, 3, 4, 5], 2)
    assert "cannot unzip data with specified number of lines (n=2)" in str(e)


@pytest.mark.parametrize(
    "items,n,expected",
    [
        ([1, 2, 3, 4, 5, 6], 2, [[1, 3, 5], [2, 4, 6]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
    ],
)
def test_unzip_items_typical_usage(items: list[int], n: int, expected: list[list[int]]) -> None:
    """Test typical usages of `unzip_items`."""
    arr = unzip_items(items, n)
    assert arr == expected


def test_unzip_items_fails_if_num_of_items_is_not_factor_of_n() -> None:
    """Test if an exception is raised when the number of items is not a factor of n."""
    with pytest.raises(ValueError) as e:
        unzip_items([1, 2, 3, 4, 5], 2)
    assert "cannot unzip data with specified number of lines (n=2)" in str(e)
