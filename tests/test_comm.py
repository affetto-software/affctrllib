"""Tests for `socket` module."""

from __future__ import annotations

import socket
from socket import AF_INET, SOCK_DGRAM, SOCK_STREAM

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from affctrllib.comm import (
    IPv4Socket,
    encode_data,
    join_data,
    split_data,
    unzip_items,
    unzip_items_as_array,
    zip_items,
    zip_items_as_array,
)


def default_address() -> tuple[str, int]:
    return ("localhost", 123456)


@pytest.fixture
def default_socket() -> IPv4Socket:
    return IPv4Socket(default_address())


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
    "arr,expected",
    [
        ([0, 1, 2], "0 1 2"),
        ([0, 1, 2, 3, 4], "0 1 2 3 4"),
        ([1.2, 3.2, 0.4, 8.7, 5.5], "1 3 0 9 6"),
        ([0.5, 1.5, 2.5, 3.5, 4.5], "0 2 2 4 4"),
    ],
)
def test_join_data(arr: list[int] | list[float], expected: str) -> None:
    """Test typical usages of `joint_data`."""
    s = join_data(arr)
    assert s == expected


@pytest.mark.parametrize(
    "arr,sep,expected",
    [
        ([0, 1, 2], ",", "0,1,2"),
        ([0, 1, 2], "|", "0|1|2"),
        ([0, 1, 2], "  ", "0  1  2"),
    ],
)
def test_join_data_with_specific_sep(arr: list[int] | list[float], sep: str, expected: str) -> None:
    """Test to join data with specific separator."""
    s = join_data(arr, sep=sep)
    assert s == expected


@pytest.mark.parametrize(
    "arr,specifier,expected",
    [
        ([0, 1, 2], "2d", " 0  1  2"),
        ([0, 1, 2], ".3f", "0.000 1.000 2.000"),
        ([1.333, 3.28, 5.5, 10.215], "05.2f", "01.33 03.28 05.50 10.21"),
    ],
)
def test_join_data_with_specifier(arr: list[int] | list[float], specifier: str, expected: str) -> None:
    """Test to join data in which each element is formated with given specifier."""
    s = join_data(arr, specifier=specifier)
    assert s == expected


@pytest.mark.parametrize(
    "arr,precision,expected",
    [
        ([0.54892, 1.289285, 2.889013], 1, "0.5 1.3 2.9"),
        ([0.54892, 1.289285, 2.889013], 3, "0.549 1.289 2.889"),
        ([0.54892, 1.289285, 2.889013], 5, "0.54892 1.28929 2.88901"),
    ],
)
def test_join_data_with_specific_precision(arr: list[int] | list[float], precision: int, expected: str) -> None:
    """Test to join data with specific precision."""
    s = join_data(arr, precision=precision)
    assert s == expected


@pytest.mark.parametrize(
    "arr,sep,specifier,expected",
    [
        (np.array([0, 1, 2]), " ", ".0f", "0 1 2"),
        (np.array([0, 1, 2]), ",", ".2f", "0.00,1.00,2.00"),
    ],
)
def test_join_data_ndarray(arr: list[int] | list[float], sep: str, specifier: str, expected: str) -> None:
    """Test to join when data is given as a numpy array."""
    s = join_data(arr, sep=sep, specifier=specifier)
    assert s == expected


@pytest.mark.parametrize(
    "arr,expected",
    [
        ([0, 1, 2], b"0 1 2"),
        ([0, 1, 2, 3, 4], b"0 1 2 3 4"),
        ([1.2, 3.2, 0.4, 8.7, 5.5], b"1 3 0 9 6"),
        ([0.5, 1.5, 2.5, 3.5, 4.5], b"0 2 2 4 4"),
    ],
)
def test_encode_data(arr: list[int] | list[float], expected: bytes) -> None:
    """Test if the joined data is encoded as a bytes object."""
    b = encode_data(arr)
    assert b == expected


@pytest.mark.parametrize(
    "items,n,expected",
    [
        ([1, 2, 3, 4, 5, 6], 2, [[1, 3, 5], [2, 4, 6]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 1, [[1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (np.array([2, 3, 4, 5, 6, 7]), 2, [[2, 4, 6], [3, 5, 7]]),
    ],
)
def test_unzip_items_as_array_typical_usage(items: list[int] | np.ndarray, n: int, expected: list[list[int]]) -> None:
    """Test typical usages of `unzip_items_as_array`."""
    arr = unzip_items_as_array(items, n)
    assert type(arr) == np.ndarray
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
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 1, [[1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (np.array([2, 3, 4, 5, 6, 7]), 2, [[2, 4, 6], [3, 5, 7]]),
    ],
)
def test_unzip_items_typical_usage(items: list[int] | np.ndarray, n: int, expected: list[list[int]]) -> None:
    """Test typical usages of `unzip_items`."""
    arr = unzip_items(items, n)
    assert arr == expected


def test_unzip_items_fails_if_num_of_items_is_not_factor_of_n() -> None:
    """Test if an exception is raised when the number of items is not a factor of n."""
    with pytest.raises(ValueError) as e:
        unzip_items([1, 2, 3, 4, 5], 2)
    assert "cannot unzip data with specified number of lines (n=2)" in str(e.value)


@pytest.mark.parametrize(
    "line1,line2,expected",
    [
        ([1, 2, 3], [4, 5, 6], [1, 4, 2, 5, 3, 6]),
        ([7, 8, 9], np.array([1, 2, 3]), [7, 1, 8, 2, 9, 3]),
        (np.array([4, 5, 6]), np.array([7, 8, 9]), [4, 7, 5, 8, 6, 9]),
    ],
)
def test_zip_two_lines_as_array(
    line1: list[int] | np.ndarray, line2: list[int] | np.ndarray, expected: list[int]
) -> None:
    """Test that given two lists are zipped correctly and return as a numpy array."""
    arr = zip_items_as_array(line1, line2)
    assert type(arr) == np.ndarray
    assert_array_equal(arr, expected)


@pytest.mark.parametrize(
    "line1,line2,line3,expected",
    [
        ([1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7, 2, 5, 8, 3, 6, 9]),
        ([4, 5, 6], [7, 8, 9], np.array([1, 2, 3]), [4, 7, 1, 5, 8, 2, 6, 9, 3]),
        (np.array([7, 8, 9]), [1, 2, 3], np.array([4, 5, 6]), [7, 1, 4, 8, 2, 5, 9, 3, 6]),
    ],
)
def test_zip_three_lines_as_array(
    line1: list[int] | np.ndarray, line2: list[int] | np.ndarray, line3: list[int] | np.ndarray, expected: list[int]
) -> None:
    """Test that given three lists are zipped correctly and return as a numpy array."""
    arr = zip_items_as_array(line1, line2, line3)
    assert type(arr) == np.ndarray
    assert_array_equal(arr, expected)


def test_zip_items_as_array_fails_if_sizes_of_lines_not_match() -> None:
    """Test if an exception is raised when sizes of lines do not match."""
    with pytest.raises(ValueError) as e:
        zip_items_as_array([1, 2, 3], [4, 5, 6, 7])
    assert "all input items must have the same size" in str(e.value)


@pytest.mark.parametrize(
    "line1,line2,expected",
    [
        ([1, 2, 3], [4, 5, 6], [1, 4, 2, 5, 3, 6]),
        ([7, 8, 9], np.array([1, 2, 3]), [7, 1, 8, 2, 9, 3]),
        (np.array([4, 5, 6]), np.array([7, 8, 9]), [4, 7, 5, 8, 6, 9]),
    ],
)
def test_zip_two_lines(line1: list[int] | np.ndarray, line2: list[int] | np.ndarray, expected: list[int]) -> None:
    """Test that given two lists are zipped correctly and return as a list."""
    arr = zip_items(line1, line2)
    assert arr == expected


@pytest.mark.parametrize(
    "line1,line2,line3,expected",
    [
        ([1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7, 2, 5, 8, 3, 6, 9]),
        ([4, 5, 6], [7, 8, 9], np.array([1, 2, 3]), [4, 7, 1, 5, 8, 2, 6, 9, 3]),
        (np.array([7, 8, 9]), [1, 2, 3], np.array([4, 5, 6]), [7, 1, 4, 8, 2, 5, 9, 3, 6]),
    ],
)
def test_zip_three_lines(
    line1: list[int] | np.ndarray, line2: list[int] | np.ndarray, line3: list[int] | np.ndarray, expected: list[int]
) -> None:
    """Test that given three lists are zipped correctly and return as a list."""
    arr = zip_items(line1, line2, line3)
    assert arr == expected


def test_zip_items_fails_if_sizes_of_lines_not_match() -> None:
    """Test if an exception is raised when sizes of lines do not match."""
    with pytest.raises(ValueError) as e:
        zip_items([1, 2, 3], [4, 5, 6, 7])
    assert "all input items must have the same size" in str(e.value)


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
