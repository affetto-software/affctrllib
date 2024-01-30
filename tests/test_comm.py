"""Tests for `socket` module."""

from __future__ import annotations

import socket
import threading
import time
from itertools import count
from socket import AF_INET, SOCK_DGRAM, SOCK_STREAM
from typing import Any, NoReturn, Sequence

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from affctrllib.comm import (
    IPv4Socket,
    encode_data,
    join_data,
    split_data,
    unzip_sequence,
    unzip_sequence_as_array,
    zip_sequences,
    zip_sequences_as_array,
)


def default_address() -> tuple[str, int]:
    return ("localhost", 50000)


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
    "sequence,n,expected",
    [
        ([1, 2, 3, 4, 5, 6], 2, [[1, 3, 5], [2, 4, 6]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 1, [[1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (np.array([2, 3, 4, 5, 6, 7]), 2, [[2, 4, 6], [3, 5, 7]]),
    ],
)
def test_unzip_sequence_as_array_typical_usage(
    sequence: list[int] | np.ndarray, n: int, expected: list[list[int]]
) -> None:
    """Test typical usages of `unzip_sequence_as_array`."""
    arr = unzip_sequence_as_array(sequence, n)
    assert type(arr) == np.ndarray
    assert_array_equal(arr, expected)


def test_unzip_sequence_as_array_fails_if_size_of_sequence_is_not_factor_of_n() -> None:
    """Test if an exception is raised when the size of sequence is not a factor of n."""
    with pytest.raises(ValueError) as e:
        unzip_sequence_as_array([1, 2, 3, 4, 5], 2)
    assert "cannot unzip data with specified number of lines (n=2)" in str(e)


@pytest.mark.parametrize(
    "sequence,n,expected",
    [
        ([1, 2, 3, 4, 5, 6], 2, [[1, 3, 5], [2, 4, 6]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 3, [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
        ([1, 2, 3, 4, 5, 6, 7, 8, 9], 1, [[1, 2, 3, 4, 5, 6, 7, 8, 9]]),
        (np.array([2, 3, 4, 5, 6, 7]), 2, [[2, 4, 6], [3, 5, 7]]),
    ],
)
def test_unzip_sequence_typical_usage(sequence: list[int] | np.ndarray, n: int, expected: list[list[int]]) -> None:
    """Test typical usages of `unzip_sequence`."""
    arr = unzip_sequence(sequence, n)
    assert arr == expected


def test_unzip_sequence_fails_if_size_of_sequence_is_not_factor_of_n() -> None:
    """Test if an exception is raised when the size of sequence is not a factor of n."""
    with pytest.raises(ValueError) as e:
        unzip_sequence([1, 2, 3, 4, 5], 2)
    assert "cannot unzip data with specified number of lines (n=2)" in str(e.value)


@pytest.mark.parametrize(
    "sequence1,sequence2,expected",
    [
        ([1, 2, 3], [4, 5, 6], [1, 4, 2, 5, 3, 6]),
        ([7, 8, 9], np.array([1, 2, 3]), [7, 1, 8, 2, 9, 3]),
        (np.array([4, 5, 6]), np.array([7, 8, 9]), [4, 7, 5, 8, 6, 9]),
    ],
)
def test_zip_two_sequences_as_array(
    sequence1: list[int] | np.ndarray, sequence2: list[int] | np.ndarray, expected: list[int]
) -> None:
    """Test that given two sequences are zipped correctly and return as a numpy
    array."""
    arr = zip_sequences_as_array(sequence1, sequence2)
    assert type(arr) == np.ndarray
    assert_array_equal(arr, expected)


@pytest.mark.parametrize(
    "sequence1,sequence2,sequence3,expected",
    [
        ([1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7, 2, 5, 8, 3, 6, 9]),
        ([4, 5, 6], [7, 8, 9], np.array([1, 2, 3]), [4, 7, 1, 5, 8, 2, 6, 9, 3]),
        (np.array([7, 8, 9]), [1, 2, 3], np.array([4, 5, 6]), [7, 1, 4, 8, 2, 5, 9, 3, 6]),
    ],
)
def test_zip_three_sequences_as_array(
    sequence1: list[int] | np.ndarray,
    sequence2: list[int] | np.ndarray,
    sequence3: list[int] | np.ndarray,
    expected: list[int],
) -> None:
    """Test that given three sequences are zipped correctly and return as a numpy
    array."""
    arr = zip_sequences_as_array(sequence1, sequence2, sequence3)
    assert type(arr) == np.ndarray
    assert_array_equal(arr, expected)


def test_zip_sequences_as_array_fails_if_sizes_of_sequences_not_match() -> None:
    """Test if an exception is raised when sizes of sequences do not match."""
    with pytest.raises(ValueError) as e:
        zip_sequences_as_array([1, 2, 3], [4, 5, 6, 7])
    assert "all input sequences must have the same size" in str(e.value)


@pytest.mark.parametrize(
    "sequence1,sequence2,expected",
    [
        ([1, 2, 3], [4, 5, 6], [1, 4, 2, 5, 3, 6]),
        ([7, 8, 9], np.array([1, 2, 3]), [7, 1, 8, 2, 9, 3]),
        (np.array([4, 5, 6]), np.array([7, 8, 9]), [4, 7, 5, 8, 6, 9]),
    ],
)
def test_zip_two_sequences(
    sequence1: list[int] | np.ndarray, sequence2: list[int] | np.ndarray, expected: list[int]
) -> None:
    """Test that the given two sequences are zipped correctly and return as a list."""
    arr = zip_sequences(sequence1, sequence2)
    assert arr == expected


@pytest.mark.parametrize(
    "sequence1,sequence2,sequence3,expected",
    [
        ([1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 4, 7, 2, 5, 8, 3, 6, 9]),
        ([4, 5, 6], [7, 8, 9], np.array([1, 2, 3]), [4, 7, 1, 5, 8, 2, 6, 9, 3]),
        (np.array([7, 8, 9]), [1, 2, 3], np.array([4, 5, 6]), [7, 1, 4, 8, 2, 5, 9, 3, 6]),
    ],
)
def test_zip_three_sequences(
    sequence1: list[int] | np.ndarray,
    sequence2: list[int] | np.ndarray,
    sequence3: list[int] | np.ndarray,
    expected: list[int],
) -> None:
    """Test that the given three sequences are zipped correctly and return as a list."""
    arr = zip_sequences(sequence1, sequence2, sequence3)
    assert arr == expected


def test_zip_sequences_fails_if_sizes_of_sequences_not_match() -> None:
    """Test if an exception is raised when sizes of sequences do not match."""
    with pytest.raises(ValueError) as e:
        zip_sequences([1, 2, 3], [4, 5, 6, 7])
    assert "all input sequences must have the same size" in str(e.value)


@pytest.mark.parametrize(
    "address,socket_type,nonblock",
    [
        (("localhost", 20202), SOCK_DGRAM, False),
        (("192.158.5.101", 50000), SOCK_STREAM, True),
    ],
)
def test_init_from_config(address: tuple[str, int], socket_type: int, nonblock: bool) -> None:
    config = {"host": address[0], "port": address[1], "type": socket_type, "nonblock": nonblock}
    s = IPv4Socket(config)
    assert s.address == address
    assert s.type == socket_type
    assert s.is_nonblocking() == nonblock


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"host": "localhost", "port": 50000}, IPv4Socket(("localhost", 50000))),
        ({"host": "127.0.0.1", "port": 50001, "type": SOCK_STREAM}, IPv4Socket(("127.0.0.1", 50001), SOCK_STREAM)),
        ({"host": "192.168.5.11", "port": 50002, "nonblock": True}, IPv4Socket(("192.168.5.11", 50002), nonblock=True)),
    ],
)
def test_init_from_incomplete_config(config: dict[str, Any], expected: IPv4Socket) -> None:
    s = IPv4Socket(config)
    assert s.address == expected.address
    assert s.type == expected.type
    assert s.is_nonblocking() == expected.is_nonblocking()


@pytest.mark.parametrize(
    "config,required_key",
    [({"port": 50000}, "host"), ({"host": "localhost"}, "port")],
)
def test_error_when_required_key_missing(config: dict[str, Any], required_key: str) -> None:
    with pytest.raises(KeyError) as e:
        _ = IPv4Socket(config)
    assert f"IPv4Socket: '{required_key}' is required in config." in str(e.value)


def test_check_repr(default_socket: IPv4Socket) -> None:
    assert repr(default_socket) == "IPv4Socket(('localhost', 50000), SOCK_DGRAM, nonblock=False)"


def test_check_str(default_socket: IPv4Socket) -> None:
    assert str(default_socket) == "IPv4Socket<localhost:50000> (UDP)"
    s = IPv4Socket(default_address(), SOCK_STREAM)
    assert str(s) == "IPv4Socket<localhost:50000> (TCP)"
    s = IPv4Socket(default_address(), nonblock=True)
    assert str(s) == "IPv4Socket<localhost:50000> (UDP,nonblock)"


@pytest.mark.parametrize("host,port", [("localhost", 12345), ("192.168.5.11", 22222)])
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
    s = IPv4Socket(("localhost", 12345), socket_type=socket_type)
    assert s.type == socket_type


def test_enable_blocking_mode_by_default(default_socket: IPv4Socket) -> None:
    """Test that the blocking mode is enabled by default."""
    assert default_socket.is_nonblocking() == False


@pytest.mark.parametrize("nonblock", [False, True])
def test_check_nonblocking_mode(nonblock: bool) -> None:
    """Check if `is_nonblocking` returns False in blocking mode."""
    s = IPv4Socket(("localhost", 12345), nonblock=nonblock)
    assert s.is_nonblocking() == nonblock


HOST = "localhost"
PORT = 50000


class EchoServer:
    """Mock an echo server.

    This is intended to be used in fixtures for pytest and the method `serve` should be
    run in a thread.
    """

    canned_response: bytes | None
    _counter = count(0)

    def __init__(self, canned_response: str | bytes | None = None):
        """Initialize an echo server.

        Parameters
        ----------
        canned_response : str | bytes, optional
            If `canned_response` is given, the serever responses the given bytes object or
            encoded string.
        """
        self.canned_response = None
        if canned_response is not None:
            self.set_canned_response(canned_response)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._cnt = next(self._counter)
        self.port = PORT + self._cnt

    def __enter__(self) -> EchoServer:
        self._socket.bind((HOST, self.port))
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        _ = exception_type, exception_value, traceback
        time.sleep(0.1)
        self._socket.close()

    @property
    def address(self) -> tuple[str, int]:
        return (HOST, self.port)

    def set_canned_response(self, canned_response: str | bytes) -> bytes:
        if isinstance(canned_response, str):
            canned_response = canned_response.encode()
        self.canned_response = canned_response
        return canned_response

    def serve(self) -> NoReturn:
        while True:
            self._socket.listen(5)
            conn, _ = self._socket.accept()
            if self.canned_response is None:
                msg = conn.recv(1024)
                response = msg
            else:
                response = self.canned_response
            conn.sendall(response)
            conn.close()


@pytest.fixture(scope="function")
def echo_server():
    """Create an echo server that just returns a received message."""
    server = EchoServer()
    with server as s:
        thread = threading.Thread(target=s.serve)
        thread.daemon = True
        thread.start()
        yield s


@pytest.fixture(scope="function")
def fixed_response_server(msg: str):
    """Create an echo server that just returns a fixed message."""
    server = EchoServer(canned_response=msg)
    with server as s:
        thread = threading.Thread(target=s.serve)
        thread.daemon = True
        thread.start()
        yield s


@pytest.mark.parametrize("msg", ["hello", "world"])
def test_send_and_recv(echo_server, msg: str) -> None:
    """Test that an echo server works."""
    s = IPv4Socket(echo_server.address, SOCK_STREAM)
    s.connect()
    s.send(msg.encode())
    data = s.recv()
    s.close()
    assert data.decode() == msg


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("0 1 2 3 4 5", [[0, 3], [1, 4], [2, 5]]),
        ("1 2 3 4 5 6 7 8 9", [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
    ],
)
def test_recv_as_list(fixed_response_server, msg: str, expected: list[list[int]]) -> None:
    """Test the returned value from `recv_as_list`."""
    fixed_response_server.set_canned_response(msg)
    s = IPv4Socket(fixed_response_server.address, SOCK_STREAM)
    s.connect()
    s.send(b"request")
    data = s.recv_as_list()
    s.close()
    assert data == expected


@pytest.mark.parametrize(
    "msg,expected",
    [
        ("0 1 2 3 4 5", [[0, 3], [1, 4], [2, 5]]),
        ("1 2 3 4 5 6 7 8 9", [[1, 4, 7], [2, 5, 8], [3, 6, 9]]),
    ],
)
def test_recv_as_array(fixed_response_server, msg: str, expected: np.ndarray) -> None:
    """Test the returned value from `recv_as_array`."""
    fixed_response_server.set_canned_response(msg)
    s = IPv4Socket(fixed_response_server.address, SOCK_STREAM)
    s.connect()
    s.send(b"request")
    data = s.recv_as_array()
    s.close()
    assert type(data) is np.ndarray
    assert_array_equal(data, expected)


@pytest.mark.parametrize(
    "sequences,expected",
    [
        (([1, 2, 3], [4, 5, 6]), "1 4 2 5 3 6"),
        ((np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])), "0 4 1 5 2 6 3 7"),
    ],
)
def test_send_sequences(echo_server, sequences: tuple[Sequence | np.ndarray], expected: str) -> None:
    """Test that an echo server works."""
    s = IPv4Socket(echo_server.address, SOCK_STREAM)
    s.connect()
    s.send_sequences(*sequences)
    data = s.recv()
    s.close()
    assert data.decode() == expected
