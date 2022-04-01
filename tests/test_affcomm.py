import os
import socket

import numpy as np
import pytest
from affctrllib._sockutil import SockAddr
from affctrllib.affcomm import (
    AffComm,
    convert_array_to_bytes,
    convert_array_to_string,
    reshape_array_for_unzip,
    split_received_msg,
    unzip_array,
)
from numpy.testing import assert_array_equal

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


@pytest.mark.parametrize(
    "data,expected_array",
    [
        (b"1 2 3", [1.0, 2.0, 3.0]),
        (b"1.1 2.2 3.3 4.4 5.5 ", [1.1, 2.2, 3.3, 4.4, 5.5]),
        (b"  1  2  3  4  5  6 ", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ],
)
def test_split_received_msg(data, expected_array) -> None:
    arr = split_received_msg(data)
    assert arr == expected_array


@pytest.mark.parametrize(
    "data,func,expected_array",
    [
        (b"1 2 3", int, [1, 2, 3]),
        (b"1 2 3 4 5 ", str, ["1", "2", "3", "4", "5"]),
    ],
)
def test_split_received_msg_alternate_mapping(data, func, expected_array) -> None:
    arr = split_received_msg(data, function=func)
    assert arr == expected_array


@pytest.mark.parametrize(
    "data,sep,expected_array",
    [
        (b"1 2 3", " ", [1.0, 2.0, 3.0]),
        (b"1 2 3 ", " ", [1.0, 2.0, 3.0]),
        (b"1,2,3,4,5", ",", [1.0, 2.0, 3.0, 4.0, 5.0]),
        (b"1,2,3,4,5,", ",", [1.0, 2.0, 3.0, 4.0, 5.0]),
    ],
)
def test_split_received_msg_alternate_sep(data, sep, expected_array) -> None:
    arr = split_received_msg(data, sep=sep)
    assert arr == expected_array


@pytest.mark.parametrize(
    "data,expected_array",
    [
        ("1 2 3", [1.0, 2.0, 3.0]),
        ("1.1 2.2 3.3 4.4 5.5 ", [1.1, 2.2, 3.3, 4.4, 5.5]),
        ("  1  2  3  4  5  6 ", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
    ],
)
def test_split_received_msg_provide_string(data, expected_array) -> None:
    arr = split_received_msg(data)
    assert arr == expected_array


@pytest.mark.parametrize(
    "data,sep,expected_array",
    [
        ("1 2 3 ", None, ["1", "2", "3"]),
        (" 1 2 3 ", " ", ["", "1", "2", "3", ""]),
        ("1,2,3,", ",", ["1", "2", "3", ""]),
    ],
)
def test_split_received_msg_no_strip(data, sep, expected_array) -> None:
    arr = split_received_msg(data, function=str, sep=sep, strip=False)
    assert arr == expected_array


@pytest.mark.parametrize(
    "arr,expected_str",
    [
        ([0, 1, 2], "0 1 2"),
        ([0, 1, 2, 3, 4], "0 1 2 3 4"),
        ([1.2, 3.2, 0.4, 8.7, 5.5], "1 3 0 9 6"),
        ([0.5, 1.5, 2.5, 3.5, 4.5], "0 2 2 4 4"),
    ],
)
def test_convert_array_to_string(arr, expected_str) -> None:
    s = convert_array_to_string(arr)
    assert s == expected_str


@pytest.mark.parametrize(
    "arr,sep,expected_str",
    [
        ([0, 1, 2], ",", "0,1,2"),
        ([0, 1, 2], "|", "0|1|2"),
        ([0, 1, 2], "  ", "0  1  2"),
    ],
)
def test_convert_array_to_string_specify_sep(arr, sep, expected_str) -> None:
    s = convert_array_to_string(arr, sep=sep)
    assert s == expected_str


@pytest.mark.parametrize(
    "arr,f_spec,expected_str",
    [
        ([0, 1, 2], "d", "0 1 2"),
        ([0, 1, 2], ".3f", "0.000 1.000 2.000"),
        ([1.333, 3.28, 5.5, 10.215], "05.2f", "01.33 03.28 05.50 10.21"),
    ],
)
def test_convert_array_to_string_specify_f_spec(arr, f_spec, expected_str) -> None:
    s = convert_array_to_string(arr, f_spec=f_spec)
    assert s == expected_str


@pytest.mark.parametrize(
    "arr,precision,expected_str",
    [
        ([0.54892, 1.289285, 2.889013], "1", "0.5 1.3 2.9"),
        ([0.54892, 1.289285, 2.889013], "3", "0.549 1.289 2.889"),
        ([0.54892, 1.289285, 2.889013], "5", "0.54892 1.28929 2.88901"),
    ],
)
def test_convert_array_to_string_specify_precision(
    arr, precision, expected_str
) -> None:
    s = convert_array_to_string(arr, precision=precision)
    assert s == expected_str


@pytest.mark.parametrize(
    "arr,expected_bytes",
    [
        ([0, 1, 2], b"0 1 2"),
        ([0, 1, 2, 3, 4], b"0 1 2 3 4"),
        ([1.2, 3.2, 0.4, 8.7, 5.5], b"1 3 0 9 6"),
        ([0.5, 1.5, 2.5, 3.5, 4.5], b"0 2 2 4 4"),
    ],
)
def test_convert_array_to_bytes(arr, expected_bytes) -> None:
    b = convert_array_to_bytes(arr)
    assert b == expected_bytes


@pytest.mark.parametrize(
    "arr,expected",
    [
        (range(6), np.array([[0, 3], [1, 4], [2, 5]])),
        ([1, 3, 5, 7, 9, 11], np.array([[1, 7], [3, 9], [5, 11]])),
        (range(15), np.array(range(15)).reshape(5, 3).T),
        (np.arange(9), np.arange(9).reshape(3, 3).T),
    ],
)
def test_reshape_array_for_unzip(arr, expected) -> None:
    ret = reshape_array_for_unzip(arr)
    assert_array_equal(ret, expected)


@pytest.mark.parametrize(
    "arr,_",
    [
        (range(5), np.array([[0, 3], [1, 4], [2, 0]])),
        (range(10), np.array([[0, 3, 6, 9], [1, 4, 7, 0], [2, 5, 8, 0]])),
    ],
)
def test_reshape_array_for_unzip_not_divisible(arr, _) -> None:
    with pytest.raises(ValueError) as excinfo:
        _ = reshape_array_for_unzip(arr)
    assert "cannot reshape array of size" in str(excinfo.value)


@pytest.mark.parametrize(
    "arr,ncol,expected",
    [
        (range(8), 4, np.array([[0, 4], [1, 5], [2, 6], [3, 7]])),
        (range(8), 2, np.array([[0, 2, 4, 6], [1, 3, 5, 7]])),
    ],
)
def test_reshape_array_for_unzip_specify_ncol(arr, ncol, expected) -> None:
    ret = reshape_array_for_unzip(list(arr), ncol=ncol)
    assert_array_equal(ret, expected)


@pytest.mark.parametrize(
    "arr,expected",
    [
        (range(6), [[0, 3], [1, 4], [2, 5]]),
        ([1, 3, 5, 7, 9, 11], [[1, 7], [3, 9], [5, 11]]),
        (range(15), [[0, 3, 6, 9, 12], [1, 4, 7, 10, 13], [2, 5, 8, 11, 14]]),
        (np.arange(9), [[0.0, 3.0, 6.0], [1.0, 4.0, 7.0], [2.0, 5.0, 8.0]]),
    ],
)
def test_unzip_array(arr, expected) -> None:
    ret = unzip_array(arr)
    assert ret == expected


@pytest.mark.parametrize(
    "arr,ncol,expected",
    [
        (range(8), 4, [[0, 4], [1, 5], [2, 6], [3, 7]]),
        (range(8), 2, [[0, 2, 4, 6], [1, 3, 5, 7]]),
    ],
)
def test_reshape_unzip_specify_ncol(arr, ncol, expected) -> None:
    ret = unzip_array(list(arr), n=ncol)
    assert ret == expected


class TestAffComm:
    def test_init(self) -> None:
        acom = AffComm()
        assert acom.config_path is None
        assert isinstance(acom.remote_addr, SockAddr)
        assert isinstance(acom.local_addr, SockAddr)

    @pytest.mark.parametrize(
        "config_file,remote_host",
        [
            ("default.toml", "192.168.1.1"),
            ("alternative.toml", "192.168.5.10"),
        ],
    )
    def test_init_load_config(self, config_file, remote_host) -> None:
        acom = AffComm(os.path.join(CONFIG_DIR_PATH, config_file))
        assert acom.remote_addr.host == remote_host

    def test_repr(self) -> None:
        acom = AffComm()
        assert repr(acom) == "affctrllib.affcomm.AffComm()"

    def test_load_config_default(self) -> None:
        acom = AffComm()
        acom.load_config(os.path.join(CONFIG_DIR_PATH, "default.toml"))

        # remote_addr
        assert acom.remote_addr.host == "192.168.1.1"
        assert acom.remote_addr.port == 50010

        # local_addr
        assert acom.local_addr.host == "localhost"
        assert acom.local_addr.port == 50000

    def test_load_config_alternative(self) -> None:
        acom = AffComm()
        acom.load_config(os.path.join(CONFIG_DIR_PATH, "alternative.toml"))

        # remote_addr
        assert acom.remote_addr.host == "192.168.5.10"
        assert acom.remote_addr.port == 60010

        # local_addr
        assert acom.local_addr.host == "192.168.5.123"
        assert acom.local_addr.port == 60000

    @pytest.mark.skip
    def test_create_sensory_socket(self) -> None:
        acom = AffComm()
        address = ("localhost", 11111)
        bufsize = 1024
        ssock = acom.create_sensory_socket(address)
        data, _ = ssock.recvfrom(bufsize)
        assert data.decode() == "hello world"

        # sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # msg = b"hello world"
        # sender.sendto(msg, address)
