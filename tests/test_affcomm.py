import os

import numpy as np
import pytest
from affctrllib._sockutil import Socket
from affctrllib.affcomm import (
    AffComm,
    convert_array_to_bytes,
    convert_array_to_string,
    split_received_msg,
    unzip_array,
    unzip_array_as_ndarray,
    zip_arrays,
)
from numpy.testing import assert_array_equal

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


@pytest.mark.parametrize(
    "data,expected_array",
    [
        (b"1 2 3", [1, 2, 3]),
        (b"1 2 3 4 5 ", [1, 2, 3, 4, 5]),
        (b"  1  2  3  4  5  6 ", [1, 2, 3, 4, 5, 6]),
    ],
)
def test_split_received_msg(data, expected_array) -> None:
    arr = split_received_msg(data)
    assert arr == expected_array


@pytest.mark.parametrize(
    "data,func,expected_array",
    [
        (b"1 2 3", float, [1.0, 2.0, 3.0]),
        (b"1 2 3 4 5 ", str, ["1", "2", "3", "4", "5"]),
    ],
)
def test_split_received_msg_alternate_mapping(data, func, expected_array) -> None:
    arr = split_received_msg(data, function=func)
    assert arr == expected_array


@pytest.mark.parametrize(
    "data,sep,expected_array",
    [
        (b"1 2 3", " ", [1, 2, 3]),
        (b"1 2 3 ", " ", [1, 2, 3]),
        (b"1,2,3,4,5", ",", [1, 2, 3, 4, 5]),
        (b"1,2,3,4,5,", ",", [1, 2, 3, 4, 5]),
    ],
)
def test_split_received_msg_alternate_sep(data, sep, expected_array) -> None:
    arr = split_received_msg(data, sep=sep)
    assert arr == expected_array


@pytest.mark.parametrize(
    "data,expected_array",
    [
        ("1 2 3", [1, 2, 3]),
        ("1 2 3 4 5 ", [1, 2, 3, 4, 5]),
        ("  1  2  3  4  5  6 ", [1, 2, 3, 4, 5, 6]),
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
def test_convert_array_to_string_specify_precision(arr, precision, expected_str) -> None:
    s = convert_array_to_string(arr, precision=precision)
    assert s == expected_str


@pytest.mark.parametrize(
    "arr,sep,f_spec,expected",
    [
        (np.array([0, 1, 2]), " ", ".0f", "0 1 2"),
        (np.array([0, 1, 2]), ",", ".2f", "0.00,1.00,2.00"),
    ],
)
def test_convert_array_to_string_ndarray(arr, sep, f_spec, expected) -> None:
    s = convert_array_to_string(arr, sep=sep, f_spec=f_spec)
    assert s == expected


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
def test_unzip_array_as_ndarray(arr, expected) -> None:
    ret = unzip_array_as_ndarray(arr)
    assert_array_equal(ret, expected)


@pytest.mark.parametrize(
    "arr,_",
    [
        (range(5), np.array([[0, 3], [1, 4], [2, 0]])),
        (range(10), np.array([[0, 3, 6, 9], [1, 4, 7, 0], [2, 5, 8, 0]])),
    ],
)
def test_unzip_array_as_ndarray_not_divisible(arr, _) -> None:
    with pytest.raises(ValueError) as excinfo:
        _ = unzip_array_as_ndarray(arr)
    assert "cannot reshape array of size" in str(excinfo.value)


@pytest.mark.parametrize(
    "arr,ncol,expected",
    [
        (range(8), 4, np.array([[0, 4], [1, 5], [2, 6], [3, 7]])),
        (range(8), 2, np.array([[0, 2, 4, 6], [1, 3, 5, 7]])),
    ],
)
def test_unzip_array_as_ndarray_specify_ncol(arr, ncol, expected) -> None:
    ret = unzip_array_as_ndarray(list(arr), ncol=ncol)
    assert_array_equal(ret, expected)


@pytest.mark.parametrize(
    "arr,ncol,expected",
    [
        (np.array(range(6)), 3, np.array([[0, 3], [1, 4], [2, 5]])),
        (np.array(range(10)), 2, np.array([[0, 2, 4, 6, 8], [1, 3, 5, 7, 9]])),
    ],
)
def test_unzip_array_as_ndarray_ndarray(arr, ncol, expected) -> None:
    ret = unzip_array_as_ndarray(arr, ncol=ncol)
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


@pytest.mark.parametrize(
    "arr,ncol,expected",
    [
        (np.array(range(6)), 3, [[0, 3], [1, 4], [2, 5]]),
        (np.array(range(8)), 2, [[0, 2, 4, 6], [1, 3, 5, 7]]),
    ],
)
def test_reshape_unzip_ndarray(arr, ncol, expected) -> None:
    ret = unzip_array(list(arr), n=ncol)
    assert ret == expected


@pytest.mark.parametrize(
    "arr1,arr2,expected",
    [
        ([0, 1, 2], [3, 4, 5], [0, 3, 1, 4, 2, 5]),
        (range(5), range(5, 10), [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]),
    ],
)
def test_zip_arrays(arr1, arr2, expected) -> None:
    ret = zip_arrays(arr1, arr2)
    assert ret == expected


@pytest.mark.parametrize(
    "arr1,arr2,arr3,expected",
    [
        ([0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6, 1, 4, 7, 2, 5, 8]),
        (
            range(5),
            range(5, 10),
            range(10, 15),
            [0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14],
        ),
    ],
)
def test_zip_arrays_3args(arr1, arr2, arr3, expected) -> None:
    ret = zip_arrays(arr1, arr2, arr3)
    assert ret == expected


@pytest.mark.parametrize(
    "arr1,arr2,expected",
    [
        (np.array([0, 1, 2]), np.array([3, 4, 5]), [0, 3, 1, 4, 2, 5]),
        (np.array(range(5)), np.array(range(5, 10)), [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]),
    ],
)
def test_zip_arrays_ndarray(arr1, arr2, expected) -> None:
    ret = zip_arrays(arr1, arr2)
    assert ret == expected


class TestAffComm:
    def test_init(self) -> None:
        acom = AffComm()
        assert isinstance(acom.sensory_socket, Socket)
        assert isinstance(acom.command_socket, Socket)

    @pytest.mark.parametrize(
        "config_file,remote_host",
        [
            ("default.toml", "192.168.1.1"),
            ("alternative.toml", "192.168.5.10"),
        ],
    )
    def test_init_load_config(self, config_file, remote_host) -> None:
        acom = AffComm(os.path.join(CONFIG_DIR_PATH, config_file))
        assert acom.command_socket.host == remote_host

    def test_repr(self) -> None:
        acom = AffComm()
        assert repr(acom) == "affctrllib.affcomm.AffComm()"

    def test_str(self) -> None:
        acom = AffComm()
        assert (
            str(acom)
            == """\
AffComm configuration:
  Config file: None
   Receive at: No address is provided
      Send to: No address is provided
"""
        )

    def test_str_load_config(self) -> None:
        cpath = os.path.join(CONFIG_DIR_PATH, "default.toml")
        acom = AffComm(cpath)
        assert (
            str(acom)
            == f"""\
AffComm configuration:
  Config file: {cpath}
   Receive at: localhost:50000
      Send to: 192.168.1.1:50010
"""
        )

    def test_load_config_default(self) -> None:
        acom = AffComm()
        acom.load_config_path(os.path.join(CONFIG_DIR_PATH, "default.toml"))

        # sensory_socket
        assert acom.sensory_socket.host == "localhost"
        assert acom.sensory_socket.port == 50000

        # command_socket
        assert acom.command_socket.host == "192.168.1.1"
        assert acom.command_socket.port == 50010

    def test_load_config_alternative(self) -> None:
        acom = AffComm()
        acom.load_config_path(os.path.join(CONFIG_DIR_PATH, "alternative.toml"))

        # sensory_socket
        assert acom.sensory_socket.host == "192.168.5.123"
        assert acom.sensory_socket.port == 60000

        # command_socket
        assert acom.command_socket.host == "192.168.5.10"
        assert acom.command_socket.port == 60010
