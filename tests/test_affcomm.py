import os
import socket

import pytest
from affctrllib._sockutil import SockAddr
from affctrllib.affcomm import AffComm

CONFIG_DIR_PATH = os.path.join(os.path.dirname(__file__), "config")


def vec2str(L, form="%3.2f"):
    sexp = "%%s %s" % form
    sexp0 = form
    print(f"sexp={sexp}")
    s = ""
    print(f"s={s}")
    for i in range(0, len(L)):
        if i > 0:
            s = sexp % (s, L[i])
            print(f"i={i}, s={s}")
        else:
            s = form % L[i]
            print(f"i={i}, s={s}")
    return s


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

    @pytest.mark.parametrize(
        "data,expected_array",
        [
            (b"1 2 3", [1.0, 2.0, 3.0]),
            (b"1.1 2.2 3.3 4.4 5.5 ", [1.1, 2.2, 3.3, 4.4, 5.5]),
            (b"  1  2  3  4  5  6 ", [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ],
    )
    def test_process_received_bytes(self, data, expected_array) -> None:
        acom = AffComm()
        arr = acom.process_received_bytes(data)
        assert arr == expected_array

    @pytest.mark.parametrize(
        "data,func,expected_array",
        [
            (b"1 2 3", int, [1, 2, 3]),
            (b"1 2 3 4 5 ", str, ["1", "2", "3", "4", "5"]),
        ],
    )
    def test_process_received_bytes_alternate_mapping(
        self, data, func, expected_array
    ) -> None:
        acom = AffComm()
        arr = acom.process_received_bytes(data, function=func)
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
    def test_process_received_bytes_alternate_sep(
        self, data, sep, expected_array
    ) -> None:
        acom = AffComm()
        arr = acom.process_received_bytes(data, sep=sep)
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
    def test_process_array_to_string(self, arr, expected_str) -> None:
        acom = AffComm()
        s = acom.process_array_to_string(arr)
        assert s == expected_str

    @pytest.mark.parametrize(
        "arr,sep,expected_str",
        [
            ([0, 1, 2], ",", "0,1,2"),
            ([0, 1, 2], "|", "0|1|2"),
            ([0, 1, 2], "  ", "0  1  2"),
        ],
    )
    def test_process_array_to_string_specify_sep(self, arr, sep, expected_str) -> None:
        acom = AffComm()
        s = acom.process_array_to_string(arr, sep=sep)
        assert s == expected_str

    @pytest.mark.parametrize(
        "arr,f_spec,expected_str",
        [
            ([0, 1, 2], "d", "0 1 2"),
            ([0, 1, 2], ".3f", "0.000 1.000 2.000"),
            ([1.333, 3.28, 5.5, 10.215], "05.2f", "01.33 03.28 05.50 10.21"),
        ],
    )
    def test_process_array_to_string_specify_f_spec(
        self, arr, f_spec, expected_str
    ) -> None:
        acom = AffComm()
        s = acom.process_array_to_string(arr, f_spec=f_spec)
        assert s == expected_str

    @pytest.mark.parametrize(
        "arr,precision,expected_str",
        [
            ([0.54892, 1.289285, 2.889013], "1", "0.5 1.3 2.9"),
            ([0.54892, 1.289285, 2.889013], "3", "0.549 1.289 2.889"),
            ([0.54892, 1.289285, 2.889013], "5", "0.54892 1.28929 2.88901"),
        ],
    )
    def test_process_array_to_string_specify_precision(
        self, arr, precision, expected_str
    ) -> None:
        acom = AffComm()
        s = acom.process_array_to_string(arr, precision=precision)
        assert s == expected_str

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
