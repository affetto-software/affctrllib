"""This module encompasses utilities to handle the socket interface and process data.

Currently, it is structured to facilitate communication with NEDO Affetto through a UDP
connection over the Internet. To facilitate the communication, this module is equipped
to instantiate a socket object and encapsulate the requisite methods.
"""

from __future__ import annotations

import socket
from socket import AF_INET, SOCK_DGRAM, SOCK_NONBLOCK, SOCK_STREAM
from typing import Callable, Iterable, Sequence, TypeVar

import numpy as np

# The default maximum amount of data when receiving or sending used in this
# module.
DEFAULT_BUFSIZE = 4096

# Define a generic type.
T = TypeVar("T")


class IPv4Socket(object):
    """Container of a socekt object using IPv4 address family.

    This class contains a socket object, the type of the socket, address information to
    communicate with.

    Attributes
    ----------
    socket
    address
    host
    port
    family
    type
    proto
    """

    _socket: socket.socket
    _address: tuple[str, int]
    _TYPE_NAMES = {f"{SOCK_DGRAM}": "SOCK_DGRAM", f"{SOCK_STREAM}": "SOCK_STREAM"}

    def __init__(self, address: tuple[str, int], socket_type: int = SOCK_DGRAM, nonblock: bool = False) -> None:
        """Initialize the IPv4 socket.

        Parameters
        ----------
        address : tuple[str, int]
            The internet address consisting of a 2-tuple (host, port).

        socket_type : {SOCK_DGRAM, SOCK_STREAM}, optional
            The socket type. Choose between SOCK_STREAM and SOCK_DGRAM.

        nonblock : bool, optional
            If True, apply SOCK_NONBLOCK bit flag when creating the socket object.
        """
        self._address = address
        if nonblock:
            self._socket = socket.socket(AF_INET, socket_type | SOCK_NONBLOCK)
        else:
            self._socket = socket.socket(AF_INET, socket_type)

    def __repr__(self) -> str:
        type_name = self._TYPE_NAMES[f"{self.type}"]
        nonblock = "True" if self.is_nonblocking() else "False"
        return f"{self.__class__.__qualname__}({self.address}, {type_name}, nonblock={nonblock})"

    def __str__(self) -> str:
        type_name = self._TYPE_NAMES[f"{self.type}"]
        attr = {"SOCK_DGRAM": "UDP", "SOCK_STREAM": "TCP"}[type_name]
        if self.is_nonblocking():
            attr += ",nonblock"
        return f"{self.__class__.__qualname__}<{self.host}:{self.port}> ({attr})"

    @property
    def socket(self) -> socket.socket:
        """The created socket object."""
        return self._socket

    @property
    def address(self) -> tuple[str, int]:
        """The internet address specified at initialization."""
        return self._address

    @property
    def host(self) -> str:
        """The hostname or IPv4 address as a string."""
        return self._address[0]

    @property
    def port(self) -> int:
        """The port number."""
        return self._address[1]

    @property
    def family(self) -> int:
        """The socket family."""
        return self._socket.family

    @property
    def type(self) -> int:
        """The socket type."""
        return self._socket.type

    @property
    def proto(self) -> int:
        """The socket protocol."""
        return self._socket.proto

    def is_nonblocking(self) -> bool:
        """Return True if socket is in non-blocking mode."""
        return not self._socket.getblocking()

    def bind(self) -> None:
        """Bind the socket to the associated address."""
        self.socket.bind(self.address)

    def connect(self, address: tuple[str, int]) -> None:
        """Conncet to a remote socket at `address`.

        ...
        Parameters
        ----------
        address : tuple[str, int]
            A remote address to connect.
        """
        return self.socket.connect(address)

    def recv(self, bufsize: int = DEFAULT_BUFSIZE) -> bytes:
        """Receive data from the socket.

        Receive data from the socket. The return value is a bytes object the
        socket received.

        Parameters
        ----------
        bufsize : int, optional
            The maximum amount of data to be received at once.

        Returns
        -------
        bytes
            A bytes object that the socket received.
        """
        recv_bytes, _ = self.socket.recvfrom(bufsize)
        # recv_bytes = self.socket.recv(bufsize)
        return recv_bytes

    def send(self, send_bytes: bytes) -> int:
        """Send data to the socket.

        The socket must be connected to a remote socket. Returns the number of
        bytes sent.

        Parameters
        ----------
        send_bytes : bytes
            The bytes object to be sent to the socket.

        Returns
        -------
        int
            The number of bytes sent to the socket.
        """
        return self.socket.sendto(send_bytes, self.address)

    def close(self) -> None:
        """Close the socket.

        The underlying system resource (e.g., a file descriptor) is also closed when all
        file objects are closed. Once this happens, all future operations on the socket
        object will fail.
        """
        return self.socket.close()


def split_data(
    data: bytes | str,
    converter: Callable[[str], T] = int,
    sep: str | None = None,
    strip: bool | str = True,
) -> list[T]:
    """Split received data from a remote socket into a list of items.

    Parameters
    ----------
    data : bytes | str
        Received data object from a remote socket.
    converter : Callable[[str], T], default=int
        Callable function to convert a string to a desired item.
    sep : str, optional
        If `sep` is given, split `data` with that. It can be multiple characters.
    strip : bool | str, default=True
        If `strip` is True, strip whitespaces from the both sides of `data`
    before processing. If `strip` is a series of characters, strip those.

    Returns
    -------
    list[T]
        A list of items converted by `converter`.

    Raises
    ------
    TypeError
        If `data` is neither a bytes object nor a string.
    """
    if isinstance(data, bytes):
        decoded_data = data.decode()
    elif isinstance(data, str):
        decoded_data = data
    else:
        raise TypeError(f"Unsupported type: {type(data)}")
    if strip is True:
        decoded_data = decoded_data.strip()
    elif strip:
        decoded_data = decoded_data.strip(strip)
    return list(map(converter, decoded_data.split(sep)))


def join_data(
    data: Iterable,
    sep: str = " ",
    specifier: str = ".0f",
    precision: int | None = None,
) -> str:
    """Given a list of values, return a string joined items in the list.

    The `data` can be an iterable object as well as a list. Delimiter to join items can
    be specified with `sep`. Format specifier, as known as 'f-string', to format each
    element in the `data` can be specified with `specifier`. When elements in the data
    are floats, their precision can be specified with `precision`.

    Parameters
    ----------
    data : Iterable
        A list of values to be joined. Any iterable object can be given as well.
    sep : str, default=\" \"
        Delimiter to join items. Default value is a single space.
    specifier : str, default=\".0f\"
        Format specifier to format items as string.
    precision : int, optional
        Precision to be formatted when items are float.

    Returns
    -------
    str
        A string where given data is joined with format specifier.

    Examples
    --------
    >>> join_data([1, 2, 3])
    \"1 2 3\"

    See Also
    --------
    eoncode_data : Returns a bytes object.
    """
    if precision is None:
        return sep.join([f"{x:{specifier}}" for x in data])
    else:
        return sep.join([f"{x:.{precision}f}" for x in data])


def encode_data(
    data: Iterable,
    sep: str = " ",
    specifier: str = ".0f",
    precision: int | None = None,
) -> bytes:
    """Given a list of values, return a bytes object encoded a joined string.

    The `data` can be an iterable object as well as a list. Delimiter to join items can
    be specified with `sep`. Format specifier, as known as 'f-string', to format each
    element in the `data` can be specified with `specifier`. When elements in the data
    are floats, their precision can be specified with `precision`.

    Parameters
    ----------
    data : Iterable
        A list of values to be joined. Any iterable object can be given as well.
    sep : str, default=\" \"
        Delimiter to join items. Default value is a single space.
    specifier : str, default=\".0f\"
        Format specifier to format items as string.
    precision : int, optional
        Precision to be formatted when items are float.

    Returns
    -------
    bytes
        A bytes object encoded a string where given data is joined with format
    specifier.

    Examples
    --------
    >>> encode_data([1, 2, 3])
    b\"1 2 3\"

    See Also
    --------
    join_data : Returns a string.
    """
    return join_data(data, sep, specifier, precision).encode()


def unzip_items_as_array(items: Sequence | np.ndarray, n: int) -> np.ndarray:
    """Return a numpy array where each row selects every `n`-th item in `items`.

    This function returns a two-dimensional numpy array where each row consists
    of items picked up every `n`-th value in the given list, `items`. If the
    length of the list is not exactly divisible by `n`, raises `ValueError`.

    Parameters
    ----------
    items : Sequence
        A list of items. It should be a one-dimentional list.
    n : int
        Pick up every `n`-th item in the list.

    Returns
    -------
    np.ndarray
        A numpy array having `n` rows.

    Raises
    ------
    ValueError
        If the length of `items` is not exactly divisible by `n`.

    See Also
    --------
    unzip_items : Return a list of lists instead of a numpy array.
    """
    try:
        # Convert a list of items into a two-dimentional array with `n` columns.
        arr = np.asarray(items).reshape((int(len(items) / n), n))
    except ValueError as e:
        if len(items) % n != 0:
            raise ValueError(f"cannot unzip data with specified number of lines (n={n})")
        else:
            # Raise unexpected error as it is.
            raise ValueError(str(e))
    # Transpose it to get each sequence as a row.
    return arr.T


def unzip_items(items: Sequence[T] | np.ndarray, n: int) -> list[list[T]]:
    """Return a list of lists where an inner list selects every `n`-th item in `items`.

    This function is designed to discompose a one-dimentional list (array) into
    `n` lines, where each line consists of every `n`-th item is selected from
    the given list.

    Parameters
    ----------
    items : Sequence
        A list of items. It should be a one-dimentional list.
    n : int
        Pick up every `n`-th item in the list.

    Returns
    -------
    list[list[T]]
        A list of `n` lists where each inner list consists of every `n`-th item
    in the given `items`.

    Raises
    ------
    ValueError
        If the length of `items` is not exactly divisible by `n`.

    See Also
    --------
    unzip_items_as_array : Return a numpy array composed of unzipped lines.

    Examples
    --------
    >>> unzip_items([1, 2, 3, 4, 5, 6, 7, 8, 9], 3)
    [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

    Let's say we have a list of ints, [1, 2, 3, 4, 5, 6, 7, 8, 9]. This function
    unzips the list into 3, passed as `n`, lines shown as above:
    """
    return unzip_items_as_array(items, n).tolist()


def zip_items_as_array(*items: Sequence | np.ndarray) -> np.ndarray:
    """Return a one-dimensional numpy array that zipped given items.

    This function returns a one-dimensional numpy array consisting of items which
    picked up alternately from given sequences. If the size of each sequence do
    not match, raises an error.

    Parameters
    ----------
    *items : Sequence | np.ndarray
        Lists of items to be zipped into one.

    Returns
    -------
    np.ndarray
        A numpy array obtained by being zipped given items.

    Raises
    ------
    ValueError
        If not all the sizes of given items are same.

    See Also
    --------
    zip_items : Return a list composed of zipped items.
    """
    try:
        arr = np.stack(items, axis=1)
    except ValueError as e:
        sizes = list(map(len, items))
        if not all(sizes[0] == x for x in sizes):
            raise ValueError("all input items must have the same size")
        else:
            raise ValueError(str(e))
    return arr.flatten()


def zip_items(*items: Sequence[T] | np.ndarray) -> list[T]:
    """Return a list that zipped given items.

    This function returns a list of values consisting of items which picked up
    alternately from given sequences. If the size of each sequence do not match,
    raises an error.

    Parameters
    ----------
    *items : Sequence[T] | np.ndarray
        Lists of items to be zipped into one.

    Returns
    -------
    list[T]
        A list obtained by being zipped given items.

    See Also
    --------
    zip_items_as_array : Return a numpy array composed of zipped items.

    Examples
    --------
    >>> zip_items([1, 2, 3], [4, 5, 6])
    [1, 4, 2, 5, 3, 6]

    Given two lists of values, it returns a list where each element is picked up
    alternately from the given lists.
    """
    return list(zip_items_as_array(*items))
