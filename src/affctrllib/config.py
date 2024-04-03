"""This module provides a class to manage a configuration file.

It provides a class that contains functions to load a configuration file written in
TOML, store the configuration file path, get values from specified keys, and so on.
"""

from __future__ import annotations

import sys
from pathlib import Path
from traceback import format_tb
from typing import Any, TypeVar, overload

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib

CONFIG_GET_VALUE_DEFAULT_T = TypeVar("CONFIG_GET_VALUE_DEFAULT_T")


class CONFIG_GET_VALUE_NO_DEFAULT_T(int):
    pass


CONFIG_GET_VALUE_NO_DEFAULT = CONFIG_GET_VALUE_NO_DEFAULT_T(-1)


class Configuration(object):
    """The configuration class.

    This class is implemented as an abstract class.
    """

    _path: Path
    _mapping: dict[str, Any]

    def __init__(self, path: str | Path | None = None) -> None:
        """Initialize the Configuration object.

        Parameters
        ----------
        path : str | Path, optional
            A string or a Path object containing a TOML filename.

        Raises
        ------
        RuntimeError
            If loading is terminated with an error.
        """

        if path is None:
            return

        self.load_mapping(path)

    def __getitem__(self, key: str) -> Any:
        return self.mapping[key]

    def __contains__(self, key: str) -> bool:
        return key in self.mapping

    @property
    def path(self) -> Path:
        """Return `Path` if the object is loaded from a file.

        Return a `Path` object if the `Configuration` object has been
        loaded from a configuration file. Otherwise, it raises an
        exception.

        Returns
        -------
        Path
            Path object that the Configuration has been loaded from.

        Raises
        ------
        AttributeError
            If the Configuration object is not created from a file.
        """

        try:
            return self._path
        except AttributeError as err:
            msg = (
                "'Configuration' object is not loaded from configuration file.\n"
                + format_tb(err.__traceback__)[0]
                + err.args[0]
            )
            raise AttributeError(msg) from None

    @property
    def mapping(self) -> dict[str, Any]:
        """Return a dict loaded from TOML file or string.

        Return a dict object loaded from TOML file or string. If the
        Configuration object has not been initialized yet, raises an
        exception.

        Returns
        -------
        dict[str, Any]
            A dict object loaded from TOML file or string.

        Raises
        ------
        AttributeError
            If the Configuration object has not been initialized yet.
        """

        try:
            return self._mapping
        except AttributeError as err:
            msg = "'Configuration' object has not initialized yet.\n" + format_tb(err.__traceback__)[0] + err.args[0]
            raise AttributeError(msg) from None

    def is_loaded_from_file(self) -> bool:
        """Check if the object is loaded from a file.

        Retrun True if the object is loaded from a TOML file.
        Otherwise, it returns False.

        Returns
        -------
        bool
            True if the object is loaded from a file.
        """

        return hasattr(self, "_path")

    @classmethod
    def load_from_mapping(cls, mapping: dict[str, Any]) -> Configuration:
        """Initialize the Configuration class from given mapping.

        Parameters
        ----------
        mapping : dict[str, Any]
            A mapping to be set in initialized object.

        Returns
        -------
        Configuration
            Initialized object that the given mapping is set.

        Examples
        --------
        >>> c = Configuration.load_from_mapping({"robot": {"name": "sample"}})
        >>> c.get_value("robot", "name")
        "sample"
        >>> c.is_loaded_from_file()
        False
        """

        c = Configuration()
        c.set_mapping(mapping)
        return c

    def load_mapping(self, path: str | Path) -> dict[str, Any]:
        """Read a TOML file.

        This method just reads a TOML file. To parse the loaded
        object, the `load` method needs to be invoked.

        Parameters
        ----------
        path : str | Path
            A `Path` object or a string containing the filename.

        Returns
        -------
        dict[str, Any]
            A dict object loaded from a TOML file.

        Examples
        --------
        >>> c = Configuration()
        >>> c.load_mapping("config.toml")
        >>> c.load()
        """

        self._path = Path(path)
        with open(path, "rb") as f:
            self._set_mapping(tomllib.load(f))
        return self._mapping

    def _set_mapping(self, mapping: dict[str, Any]) -> dict[str, Any]:
        self._mapping = mapping.copy()
        return self._mapping

    def set_mapping(self, mapping: dict[str, Any], delete_path: bool = True) -> dict[str, Any]:
        """Set a mapping.

        This method just copies a provided mapping object. To parse
        the loaded object, the `load` method needs to be invoked.

        Parameters
        ----------
        mapping : dict[str, Any]
            A dict object that will be copied inside.
        delete_path : bool, default=True
            By default, the `path` attribute will be deleted when
        `set_mapping` is called. However, if `delete_path` is False,
        the `path` attribute will be left intact.

        Returns
        -------
        dict[str, Any]
            A dict object copied from the provided.

        Examples
        --------
        >>> c = Configuration()
        >>> c.set_mapping({"robot": {"name": "hoge"}})
        >>> c.load()
        """

        self._set_mapping(mapping)
        if hasattr(self, "_path") and delete_path:
            del self._path
        return self._mapping

    @staticmethod
    def _get_value(mapping: dict[str, Any], *keys: str) -> Any:
        if len(keys) == 1:
            return mapping[keys[0]]
        else:
            return Configuration._get_value(mapping[keys[0]], *keys[1:])

    @staticmethod
    def _get_value_default(
        mapping: dict[str, Any], *keys: str, default: CONFIG_GET_VALUE_DEFAULT_T
    ) -> CONFIG_GET_VALUE_DEFAULT_T:
        if len(keys) == 1:
            return mapping.get(keys[0], default)
        else:
            return Configuration._get_value_default(mapping[keys[0]], *keys[1:], default=default)

    @overload
    def get_value(self) -> dict[str, Any]: ...

    @overload
    def get_value(self, *keys: str) -> Any: ...

    @overload
    def get_value(self, *keys: str, default: CONFIG_GET_VALUE_DEFAULT_T) -> CONFIG_GET_VALUE_DEFAULT_T: ...

    def get_value(self, *keys, default=CONFIG_GET_VALUE_NO_DEFAULT):
        """Get value from given key sequence.

        Get a value or a dictionary object from keys provided as a
        sequence. If `default` is given, it returns the default value
        when the last key is not found in the mapping.

        Parameters
        ----------
        *keys : str
            A sequence of keys.
        default : Any, optional
            When the last key in the provided keys sequence does not
        match with the mapping, returns the default value.

        Examples
        --------
        >>> c = Configuration.load_from_mapping({"robot": {"name": "sample"}})
        >>> c.get_value("robot", "name")
        "sample"
        >>> c.get_value("robot", "dof", default=13)
        13
        """

        if len(keys) == 0:
            return self.mapping

        if isinstance(default, CONFIG_GET_VALUE_NO_DEFAULT_T) and default == CONFIG_GET_VALUE_NO_DEFAULT:
            value = self._get_value(self.mapping, *keys)
        else:
            value = self._get_value_default(self.mapping, *keys, default=default)
        return value
