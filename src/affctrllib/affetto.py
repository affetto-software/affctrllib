import sys
import warnings
from abc import abstractmethod
from pathlib import Path
from typing import Any, final

if sys.version_info < (3, 11):
    import tomli as tomllib
else:
    import tomllib


class Chain(object):
    _dof: int

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            self.load(config)

    @property
    def dof(self) -> int:
        return self._dof

    def set_dof(self, dof: int) -> None:
        self._dof = dof

    def load(self, config: dict[str, Any]) -> None:
        dof = 0
        links: list[dict[str, Any]] = config["link"]
        for link in links:
            if link["jointtype"] in ["revolute", "prismatic"]:
                dof += 1
        self._dof = dof


class Affetto(object):
    _config_path: Path
    _config: dict[str, Any]
    _name: str
    _chain: Chain

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            self.load_config_path(config_path)

    @property
    def config_path(self) -> Path:
        return self._config_path

    @final
    def load_config_path(self, config_path: str | Path):
        self._config_path = Path(config_path)
        with open(self._config_path, "rb") as f:
            c = tomllib.load(f)
        self.load_config(c)

    @property
    def config(self) -> dict[str, Any]:
        return self._config

    @abstractmethod
    def load_config(self, config_dict: dict[str, Any]) -> None:
        affetto_config = config_dict["affetto"]
        self._config = affetto_config
        self.load_name(affetto_config)
        self.load_chain(affetto_config)

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def load_name(self, config: dict[str, Any]) -> None:
        try:
            self._name = config["name"]
        except KeyError:
            self._name = "affetto"

    @property
    def chain(self) -> Chain:
        return self._chain

    @abstractmethod
    def load_chain(self, config: dict[str, Any]):
        try:
            self._chain = Chain(config["chain"])
        except KeyError:
            warnings.warn("'chain' field is not defined", UserWarning)

    @property
    def dof(self) -> int:
        try:
            return self._chain.dof
        except AttributeError:
            return 13
