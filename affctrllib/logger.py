from collections.abc import Iterable
from pathlib import Path
from typing import Any


class Logger(object):
    _sep: str
    _eol: str
    _labels: list[str]
    _rawdata: list[list[Any]]

    def __init__(self, sep: str = ",", eol: str = "\n") -> None:
        self._sep = sep
        self._eol = eol
        self._labels = []
        self._rawdata = []

    @property
    def sep(self) -> str:
        return self._sep

    @property
    def eol(self) -> str:
        return self._eol

    def set_labels(self, labels: Iterable[str]) -> None:
        self._labels = list(labels)

    def extend_labels(self, labels: Iterable[str]) -> None:
        self._labels.extend(list(labels))

    def get_header(self) -> str | None:
        if len(self._labels):
            return self.sep.join(self._labels)
        else:
            return None

    def store_data(self, data: Iterable[Any]) -> None:
        self._rawdata.append(list(data))

    def extend_data(self, data: Iterable[Any]) -> None:
        try:
            self._rawdata[-1].extend(list(data))
        except IndexError:
            self.store_data(data)

    def get_data(self) -> list[list[Any]]:
        return self._rawdata

    def _generate_alternative_fname(self, basefn: Path) -> Path:
        stem = basefn.stem
        suffix = basefn.suffix
        cnt = 1
        while True:
            cand = basefn.parent / Path(f"{stem}.{cnt}{suffix}")
            if not cand.exists():
                return cand
            cnt += 1

    def dump(self, fname: str | Path, overwrite=True, mode="w") -> None:
        if overwrite:
            p = Path(fname)
        else:
            p = self._generate_alternative_fname(Path(fname))
        with open(p, mode=mode) as f:
            if len(self._labels):
                f.write(self.sep.join(self._labels) + self.eol)
            f.writelines(
                self.sep.join(f"{x}" for x in row) + self.eol for row in self._rawdata
            )
