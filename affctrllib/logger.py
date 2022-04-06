from collections.abc import Iterable
from pathlib import Path
from typing import Any


class Logger(object):
    _sep: str
    _eol: str
    _labels: list[str]
    _rawdata: list[list[Any]]
    _fpath: Path | None

    def __init__(
        self, fname: str | Path | None = None, sep: str = ",", eol: str = "\n"
    ) -> None:
        if fname is None:
            self._fpath = None
        else:
            self._fpath = Path(fname)
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

    @property
    def fpath(self) -> str:
        return str(self._fpath)

    def get_filename(self) -> str:
        return self.fpath

    def set_labels(self, labels: Iterable[str]) -> None:
        self._labels = list(labels)

    def extend_labels(self, labels: Iterable[str]) -> None:
        self._labels.extend(list(labels))

    def get_header(self) -> str:
        return self.sep.join(self._labels)

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

    def _dump_print(self, output=None) -> None:
        if len(self._labels):
            print(self.get_header(), end=self.eol, file=output)
        for line in self._rawdata:
            print(self.sep.join(f"{x}" for x in line), end=self.eol, file=output)

    def _dump_open(self, fpath: Path, mode: str) -> None:
        with open(fpath, mode=mode) as fobj:
            if len(self._labels):
                fobj.write(self.get_header() + self.eol)
            fobj.writelines(
                self.sep.join(f"{x}" for x in line) + self.eol for line in self._rawdata
            )

    def dump(self, fname: str | Path | None = None, overwrite=True, mode="w") -> None:
        fpath: Path
        if fname is not None:
            fpath = Path(fname)
        elif self._fpath is not None:
            fpath = self._fpath
        else:
            self._dump_print()
            return

        if not overwrite:
            fpath = self._generate_alternative_fname(fpath)
        self._dump_open(fpath, mode)
