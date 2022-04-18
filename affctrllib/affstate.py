from pathlib import Path
from typing import Any

import numpy as np

from .affcomm import reshape_array_for_unzip
from .affetto import Affetto
from .filter import Filter


class AffState(Affetto):
    state_config: dict[str, Any]
    _dt: float
    _freq: float
    _filter_list: list[Filter | None]
    _raw_data: list[float] | list[int] | np.ndarray
    _reshaped_data: np.ndarray
    _filtered_data: list[np.ndarray]
    _q_prev: np.ndarray
    _dq: np.ndarray

    def __init__(
        self,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
    ) -> None:
        super().__init__(config)

        if all(v is not None for v in (dt, freq)):
            raise ValueError("Unable to specify DT and FREQ simultaneously")

        if dt is not None:
            self.dt = dt
        elif freq is not None:
            self.freq = freq

        if not hasattr(self, "_dt") and not hasattr(self, "_freq"):
            raise ValueError("Require DT or FREQ")

        self._filter_list = [Filter(), Filter(), Filter()]

    def load_config(self, config: dict[str, Any]) -> None:
        super().load_config(config)
        self.load_state_config()

    def load_state_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.state_config = c["state"]
        if all(k in self.state_config for k in ("dt", "freq")):
            raise ValueError("Unable to specify DT and FREQ simultaneously")
        elif "dt" in self.state_config:
            self.dt = self.state_config["dt"]
        elif "freq" in self.state_config:
            self.freq = self.state_config["freq"]

    @property
    def dt(self) -> float:
        return self._dt

    def set_dt(self, dt: float) -> None:
        self._dt = dt
        self._freq = 1.0 / dt

    @dt.setter
    def dt(self, dt: float) -> None:
        self.set_dt(dt)

    @property
    def freq(self) -> float:
        return self._freq

    def set_freq(self, freq: float) -> None:
        self._freq = freq
        self._dt = 1.0 / freq

    @freq.setter
    def freq(self, freq: float) -> None:
        self.set_freq(freq)

    @property
    def raw_data(self) -> list[float] | list[int] | np.ndarray:
        return self._raw_data

    @property
    def raw_q(self) -> np.ndarray:
        return self._reshaped_data[0]

    @property
    def raw_pa(self) -> np.ndarray:
        return self._reshaped_data[1]

    @property
    def raw_pb(self) -> np.ndarray:
        return self._reshaped_data[2]

    @property
    def q(self) -> np.ndarray:
        return self._filtered_data[0]

    @property
    def pa(self) -> np.ndarray:
        return self._filtered_data[1]

    @property
    def pb(self) -> np.ndarray:
        return self._filtered_data[2]

    @property
    def dq(self) -> np.ndarray:
        return self._dq

    def update(self, raw_data: list[float] | list[int] | np.ndarray) -> None:
        self._raw_data = raw_data
        self._reshaped_data = reshape_array_for_unzip(self._raw_data, ncol=3)
        # Process input signal filtering.
        self._filtered_data = [
            f.update(d) if f is not None else d
            for f, d in zip(self._filter_list, self._reshaped_data)
        ]
        # Calculate time derivative of q.
        try:
            self._dq = (self.q - self._q_prev) / self.dt  # type: ignore
        except AttributeError:
            self._dq = np.zeros(shape=self.q.shape)
        self._q_prev = self.q
