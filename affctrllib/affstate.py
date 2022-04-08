import numpy as np
import numpy.typing as npt

from affctrllib.affcomm import reshape_array_for_unzip
from affctrllib.filter import Filter


class AffState(object):
    _dt: float
    _freq: float
    _filter_list: list[Filter | None]
    _raw_data: list[int]
    _reshaped_data: npt.ArrayLike
    _filtered_data: list[npt.ArrayLike]
    _q_prev: npt.ArrayLike
    _dq: npt.ArrayLike

    def __init__(self, dt: float | None = None, freq: float | None = None) -> None:
        if dt is not None and freq is not None:
            raise ValueError("Unable to specify DT and FREQ simultaneously")

        if dt is not None:
            self.dt = dt
        elif freq is not None:
            self.freq = freq
        else:
            raise ValueError("Require DT or FREQ")

        self._filter_list = [Filter(), Filter(), Filter()]
        self._q_prev = 0

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
    def raw_data(self) -> list[int]:
        return self._raw_data

    @property
    def q(self) -> npt.ArrayLike:
        return self._filtered_data[0]

    @property
    def pa(self) -> npt.ArrayLike:
        return self._filtered_data[1]

    @property
    def pb(self) -> npt.ArrayLike:
        return self._filtered_data[2]

    @property
    def dq(self) -> npt.ArrayLike:
        return self._dq

    def update(self, raw_data: list[int]) -> None:
        self._raw_data = raw_data
        self._reshaped_data = reshape_array_for_unzip(self._raw_data, ncol=3)
        # Process input signal filtering.
        self._filtered_data = [  # type: ignore
            self._filter_list[i].update(self._reshaped_data[i])  # type: ignore
            if isinstance(self._filter_list[i], Filter)
            else self._reshaped_data[i]  # type: ignore
            for i in range(3)
        ]
        # Calculate time derivative of q.
        self._dq = (self.q - self._q_prev) / self.dt  # type: ignore
        self._q_prev = self.q
