import itertools
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np

from .affcomm import AffComm, unzip_array_as_ndarray
from .affetto import Affetto
from .filter import Filter
from .timer import Timer


class AffState(Affetto):
    state_config: dict[str, Any]
    _dt: float
    _freq: float
    _filter_list: list[Filter | None]
    _raw_data: list[float] | list[int] | np.ndarray
    _data_ndarray: np.ndarray
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
        return self._data_ndarray[0]

    @property
    def raw_dq(self) -> np.ndarray:
        return self._dq

    @property
    def raw_pa(self) -> np.ndarray:
        return self._data_ndarray[1]

    @property
    def raw_pb(self) -> np.ndarray:
        return self._data_ndarray[2]

    @property
    def q(self) -> np.ndarray:
        return self._filtered_data[0]

    @property
    def dq(self) -> np.ndarray:
        return self._dq

    @property
    def pa(self) -> np.ndarray:
        return self._filtered_data[1]

    @property
    def pb(self) -> np.ndarray:
        return self._filtered_data[2]

    def update(self, raw_data: list[float] | list[int] | np.ndarray) -> None:
        self._raw_data = raw_data
        self._data_ndarray = unzip_array_as_ndarray(self._raw_data, ncol=3)
        # Process input signal filtering.
        self._filtered_data = [
            f.update(d) if f is not None else d
            for f, d in zip(self._filter_list, self._data_ndarray)
        ]
        # Calculate time derivative of q.
        try:
            self._dq = (self.q - self._q_prev) / self.dt  # type: ignore
        except AttributeError:
            self._dq = np.zeros(shape=self.q.shape)
        self._q_prev = self.q

    def idle(
        self,
        acom: AffComm,
        n_sample: int = 100,
        freq_tol: float = 1,
        no_error: bool = False,
        quiet: bool = False,
    ) -> None:
        spinner = itertools.cycle(["-", "/", "|", "\\"])
        if not quiet:
            sys.stdout.write("Idling sensory module... ")
            sys.stdout.flush()
        if not acom.sensory_socket.is_created():
            acom.create_sensory_socket()
        timer = Timer()
        received_time_series = []
        timer.start()
        for i in range(n_sample):
            sarr = acom.receive_as_list()
            received_time_series.append(timer.elapsed_time())
            self.update(sarr)
            if not quiet and i % 10 == 0:
                sys.stdout.write(next(spinner))
                sys.stdout.flush()
                sys.stdout.write("\b")
        time_series = np.array(received_time_series)
        dt_series = np.subtract(time_series[1:], time_series[:-1])
        estimated_freq = 1.0 / np.mean(dt_series)
        if not no_error and abs(self.freq - estimated_freq) > freq_tol:
            msg = f"Specified sampling frequency is probably incorrect:\n"
            msg += f"  {estimated_freq:.3f} (estimated) vs {self.freq:.3f} (specified)"
            raise RuntimeError(msg)
        if not quiet:
            sys.stdout.write("done.\n")


class AffStateThread(threading.Thread):
    _acom: AffComm
    _astate: AffState
    _lock: threading.Lock
    _stopped: threading.Event

    def __init__(
        self,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
    ) -> None:
        self._acom = AffComm(config)
        self._astate = AffState(config, dt, freq)
        self._lock = threading.Lock()
        self._stopped = threading.Event()
        threading.Thread.__init__(self)

    def prepare(
        self,
        n_sample: int = 100,
        freq_tol: float = 1,
        no_error: bool = False,
        quiet: bool = False,
    ) -> None:
        self._astate.idle(self._acom, n_sample, freq_tol, no_error, quiet)

    def run(self):
        while not self._stopped.is_set():
            sarr = self._acom.receive_as_list()
            with self._lock:
                self._astate.update(sarr)

    def join(self, timeout=None):
        self.stop()
        threading.Thread.join(self, timeout)

    def stop(self) -> None:
        self._acom.close_sensory_socket()
        self._stopped.set()

    def get_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        with self._lock:
            s = self._astate
            return s.q, s.dq, s.pa, s.pb

    @property
    def raw_data(self) -> list[float] | list[int] | np.ndarray:
        with self._lock:
            return self._astate.raw_data

    @property
    def raw_q(self) -> np.ndarray:
        with self._lock:
            return self._astate.raw_q

    @property
    def raw_dq(self) -> np.ndarray:
        with self._lock:
            return self._astate.raw_dq

    @property
    def raw_pa(self) -> np.ndarray:
        with self._lock:
            return self._astate.raw_pa

    @property
    def raw_pb(self) -> np.ndarray:
        with self._lock:
            return self._astate.raw_pb

    @property
    def q(self) -> np.ndarray:
        with self._lock:
            return self._astate.q

    @property
    def dq(self) -> np.ndarray:
        with self._lock:
            return self._astate.dq

    @property
    def pa(self) -> np.ndarray:
        with self._lock:
            return self._astate.pa

    @property
    def pb(self) -> np.ndarray:
        with self._lock:
            return self._astate.pb
