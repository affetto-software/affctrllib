import time
import warnings
from typing import Callable


class Timer(object):
    _rate: float | None
    _period: float | None
    _period_ns: int
    _time_started_ns: int
    _time_last_blocked_ns: int
    _time_ns_func: Callable[[], int]

    def __init__(self, rate: float | None = None, period: float | None = None) -> None:
        self._rate = None
        self._period = None
        self._period_ns = 0
        if rate is not None:
            self.rate = rate
        if period is not None:
            self.period = period
        self._time_started_ns = 0
        self._time_ns_func = time.time_ns

    @property
    def rate(self) -> float:
        if self._rate is None:
            raise ValueError("Timer: rate is not set")
        return self._rate

    def set_rate(self, rate: float) -> None:
        self._rate = rate
        self._period = 1.0 / rate
        self._period_ns = int(self._period * 1e9)

    @rate.setter
    def rate(self, rate: float) -> None:
        self.set_rate(rate)

    @property
    def period(self) -> float:
        if self._period is None:
            raise ValueError("Timer: period is not set")
        return self._period

    def set_period(self, period: float) -> None:
        self._period = period
        self._period_ns = int(self._period * 1e9)
        self._rate = 1.0 / period

    @period.setter
    def period(self, period: float) -> None:
        self.set_period(period)

    @property
    def period_ns(self) -> int:
        return self._period_ns

    def start(self) -> None:
        self._time_started_ns = self._time_ns_func()
        self._time_last_blocked_ns = self._time_started_ns

    def reset(self) -> None:
        self.start()

    def elapsed_time(self) -> float:
        return (self._time_ns_func() - self._time_started_ns) * 1e-9

    def elapsed_time_ns(self) -> int:
        return self._time_ns_func() - self._time_started_ns

    def sleep(self) -> None:
        time.sleep(self._period if self._period is not None else 0)

    def block(self) -> None:
        try:
            elapsed_since_last_blocked = self._time_ns_func() - self._time_last_blocked_ns
        except AttributeError:
            raise RuntimeError("Timer.start() must be called before Timer.block()")

        time_to_sleep = self._period_ns - elapsed_since_last_blocked
        if time_to_sleep > 0:
            time.sleep(time_to_sleep * 1e-9)
        else:
            msg = f"It took longer than specified period at t={self.elapsed_time()}"
            warnings.warn(msg, RuntimeWarning)
        self._time_last_blocked_ns = self._time_ns_func()
