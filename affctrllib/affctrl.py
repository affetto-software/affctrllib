import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any, Callable, Generic, TypeVar

import numpy as np

from .affcomm import AffComm
from .affetto import Affetto
from .affstate import AffStateThread
from .logger import Logger
from .timer import Timer

JointT = TypeVar("JointT", int, float, np.ndarray)


class Feedback(ABC, Generic[JointT]):
    _kP: JointT
    _kD: JointT
    _kI: JointT
    _accum_qerr: JointT

    def __init__(self, **kwargs) -> None:
        if "kP" in kwargs:
            self.kP = kwargs["kP"]
        if "kD" in kwargs:
            self.kD = kwargs["kD"]
        if "kI" in kwargs:
            self.kI = kwargs["kI"]

    @property
    def kP(self) -> JointT:
        return self._kP

    @kP.setter
    def kP(self, kP: JointT) -> None:
        self._kP = kP

    @property
    def kD(self) -> JointT:
        return self._kD

    @kD.setter
    def kD(self, kD: JointT) -> None:
        self._kD = kD

    @property
    def kI(self) -> JointT:
        return self._kI

    @kI.setter
    def kI(self, kI: JointT) -> None:
        self._kI = kI

    @property
    def stiff(self) -> JointT:
        return self._stiff

    @stiff.setter
    def stiff(self, stiff) -> None:
        self._stiff = stiff

    def positional_feedback(
        self,
        t: float,
        q: JointT,
        dq: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> JointT:
        _ = t  # avoid typing error
        qerr = qdes - q
        try:
            self._accum_qerr = self._accum_qerr + qerr
        except AttributeError:
            self._accum_qerr = qerr

        # NOTE: operator '*' in np.array does multiplications in
        # element-wise way.
        return self.kP * qerr + self.kD * (dqdes - dq) + self.kI * self._accum_qerr

    @abstractmethod
    def update(
        self,
        t: float,
        q: JointT,
        dq: JointT,
        pa: JointT,
        pb: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]:
        ...


class FeedbackPID(Feedback[JointT]):
    _stiff: JointT

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if "stiff" in kwargs:
            self.stiff = kwargs["stiff"]

    def update(
        self,
        t: float,
        q: JointT,
        pa: JointT,
        pb: JointT,
        dq: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]:
        _, _ = pa, pb  # avoid typing error
        e = self.positional_feedback(t, q, dq, qdes, dqdes)
        return (self.stiff + e, self.stiff - e)


class FeedbackPIDF(Feedback[JointT]):
    _stiff: JointT
    _press_gain: JointT

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if "stiff" in kwargs:
            self.stiff = kwargs["stiff"]
        if "press_gain" in kwargs:
            self.press_gain = kwargs["press_gain"]

    @property
    def press_gain(self) -> JointT:
        return self._press_gain

    @press_gain.setter
    def press_gain(self, press_gain) -> None:
        self._press_gain = press_gain

    def update(
        self,
        t: float,
        q: JointT,
        pa: JointT,
        pb: JointT,
        dq: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]:
        try:
            d = self.press_gain * (pa - pb)
        except AttributeError:
            d = pa - pb
        e = self.positional_feedback(t, q, dq, qdes, dqdes)
        return (self.stiff + e - d, self.stiff - e + d)


AFFCTRL_ACCEPTABLE_FEEDBACK_SCHEME_NAMES = {
    "PID": FeedbackPID,
    "pid": FeedbackPID,
    "PIDF": FeedbackPIDF,
    "pidf": FeedbackPIDF,
}

AFFCTRL_FEEDBACK_SCHEME_TO_CONFIG_KEY = {
    "FeedbackPID": "pid",
    "FeedbackPIDF": "pidf",
}


class AffCtrl(Affetto, Generic[JointT]):
    ctrl_config: dict[str, Any]
    _dt: float
    _freq: float
    _input_range: tuple[float, float]
    _scale_gain: float
    _inactive_joints: np.ndarray
    _feedback_scheme: Feedback

    DEFAULT_FREQ = 30
    DEFAULT_INACTIVE_PRESSURE = 0

    def __init__(
        self,
        config_path: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
    ) -> None:
        self.reset_inactive_joints()
        super().__init__(config_path)
        self.set_frequency(dt=dt, freq=freq)

        if not hasattr(self, "_freq"):
            self.set_freq(self.DEFAULT_FREQ)
            warnings.warn(
                f"Control frequency is not provided, set to default: {self._freq}"
            )

    def __repr__(self) -> str:
        return ""

    def __str__(self) -> str:
        return ""

    def set_frequency(self, dt: float | None = None, freq: float | None = None) -> None:
        if dt is not None and freq is not None:
            raise ValueError("Unable to specify DT and FREQ simultaneously")
        if dt is not None:
            self._dt = dt
            self._freq = 1.0 / dt
        elif freq is not None:
            self._freq = freq
            self._dt = 1.0 / freq

    @property
    def dt(self) -> float:
        return self._dt

    def set_dt(self, dt: float) -> None:
        self.set_frequency(dt=dt)

    @dt.setter
    def dt(self, dt: float) -> None:
        self.set_dt(dt)

    @property
    def freq(self) -> float:
        return self._freq

    def set_freq(self, freq: float) -> None:
        self.set_frequency(freq=freq)

    @freq.setter
    def freq(self, freq: float) -> None:
        self.set_freq(freq)

    @property
    def feedback_scheme(self) -> Feedback:
        return self._feedback_scheme

    def load_config(self, config: dict[str, Any]) -> None:
        super().load_config(config)
        self.load_ctrl_config()

    def load_ctrl_config(self, config: dict[str, Any] | None = None) -> None:
        if config is not None:
            c = config
        else:
            c = self.config
        self.ctrl_config = c["ctrl"]
        dt = self.ctrl_config.get("dt", None)
        freq = self.ctrl_config.get("freq", None)
        self.set_frequency(dt=dt, freq=freq)
        try:
            self.input_range = tuple(self.ctrl_config["input_range"])
        except KeyError:
            pass
        self.load_inactive_joints()
        try:
            scheme = self.ctrl_config["scheme"]
        except KeyError:
            scheme = "pid"
        self.load_feedback_scheme(scheme)

    def load_inactive_joints(self, ctrl_config: dict[str, Any] | None = None) -> None:
        if ctrl_config is None:
            ctrl_config = self.ctrl_config
        if "inactive_joints" in ctrl_config:
            for inactive_joint in ctrl_config["inactive_joints"]:
                index = inactive_joint["index"]
                try:
                    self.add_inactive_joints(index, inactive_joint["pressure"])
                except KeyError:
                    self.add_inactive_joints(index)

    @classmethod
    def _load_feedback_scheme_find_array(
        cls,
        ctrl_config: dict[str, Any],
        scheme_key: str,
        param_key: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            kwargs[param_key] = np.array(ctrl_config[scheme_key][param_key])
        except KeyError:
            pass
        return kwargs

    def load_feedback_scheme(
        self, scheme: str, ctrl_config: dict[str, Any] | None = None
    ) -> None:
        if ctrl_config is None:
            ctrl_config = self.ctrl_config
        scheme_class = AFFCTRL_ACCEPTABLE_FEEDBACK_SCHEME_NAMES[scheme]
        scheme_key = AFFCTRL_FEEDBACK_SCHEME_TO_CONFIG_KEY[scheme_class.__name__]
        kwargs = {}
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "kP", kwargs)
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "kD", kwargs)
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "kI", kwargs)
        self._load_feedback_scheme_find_array(ctrl_config, scheme_key, "stiff", kwargs)
        self._load_feedback_scheme_find_array(
            ctrl_config, scheme_key, "press_gain", kwargs
        )
        self._feedback_scheme = scheme_class(**kwargs)

    @property
    def input_range(self) -> tuple[float, float]:
        return self._input_range

    def set_input_range(self, input_range: tuple[float, float]) -> None:
        self._input_range = input_range
        self._scale_gain = 255.0 / (self._input_range[1] - self._input_range[0])

    @input_range.setter
    def input_range(self, input_range: tuple[float, float]) -> None:
        self.set_input_range(input_range)

    @property
    def scale_gain(self) -> float:
        return self._scale_gain

    def scale(self, u1: JointT, u2: JointT) -> tuple[JointT, JointT]:
        try:
            c1 = np.clip(u1, self._input_range[0], self._input_range[1])
            c2 = np.clip(u2, self._input_range[0], self._input_range[1])
            return (
                self._scale_gain * (c1 - self._input_range[0]),
                self._scale_gain * (c2 - self._input_range[0]),
            )
        except AttributeError:
            return (u1, u2)

    @property
    def inactive_joints(self) -> np.ndarray:
        return self._inactive_joints

    def _expand_as_index_range(self, pattern: str) -> list[int]:
        indices = []
        _range = pattern.split("-")
        if len(_range) > 1:
            try:
                beg = int(_range[0])
            except ValueError:
                beg = 0
            try:
                end = int(_range[1]) + 1
            except ValueError:
                end = self.dof
            indices.extend(range(beg, end))
        return indices

    def _expand_as_index(self, pattern: str) -> list[int]:
        index = []
        for p in pattern.split(","):
            if "-" in p:
                index.extend(self._expand_as_index_range(p))
            else:
                index.extend([int(p)])
        return index

    def _make_inactive_joints_array(
        self, pattern: int | str, pressure: float | None = None
    ) -> np.ndarray:
        if pressure is None:
            pressure = self.DEFAULT_INACTIVE_PRESSURE
        if isinstance(pattern, str):
            index = self._expand_as_index(pattern)
        else:
            index = [int(pattern)]
        arr = np.full((len(index), 3), pressure)
        arr[:, 0] = index
        return arr

    def set_inactive_joints(
        self,
        pattern: int | str,
        pressure: float | None = None,
    ) -> None:
        self._inactive_joints = self._make_inactive_joints_array(pattern, pressure)

    def add_inactive_joints(
        self,
        pattern: int | str,
        pressure: float | None = None,
    ) -> None:
        self._inactive_joints = np.append(
            self._inactive_joints,
            self._make_inactive_joints_array(pattern, pressure),
            axis=0,
        )

    def reset_inactive_joints(self) -> None:
        self._inactive_joints = np.empty((0, 3), dtype=float)

    def mask(self, u1: JointT, u2: JointT) -> tuple[JointT, JointT]:
        mask = self.inactive_joints[:, 0].astype(int)
        try:
            np.put(u1, mask, self.inactive_joints[:, 1])  # type: ignore
            np.put(u2, mask, self.inactive_joints[:, 2])  # type: ignore
        except TypeError:
            # Pass over TypeError, which will be raised when u1 and u2
            # are not instance of np.ndarray.
            pass
        return (u1, u2)

    def update(
        self,
        t: float,
        q: JointT,
        dq: JointT,
        pa: JointT,
        pb: JointT,
        qdes: JointT,
        dqdes: JointT,
    ) -> tuple[JointT, JointT]:
        u1, u2 = self.feedback_scheme.update(t, q, dq, pa, pb, qdes, dqdes)
        u1, u2 = self.mask(u1, u2)
        return self.scale(u1, u2)


class AffCtrlThread(Thread):
    _acom: AffComm
    _actrl: AffCtrl
    _astate: AffStateThread
    _lock: Lock
    _stopped: Event
    _timer: Timer
    _current_time: float
    _qdes_func: Callable[[float], np.ndarray]
    _dqdes_func: Callable[[float], np.ndarray]
    _logger: Logger

    def __init__(
        self,
        astate: AffStateThread | None = None,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
        output: str | Path | None = None,
        sensor_dt: float | None = None,
        sensor_freq: float | None = None,
    ):
        self._acom = AffComm(config)
        self._acom.create_command_socket()
        if astate is not None:
            self._astate = astate
        else:
            self._astate = self._create_state_estimator(
                config, dt=sensor_dt, freq=sensor_freq
            )
        self._actrl = AffCtrl(config, dt, freq)
        self._lock = Lock()
        self._stopped = Event()
        self._timer = Timer(rate=self._actrl.freq)
        self._current_time = 0
        self.reset_trajectory()
        if output:
            self._create_logger(output)
        Thread.__init__(self)

        self.acquire = self._lock.acquire
        self.release = self._lock.release

    def _create_state_estimator(
        self,
        config: str | Path | None = None,
        dt: float | None = None,
        freq: float | None = None,
    ) -> AffStateThread:
        return AffStateThread(config, dt=dt, freq=freq)

    def _create_logger(self, output: str | Path) -> Logger:
        self._logger = Logger(output)
        self._logger.set_labels(
            "t",
            # raw data
            [f"rq{i}" for i in range(self._actrl.dof)],
            [f"rdq{i}" for i in range(self._actrl.dof)],
            [f"rpa{i}" for i in range(self._actrl.dof)],
            [f"rpb{i}" for i in range(self._actrl.dof)],
            # estimated states
            [f"q{i}" for i in range(self._actrl.dof)],
            [f"dq{i}" for i in range(self._actrl.dof)],
            [f"pa{i}" for i in range(self._actrl.dof)],
            [f"pb{i}" for i in range(self._actrl.dof)],
            # command data
            [f"qdes{i}" for i in range(self._actrl.dof)],
            [f"dqdes{i}" for i in range(self._actrl.dof)],
            [f"ca{i}" for i in range(self._actrl.dof)],
            [f"cb{i}" for i in range(self._actrl.dof)],
        )
        return self._logger

    def run(self):
        # Since idling process may take several seconds to finish,
        # interaction with control thread should be started after
        # idling process has finished by using
        # AffCtrlThread.wait_for_idling().

        # Start state estimator thread if not started.
        if not self._astate.is_alive():
            if not self._astate.prepared():
                self._astate.prepare()
            self._astate.start()

        # Start timer.
        self._timer.start()

        # Start the main loop.
        while not self._stopped.is_set():
            t = self._timer.elapsed_time()
            rq, rdq, rpa, rpb = self._astate.get_raw_states()
            q, dq, pa, pb = self._astate.get_states()
            with self._lock:
                self._current_time = t
                qdes = self._qdes_func(t)
                dqdes = self._dqdes_func(t)
                ca, cb = self._actrl.update(t, q, dq, pa, pb, qdes, dqdes)
            self._acom.send_commands(ca, cb)
            try:
                self._logger.store(
                    t, rq, rdq, rpa, rpb, q, dq, pa, pb, qdes, dqdes, ca, cb
                )
            except AttributeError:
                pass
            self._timer.block()

        # Close socket after having left the loop.
        self._acom.close_command_socket()

    def join(self, timeout=None):
        self.stop()
        Thread.join(self, timeout)

    def stop(self) -> None:
        try:
            self._logger.dump()
        except AttributeError:
            pass
        self._stopped.set()

    def wait_for_idling(self, timeout=None) -> bool:
        return self._astate.wait_for_idling(timeout)

    @property
    def dof(self) -> int:
        with self._lock:
            return self._actrl.dof

    @property
    def current_time(self) -> float:
        with self._lock:
            return self._current_time

    @property
    def state(self) -> AffStateThread:
        return self._astate

    @property
    def dt(self) -> float:
        with self._lock:
            return self._actrl.dt

    @dt.setter
    def dt(self, dt: float) -> None:
        with self._lock:
            self._actrl.dt = dt

    @property
    def freq(self) -> float:
        with self._lock:
            return self._actrl.freq

    @freq.setter
    def freq(self, freq: float) -> None:
        with self._lock:
            self._actrl.freq = freq

    @property
    def feedback_scheme(self) -> Feedback:
        with self._lock:
            return self._actrl.feedback_scheme

    def load_feedback_scheme(
        self, scheme: str, ctrl_config: dict[str, Any] | None = None
    ) -> None:
        with self._lock:
            self._actrl.load_feedback_scheme(scheme, ctrl_config)

    @property
    def inactive_joints(self) -> np.ndarray:
        with self._lock:
            return np.copy(self._actrl.inactive_joints)

    def set_inactive_joints(
        self,
        pattern: int | str,
        pressure: float | None = None,
    ) -> None:
        with self._lock:
            self._actrl.set_inactive_joints(pattern, pressure)

    def add_inactive_joints(
        self,
        pattern: int | str,
        pressure: float | None = None,
    ) -> None:
        with self._lock:
            self._actrl.add_inactive_joints(pattern, pressure)

    def reset_inactive_joints(self) -> None:
        with self._lock:
            self._actrl.reset_inactive_joints()

    def set_qdes_func(self, qdes_func: Callable[[float], np.ndarray]) -> None:
        with self._lock:
            self._qdes_func = qdes_func

    def set_dqdes_func(self, dqdes_func: Callable[[float], np.ndarray]) -> None:
        with self._lock:
            self._dqdes_func = dqdes_func

    def set_trajectory(
        self,
        qdes_func: Callable[[float], np.ndarray],
        dqdes_func: Callable[[float], np.ndarray],
    ) -> None:
        with self._lock:
            self._qdes_func = qdes_func
            self._dqdes_func = dqdes_func

    def reset_trajectory(self) -> None:
        dof = self._actrl.dof
        self.set_trajectory(
            lambda _: np.zeros((dof,)),
            lambda _: np.zeros((dof,)),
        )
