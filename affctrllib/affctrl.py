from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

from .affetto import Affetto

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
    _input_range: tuple[float, float]
    _scale_gain: float
    _inactive_joints: np.ndarray
    _feedback_scheme: Feedback

    DEFAULT_INACTIVE_PRESSURE = 0

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.reset_inactive_joints()
        super().__init__(config_path)

    def __repr__(self) -> str:
        return ""

    def __str__(self) -> str:
        return ""

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
        try:
            self.input_range = tuple(self.ctrl_config["input_range"])
        except KeyError:
            pass
        if "inactive_joints" in self.ctrl_config:
            for inactive_joint in self.ctrl_config["inactive_joints"]:
                index = inactive_joint["index"]
                try:
                    self.set_inactive_joint(index, inactive_joint["pressure"])
                except KeyError:
                    self.set_inactive_joint(index)
        scheme = self.ctrl_config["scheme"]
        scheme_class = AFFCTRL_ACCEPTABLE_FEEDBACK_SCHEME_NAMES[scheme]
        scheme_key = AFFCTRL_FEEDBACK_SCHEME_TO_CONFIG_KEY[scheme_class.__name__]
        kP = np.array(self.ctrl_config[scheme_key]["kP"])
        kD = np.array(self.ctrl_config[scheme_key]["kD"])
        kI = np.array(self.ctrl_config[scheme_key]["kI"])
        stiff = np.array(self.ctrl_config[scheme_key]["stiff"])
        self._feedback_scheme = scheme_class(kP=kP, kD=kD, kI=kI, stiff=stiff)

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
        indices = []
        for p in pattern.split(","):
            if "-" in p:
                indices.extend(self._expand_as_index_range(p))
            else:
                indices.extend([int(p)])
        return indices

    def set_inactive_joint(
        self,
        index: int | str,
        pressure: float | None = None,
    ) -> None:
        if pressure is None:
            pressure = self.DEFAULT_INACTIVE_PRESSURE
        if isinstance(index, str):
            indices = self._expand_as_index(index)
        else:
            indices = [int(index)]
        arr = np.full((len(indices), 3), pressure)
        arr[:, 0] = indices
        self._inactive_joints = np.append(self._inactive_joints, arr, axis=0)

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
