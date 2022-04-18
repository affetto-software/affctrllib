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
    _feedback_scheme: Feedback

    def __init__(self, config_path: str | Path | None = None) -> None:
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
        scheme = self.ctrl_config["scheme"]
        scheme_class = AFFCTRL_ACCEPTABLE_FEEDBACK_SCHEME_NAMES[scheme]
        scheme_key = AFFCTRL_FEEDBACK_SCHEME_TO_CONFIG_KEY[scheme_class.__name__]
        kP = np.array(self.ctrl_config[scheme_key]["kP"])
        kD = np.array(self.ctrl_config[scheme_key]["kD"])
        kI = np.array(self.ctrl_config[scheme_key]["kI"])
        stiff = np.array(self.ctrl_config[scheme_key]["stiff"])
        self._feedback_scheme = scheme_class(kP=kP, kD=kD, kI=kI, stiff=stiff)

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
        return self.feedback_scheme.update(t, q, dq, pa, pb, qdes, dqdes)
