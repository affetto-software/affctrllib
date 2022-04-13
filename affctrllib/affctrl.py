from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, TypeVar

import numpy as np
import tomli

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


class AffCtrl(Generic[JointT]):
    _dof: int
    _config_path: Path | None
    _feedback_scheme: Feedback

    def __init__(self, config_path: str | Path | None = None) -> None:
        if config_path is not None:
            self.load_config(config_path)
        else:
            self._config_path = None
        self._dof = 13

    def __repr__(self) -> str:
        return ""

    def __str__(self) -> str:
        return ""

    @property
    def config_path(self) -> str | None:
        if self._config_path is not None:
            return str(self._config_path)
        else:
            return None

    @property
    def feedback_scheme(self) -> Feedback:
        return self._feedback_scheme

    def load_config(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        with open(self._config_path, "rb") as f:
            config_dict = tomli.load(f)

        ctrl_dict = config_dict["affetto"]["ctrl"]
        scheme = ctrl_dict["scheme"]
        scheme_class = AFFCTRL_ACCEPTABLE_FEEDBACK_SCHEME_NAMES[scheme]
        scheme_key = AFFCTRL_FEEDBACK_SCHEME_TO_CONFIG_KEY[scheme_class.__name__]
        kP = np.array(ctrl_dict[scheme_key]["kP"])
        kD = np.array(ctrl_dict[scheme_key]["kD"])
        kI = np.array(ctrl_dict[scheme_key]["kI"])
        stiff = np.array(ctrl_dict[scheme_key]["stiff"])
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
