from abc import ABC, abstractmethod
from typing import Generic, TypeVar

JointT = TypeVar("JointT")


class Profile(ABC):
    def __init__(self, q0, qF, T, t0):
        self._q0 = q0
        self._qF = qF
        self._T = T
        self._t0 = t0
        self._tF = t0 + T

    @abstractmethod
    def q(self, t):
        ...

    @abstractmethod
    def dq(self, t):
        ...


class TriangularVelocityProfile(Profile):
    def __init__(self, q0, qF, T, t0):
        Profile.__init__(self, q0, qF, T, t0)
        self._midpoint = t0 + 0.5 * T
        self._ascend_q_coeff = (2.0 / (T * T)) * (qF - q0)
        self._descend_q_coeff = (2.0 / (T * T)) * (q0 - qF)
        self._ascend_dq_coeff = 2.0 * self._ascend_q_coeff
        self._descend_dq_coeff = 2.0 * self._descend_q_coeff

    def q(self, t):
        if t < self._t0:
            return self._q0
        elif t <= self._midpoint:
            return self._ascend_q_coeff * (t - self._t0) * (t - self._t0) + self._q0
        elif t <= self._tF:
            return self._descend_q_coeff * (t - self._tF) * (t - self._tF) + self._qF
        else:
            return self._qF

    def dq(self, t):
        if t < self._t0:
            return 0
        elif t <= self._midpoint:
            return self._ascend_dq_coeff * (t - self._t0)
        elif t <= self._tF:
            return self._descend_dq_coeff * (t - self._tF)
        else:
            return 0


class FifthOrderPolynomialProfile(Profile, Generic[JointT]):
    def __init__(self, q0, qF, T, t0):
        Profile.__init__(self, q0, qF, T, t0)
        self._q_coeff = qF - q0
        self._dq_coeff = (30.0 / T) * (qF - q0)

    def q(self, t):
        if t < self._t0:
            return self._q0
        elif t > self._tF:
            return self._qF
        t1 = (t - self._t0) / self._T
        t2 = t1 * t1
        t3 = t2 * t1
        return t3 * (6.0 * t2 - 15.0 * t1 + 10.0) * self._q_coeff + self._q0

    def dq(self, t):
        if t < self._t0 or t > self._tF:
            return 0
        t1 = (t - self._t0) / self._T
        t2 = t1 * t1
        return t2 * (t1 - 1) * (t1 - 1) * self._dq_coeff


PTP_ACCEPTABLE_PROFILE_NAMES = {
    "triangular velocity": TriangularVelocityProfile,
    "triangular": TriangularVelocityProfile,
    "tri": TriangularVelocityProfile,
    "5th-order polynomial": FifthOrderPolynomialProfile,
    "5th order polynomial": FifthOrderPolynomialProfile,
    "5th-order": FifthOrderPolynomialProfile,
    "5th order": FifthOrderPolynomialProfile,
    "5th": FifthOrderPolynomialProfile,
}


class PTP(Generic[JointT]):
    _q0: JointT
    _qF: JointT
    _T: float
    _t0: float
    _profile: Profile

    def __init__(
        self,
        q0: JointT,
        qF: JointT,
        T: float,
        t0: float = 0,
        profile_name: str = "triangular",
    ) -> None:
        self._q0 = q0
        self._qF = qF
        self._T = T
        self._t0 = t0
        self.select_profile(profile_name)

    @property
    def q0(self) -> JointT:
        return self._q0

    @property
    def qF(self) -> JointT:
        return self._qF

    @property
    def T(self) -> float:
        return self._T

    @property
    def t0(self) -> float:
        return self._t0

    @property
    def profile(self) -> Profile:
        return self._profile

    def select_profile(self, profile_name: str) -> None:
        if not profile_name in PTP_ACCEPTABLE_PROFILE_NAMES.keys():
            raise ValueError(f"Invalid profile name: {profile_name}")
        self._profile = PTP_ACCEPTABLE_PROFILE_NAMES[profile_name](
            self.q0, self.qF, self.T, self.t0
        )

    def q(self, t: float):
        return self.profile.q(t)

    def dq(self, t: float):
        return self.profile.dq(t)
