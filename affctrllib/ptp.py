from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

JointT = TypeVar("JointT", int, float, np.ndarray)


class Profile(ABC, Generic[JointT]):
    def __init__(self, q0, qF, T, t0):
        self._q0 = q0
        self._qF = qF
        self._T = T
        self._t0 = t0
        try:
            self._zeros = np.zeros(shape=q0.shape)
        except AttributeError:
            self._zeros = 0

    @abstractmethod
    def s(self, t) -> JointT:
        ...

    @abstractmethod
    def ds(self, t) -> JointT:
        ...

    @abstractmethod
    def dds(self, t) -> JointT:
        ...

    def q(self, t) -> JointT:
        s = self.s(t)
        return self._q0 * (1.0 - s) + self._qF * s

    def dq(self, t) -> JointT:
        return (self._qF - self._q0) * self.ds(t)

    def ddq(self, t) -> JointT:
        return (self._qF - self._q0) * self.dds(t)


class TriangularVelocityProfile(Profile):
    def __init__(self, q0, qF, T, t0):
        Profile.__init__(self, q0, qF, T, t0)
        self._s_coeff = 2.0 / (T * T)
        self._ds_coeff = 4.0 / (T * T)

    def s(self, t):
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._s_coeff * t_rel * t_rel
        elif t_rel <= self._T:
            return 1.0 - self._s_coeff * (self._T - t_rel) * (self._T - t_rel)
        else:
            return 1

    def ds(self, t):
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._ds_coeff * t_rel
        elif t_rel <= self._T:
            return self._ds_coeff * (self._T - t_rel)
        else:
            return 0

    def dds(self, t):
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._ds_coeff
        elif t_rel <= self._T:
            return -self._ds_coeff
        else:
            return 0


class FifthDegreePolynomialProfile(Profile):
    def __init__(self, q0, qF, T, t0):
        Profile.__init__(self, q0, qF, T, t0)
        self._ds_coeff = 30.0 / T
        self._dds_coeff = 60.0 / (T * T)

    def s(self, t):
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel > self._T:
            return 1
        t1 = t_rel / self._T
        t2 = t1 * t1
        t3 = t2 * t1
        return t3 * (6.0 * t2 - 15.0 * t1 + 10.0)

    def ds(self, t):
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        t1 = t_rel / self._T
        t2 = t1 * t1
        return t2 * (t1 - 1.0) * (t1 - 1.0) * self._ds_coeff

    def dds(self, t):
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        t1 = t_rel / self._T
        return t1 * (2.0 * t1 - 1.0) * (t1 - 1.0) * self._dds_coeff


PTP_ACCEPTABLE_PROFILE_NAMES = {
    "triangular velocity": TriangularVelocityProfile,
    "triangular": TriangularVelocityProfile,
    "tri": TriangularVelocityProfile,
    "5th-degree polynomial": FifthDegreePolynomialProfile,
    "5th degree polynomial": FifthDegreePolynomialProfile,
    "5th-degree": FifthDegreePolynomialProfile,
    "5th degree": FifthDegreePolynomialProfile,
    "5th": FifthDegreePolynomialProfile,
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
