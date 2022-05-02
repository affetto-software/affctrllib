from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np

JointT = TypeVar("JointT", int, float, np.ndarray)


class Profile(ABC, Generic[JointT]):
    _q0: JointT
    _qF: JointT
    _T: float
    _t0: float

    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float):
        self._q0 = q0
        self._qF = qF
        self._T = T
        self._t0 = t0

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

    @abstractmethod
    def s(self, t: float) -> JointT | float:
        ...

    @abstractmethod
    def ds(self, t: float) -> JointT | float:
        ...

    @abstractmethod
    def dds(self, t: float) -> JointT | float:
        ...

    def q(self, t: float) -> JointT | float:
        s = self.s(t)
        return self._q0 * (1.0 - s) + self._qF * s

    def dq(self, t: float) -> JointT | float:
        return (self._qF - self._q0) * self.ds(t)

    def ddq(self, t: float) -> JointT | float:
        return (self._qF - self._q0) * self.dds(t)


class TriangularVelocityProfile(Profile, Generic[JointT]):
    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float):
        Profile.__init__(self, q0, qF, T, t0)
        self._s_coeff = 2.0 / (T * T)
        self._ds_coeff = 4.0 / (T * T)

    def s(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._s_coeff * t_rel * t_rel
        elif t_rel <= self._T:
            return 1.0 - self._s_coeff * (self._T - t_rel) * (self._T - t_rel)
        else:
            return 1

    def ds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._ds_coeff * t_rel
        elif t_rel <= self._T:
            return self._ds_coeff * (self._T - t_rel)
        else:
            return 0

    def dds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel <= 0.5 * self._T:
            return self._ds_coeff
        elif t_rel <= self._T:
            return -self._ds_coeff
        else:
            return 0


class TrapezoidalVelocityProfile(Profile, Generic[JointT]):
    _vmax: JointT
    _tb: JointT
    _vM: JointT
    _a: JointT
    _zeros: JointT
    _ones: JointT

    def __init__(
        self,
        q0: JointT,
        qF: JointT,
        T: float,
        t0: float,
        vmax: JointT | float | None = None,
        tb: JointT | float | None = None,
    ):
        Profile.__init__(self, q0, qF, T, t0)
        if vmax is not None:
            self.set_vmax(vmax)
        elif tb is not None:
            self.set_tb(tb)
        if isinstance(vmax, np.ndarray):
            self._zeros = np.zeros(shape=vmax.shape)
            self._ones = np.ones(shape=vmax.shape)
        else:
            self._zeros = 0
            self._ones = 1

    @property
    def vmax(self) -> JointT:
        return self._vmax

    def set_vmax(self, vmax: JointT | float) -> None:
        if isinstance(self.q0, np.ndarray) and isinstance(vmax, (int, float)):
            self._vmax = np.full(self.q0.shape, vmax)
        else:
            self._vmax = vmax
        self._vM = np.absolute(self.vmax / (self.qF - self.q0))
        self._tb = self.T - 1.0 / self._vM
        self._a = self._vM / self.tb

    @property
    def tb(self) -> JointT:
        return self._tb

    def set_tb(self, tb: JointT | float) -> None:
        if isinstance(self.q0, np.ndarray) and isinstance(tb, (int, float)):
            self._tb = np.full(self.q0.shape, tb)
        else:
            self._tb = tb
        self._vM = 1.0 / (self.T - self.tb)
        self._vmax = self._vM * (self.qF - self.q0)
        self._a = self._vM / self.tb

    def s(self, t: float) -> np.ndarray | float:
        t_rel = t - self.t0
        if t_rel < 0:
            return self._zeros
        elif t_rel >= self.T:
            return self._ones
        return np.where(
            t_rel < self.tb,
            0.5 * self._a * t_rel * t_rel,
            np.where(
                (self.tb <= t_rel) & (t_rel < (self.T - self.tb)),
                self._vM * (t_rel - 0.5 * self.tb),
                np.where(
                    ((self.T - self.tb) <= t_rel) & (t_rel < self.T),
                    1.0 - 0.5 * self._a * (self.T - t_rel) * (self.T - t_rel),
                    self._zeros,
                ),
            ),
        )

    def ds(self, t: float) -> np.ndarray | float:
        t_rel = t - self.t0
        if t_rel < 0 or t_rel >= self.T:
            return self._zeros
        return np.where(
            t_rel < self.tb,
            self._a * t_rel,
            np.where(
                (self.tb <= t_rel) & (t_rel < (self.T - self.tb)),
                self._vM,
                np.where(
                    ((self.T - self.tb) <= t_rel) & (t_rel < self.T),
                    self._a * (self.T - t_rel),
                    self._zeros,
                ),
            ),
        )

    def dds(self, t: float) -> np.ndarray | float:
        t_rel = t - self.t0
        if t_rel < 0 or t_rel >= self.T:
            return self._zeros
        return np.where(
            t_rel < self.tb,
            self._a,
            np.where(
                (self.tb <= t_rel) & (t_rel < self.T - self.tb),
                self._zeros,
                np.where(
                    ((self.T - self.tb) <= t_rel) & (t_rel < self.T),
                    -self._a,
                    self._zeros,
                ),
            ),
        )


class FifthDegreePolynomialProfile(Profile, Generic[JointT]):
    def __init__(self, q0: JointT, qF: JointT, T: float, t0: float):
        Profile.__init__(self, q0, qF, T, t0)
        self._ds_coeff = 30.0 / T
        self._dds_coeff = 60.0 / (T * T)

    def s(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0:
            return 0
        elif t_rel > self._T:
            return 1
        t1 = t_rel / self._T
        t2 = t1 * t1
        t3 = t2 * t1
        return t3 * (6.0 * t2 - 15.0 * t1 + 10.0)

    def ds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        t1 = t_rel / self._T
        t2 = t1 * t1
        return t2 * (t1 - 1.0) * (t1 - 1.0) * self._ds_coeff

    def dds(self, t: float) -> float:
        t_rel = t - self._t0
        if t_rel < 0 or t_rel > self._T:
            return 0
        t1 = t_rel / self._T
        return t1 * (2.0 * t1 - 1.0) * (t1 - 1.0) * self._dds_coeff


PTP_ACCEPTABLE_PROFILE_NAMES = {
    "triangular velocity": TriangularVelocityProfile,
    "triangular": TriangularVelocityProfile,
    "tri": TriangularVelocityProfile,
    "trapezoidal velocity": TrapezoidalVelocityProfile,
    "trapezoidal": TrapezoidalVelocityProfile,
    "trapez": TrapezoidalVelocityProfile,
    "tra": TrapezoidalVelocityProfile,
    "5th-degree polynomial": FifthDegreePolynomialProfile,
    "5th degree polynomial": FifthDegreePolynomialProfile,
    "5th-degree": FifthDegreePolynomialProfile,
    "5th degree": FifthDegreePolynomialProfile,
    "5th": FifthDegreePolynomialProfile,
}


class PTP(Generic[JointT]):
    _profile: Profile

    def __init__(
        self,
        q0: JointT,
        qF: JointT,
        T: float,
        t0: float = 0,
        profile_name: str = "triangular",
        **kwargs,
    ) -> None:
        self.select_profile(q0, qF, T, t0, profile_name, **kwargs)

    @property
    def q0(self) -> JointT:
        return self.profile.q0

    @property
    def qF(self) -> JointT:
        return self.profile.qF

    @property
    def T(self) -> float:
        return self.profile.T

    @property
    def t0(self) -> float:
        return self.profile.t0

    @property
    def profile(self) -> Profile:
        return self._profile

    def select_profile(
        self, q0: JointT, qF: JointT, T: float, t0: float, profile_name: str, **kwargs
    ) -> None:
        if not profile_name in PTP_ACCEPTABLE_PROFILE_NAMES.keys():
            raise ValueError(f"Invalid profile name: {profile_name}")
        self._profile = PTP_ACCEPTABLE_PROFILE_NAMES[profile_name](
            q0, qF, T, t0, **kwargs
        )

    def q(self, t: float) -> JointT | float:
        return self.profile.q(t)

    def dq(self, t: float) -> JointT | float:
        return self.profile.dq(t)

    def ddq(self, t: float) -> JointT | float:
        return self.profile.ddq(t)
