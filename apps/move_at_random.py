#!/usr/bin/env python

import argparse
import random
import time
from pathlib import Path

import numpy as np
from affctrllib import PTP, AffPosCtrlThread, Logger

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")

DEFAULT_UPDATE_T_RANGE = (0.5, 1.5)
DEFAULT_UPDATE_Q_RANGE = (20.0, 40.0)
DEFAULT_Q_LIMIT = (5.0, 95.0)

# Set printing options for numpy array.
np.set_printoptions(precision=1, linewidth=1000, suppress=True)


class RandomTrajectory:
    joints: list[int]
    q0: np.ndarray
    t0: float
    update_t_range: tuple[float, float]
    update_q_range: tuple[float, float]
    q_limit: tuple[float, float]
    q_limits: list[tuple[float, float]]
    ptp_list: list[PTP]
    profile: str

    def __init__(
        self,
        joints: list[int],
        q0: np.ndarray,
        t0: float,
        update_t_range: tuple[float, float],
        update_q_range: tuple[float, float],
        q_limit: tuple[float, float],
        profile: str = "trapezoidal",
        seed: int | None = None,
    ):
        self.joints = joints
        self.q0 = q0.copy()
        self.t0 = t0
        self.update_t_range = update_t_range
        self.update_q_range = update_q_range
        self.q_limit = q_limit
        self.q_limits = [self.q_limit for _ in self.joints]
        for i, j in enumerate(self.joints):
            if j == 0:
                # Reduce limits of waist joint.
                self.q_limits[i] = (40.0, 60.0)
        self.profile = profile
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.ptp_list = self.initialize_ptp()

    def _get_new_T_qdes(
        self,
        q0: float,
        t_range: tuple[float, float],
        q_range: tuple[float, float],
        q_limit: tuple[float, float],
    ) -> tuple[float, float]:
        qmin, qmax = min(q_limit), max(q_limit)
        qdes = q0
        T = random.uniform(min(t_range), max(t_range))
        ok = False
        while not ok:
            q_diff = random.uniform(min(q_range), max(q_range))
            qdes = random.choice([-1, 1]) * q_diff + q0
            if qdes < qmin:
                qdes = qmin + (qmin - qdes)
            elif qdes > qmax:
                qdes = qmax - (qdes - qmax)
            qdes = max(min(qmax, qdes), qmin)
            if abs(qdes - q0) > 0.0001:
                ok = True
        return T, qdes

    def initialize_ptp(self) -> list[PTP]:
        ptp_list: list[PTP] = []
        for i in self.joints:
            T, qdes = self._get_new_T_qdes(
                self.q0[i], self.update_t_range, self.update_q_range, self.q_limits[i]
            )
            ptp_list.append(
                PTP(self.q0[i], qdes, T, self.t0, profile_name=self.profile)
            )
        return ptp_list

    def update_ptp(self, t: float) -> None:
        for i, ptp in enumerate(self.ptp_list):
            if ptp.t0 + ptp.T < t:
                new_t0 = ptp.t0 + ptp.T
                new_q0 = ptp.qF
                new_T, new_qdes = self._get_new_T_qdes(
                    new_q0, self.update_t_range, self.update_q_range, self.q_limits[i]
                )
                new_ptp = PTP(
                    new_q0, new_qdes, new_T, new_t0, profile_name=self.profile
                )
                self.ptp_list[i] = new_ptp

    def qdes(self, t: float) -> np.ndarray:
        self.update_ptp(t)
        qdes = self.q0.copy()
        qdes[self.joints] = [ptp.q(t) for ptp in self.ptp_list]
        return qdes

    def dqdes(self, t: float) -> np.ndarray:
        self.update_ptp(t)
        dqdes = np.zeros(self.q0.shape)
        dqdes[self.joints] = [ptp.dq(t) for ptp in self.ptp_list]
        return dqdes


def check_trajectory(
    joints: list[int],
    t_range: tuple[float, float] = DEFAULT_UPDATE_T_RANGE,
    q_range: tuple[float, float] = DEFAULT_UPDATE_Q_RANGE,
    q_limit: tuple[float, float] = DEFAULT_Q_LIMIT,
    output: str | None = None,
):
    T = 12
    q0 = np.full((13,), 50, dtype=float)
    traj = RandomTrajectory(joints, q0, 0, t_range, q_range, q_limit)
    N = 1000
    logger = Logger(output) if output is not None else None
    if logger is not None:
        logger.set_labels(
            ["t"], [f"q{i}" for i in range(13)], [f"dq{i}" for i in range(13)]
        )
    for i in range(N + 1):
        t = i * T / N
        qdes = traj.qdes(t)
        dqdes = traj.dqdes(t)
        if logger is not None:
            logger.store([t], qdes, dqdes)
    if logger is not None:
        logger.dump()


def mainloop(
    config: str,
    joints: list[int] | None,
    t_range: tuple[float, float] = DEFAULT_UPDATE_T_RANGE,
    q_range: tuple[float, float] = DEFAULT_UPDATE_Q_RANGE,
    q_limit: tuple[float, float] = DEFAULT_Q_LIMIT,
    sfreq: float | None = None,
    cfreq: float | None = None,
    profile: str = "trapezoidal",
    duration: float | None = None,
    output: str | None = None,
    inactive_pressure: float = 400,
):
    # Start AffPosCtrlThread.
    actrl = AffPosCtrlThread(
        config=config, freq=cfreq, sensor_freq=sfreq, output=output
    )
    actrl.set_active_joints(None, inactive_pressure)
    actrl.start()
    actrl.wait_for_idling()
    print("Waiting until robot gets stationary...")
    time.sleep(5)

    # Create trajectory.
    T = duration if duration is not None else 24 * 60 * 60
    t0 = actrl.current_time
    q0 = actrl.state.q
    if joints is None:
        joints = list(range(actrl.dof))
    traj = RandomTrajectory(joints, q0, t0, t_range, q_range, q_limit, profile)

    print("Start moving!")
    actrl.set_active_joints(joints, inactive_pressure)
    actrl.set_trajectory(traj.qdes, traj.dqdes)
    t = t0
    try:
        while t < t0 + T + 1:
            t = actrl.current_time
            q = actrl.state.q
            print(f"\rt: {t:.2f}, q: {q}", end="")
            time.sleep(0.1)
        print()
    finally:
        print("Quitting...")
        actrl.join()


def parse():
    parser = argparse.ArgumentParser(description="Get joints to move at random.")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument(
        "-F",
        "--sensor-freq",
        dest="sfreq",
        type=float,
        help="sensor frequency",
    )
    parser.add_argument(
        "-f",
        "--control-freq",
        dest="cfreq",
        type=float,
        help="control frequency",
    )
    parser.add_argument(
        "-j", "--joints", nargs="+", type=int, help="Joint index to move"
    )
    parser.add_argument(
        "-t",
        "--t-range",
        default=DEFAULT_UPDATE_T_RANGE,
        nargs="+",
        type=float,
        help="Time range to be updated",
    )
    parser.add_argument(
        "-q",
        "--q-range",
        default=DEFAULT_UPDATE_Q_RANGE,
        nargs="+",
        type=float,
        help="Angle range to be updated",
    )
    parser.add_argument(
        "-Q",
        "--q-limit",
        default=DEFAULT_Q_LIMIT,
        nargs="+",
        type=float,
        help="Angle limits",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default="trapezoidal",
        help="Point to Point interpolation profile",
    )
    parser.add_argument(
        "-T",
        "--duration",
        type=float,
        help="Total time duration",
    )
    return parser.parse_args()


def main():
    args = parse()
    # check_trajectory(
    #     args.joints,
    #     t_range=tuple(args.t_range),
    #     q_range=tuple(args.q_range),
    #     q_limit=tuple(args.q_limit),
    #     output=args.output,
    # )  # for debug
    mainloop(
        args.config,
        args.joints,
        t_range=tuple(args.t_range),
        q_range=tuple(args.q_range),
        q_limit=tuple(args.q_limit),
        sfreq=args.sfreq,
        cfreq=args.cfreq,
        profile=args.profile,
        duration=args.duration,
        output=args.output,
    )


if __name__ == "__main__":
    main()
