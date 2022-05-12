#!/usr/bin/env python

import argparse
import time
from pathlib import Path

import numpy as np
from affctrllib import PTP, AffCtrlThread, AffStateThread, Logger

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")

# Set printing options for numpy array.
np.set_printoptions(precision=1, linewidth=1000, suppress=True)


def activate_single_joint(
    actrl: AffCtrlThread,
    joint: int | None = None,
    inactive_pressure: float | None = None,
):
    pattern: str
    if joint is None:
        pattern = "-"
    else:
        if joint < 0 or actrl.dof <= joint:
            raise ValueError(f"joint index {joint} is out of bounds")
        bef = joint - 1
        aft = joint + 1
        pattern_list = []
        if bef == -1:
            pass
        elif bef == 0:
            pattern_list.append("0")
        else:
            pattern_list.append(f"0-{bef}")
        if aft == 13:
            pass
        elif aft == 12:
            pattern_list.append("12")
        else:
            pattern_list.append(f"{aft}-12")
        pattern = ",".join(pattern_list)
    actrl.reset_inactive_joints()
    actrl.set_inactive_joints(pattern, inactive_pressure)


class Trajectory:
    joint: int
    waypoints: list[float]
    intervals: list[float]
    q0: np.ndarray
    t0: float
    profile: str
    trajectories: list[PTP]
    waypoints_q: list[np.ndarray]
    passing_times: list[float]
    ptp_i: int

    def __init__(
        self,
        joint: int,
        waypoints: list[float],
        intervals: list[float],
        q0: np.ndarray,
        t0: float = 0,
        profile: str = "trapezoidal",
    ) -> None:
        self.joint = joint
        self.waypoints = waypoints
        self.intervals = intervals
        self.q0 = q0
        self.t0 = t0
        self.profile = profile

        self.create_trajectory()

    def create_trajectory(self) -> None:
        self.trajectories = []
        assert len(self.waypoints) == len(self.intervals)

        # Make all waypoints in joint space. Note that
        # len(self.waypoints_q) should be added one by
        # len(self.waypoints) since the initial position will be added
        # in front.
        self.waypoints_q = [self.q0.copy()]
        for wp in self.waypoints:
            q = self.q0.copy()
            q[self.joint] = wp
            self.waypoints_q.append(q)

        # Calculate time when trajectory passes a waypoint.
        self.passing_times = [
            sum(self.intervals[: i + 1]) + self.t0 for i in range(len(self.intervals))
        ]
        self.passing_times.insert(0, self.t0)

        # Create trajectory that passes all waypoints.
        for i in range(len(self.waypoints_q) - 1):
            ptp = PTP(
                self.waypoints_q[i],
                self.waypoints_q[i + 1],
                self.intervals[i],
                self.passing_times[i],
                self.profile,
            )
            self.trajectories.append(ptp)

        # Set the first PTP trajectory.
        self.ptp_i = 0

    def get_trajectory(self, t: float) -> PTP:
        try:
            if t >= self.passing_times[self.ptp_i + 1]:
                self.ptp_i += 1
        except AttributeError:
            raise RuntimeError("Trajectory.create_trajedtory() must be called")
        except IndexError:
            pass
        try:
            return self.trajectories[self.ptp_i]
        except IndexError:
            return self.trajectories[-1]

    def qdes(self, t: float) -> np.ndarray:
        return np.array(self.get_trajectory(t).q(t))

    def dqdes(self, t: float) -> np.ndarray:
        return np.array(self.get_trajectory(t).dq(t))


def check_trajectory(joint: int = 0, output: str | None = None):
    T = 12
    q0 = np.full((13,), 50)
    waypoints = [0, 100, q0[joint]]
    intervals = [T / 4, T / 2, T / 4]
    traj = Trajectory(joint, waypoints, intervals, q0)
    N = 1000
    logger = Logger(output) if output is not None else None
    if logger:
        logger.set_labels(
            ["t"], [f"qdes{i}" for i in range(13)], [f"dqdes{i}" for i in range(13)]
        )
    for i in range(N + 1):
        t = i * T / N
        qdes = traj.qdes(t)
        dqdes = traj.dqdes(t)
        if logger:
            logger.store([t], qdes, dqdes)
    if logger:
        logger.dump()


def mainloop(
    config: str,
    output: str | None = None,
    joint: int = 0,
    sfreq: float | None = None,
    cfreq: float | None = None,
    profile: str = "trapezoidal",
    inactive_pressure: float = 400,
):
    astate = AffStateThread(config=config, freq=sfreq)
    actrl = AffCtrlThread(astate, config=config, freq=cfreq, output=output)

    # Start AffStateThread.
    astate.prepare()  # idling
    astate.start()

    # Start AffCtrlThread.
    activate_single_joint(actrl, None, inactive_pressure)
    actrl.start()
    print("Waiting until robot gets stationary...")
    time.sleep(5)

    # Create trajectory.
    T = 20
    t0 = actrl.current_time
    q0 = astate.q
    waypoints = [0, 100, q0[joint]]
    intervals = [T / 4, T / 2, T / 4]
    traj = Trajectory(joint, waypoints, intervals, q0, t0, profile)

    print("Start moving!")
    activate_single_joint(actrl, joint, inactive_pressure)
    actrl.set_trajectory(traj.qdes, traj.dqdes)
    t = t0
    try:
        while t < t0 + T + 1:
            t = actrl.current_time
            q = astate.q
            print(f"\rt: {t:.2f}, q: {q}", end="")
            time.sleep(0.1)
        print()
    finally:
        print("Quitting...")
        actrl.join()
        astate.join()


def parse():
    parser = argparse.ArgumentParser(description="Let single joint move from 0 to 100")
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
        "-j", "--joint", default=0, type=int, help="Joint index to move"
    )
    parser.add_argument(
        "-p",
        "--profile",
        help="Point to Point interpolation profile",
    )
    return parser.parse_args()


def main():
    args = parse()
    # check_trajectory(args.joint, args.output)  # for debug
    mainloop(
        args.config,
        args.output,
        args.joint,
        args.sfreq,
        args.cfreq,
        args.profile,
    )


if __name__ == "__main__":
    main()
