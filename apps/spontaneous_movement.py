#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
from affctrllib import AffComm, AffCtrl, Logger, Timer
from scipy.integrate import solve_ivp

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")


class MatsuokaOscillatorNetwork:
    n_neuron: int
    dt: float
    rise_time_const: np.ndarray
    adapt_time_const: np.ndarray
    self_inhibit: np.ndarray
    mutual_inhibit: np.ndarray
    tonic_input: np.ndarray
    sensory_feedback: np.ndarray
    x: np.ndarray
    t: float

    def __init__(
        self,
        n_neuron: int,
        dt: float,
        rise_time_const: np.ndarray,
        adapt_time_const: np.ndarray,
        self_inhibit: np.ndarray,
        mutual_inhibit: np.ndarray,
    ) -> None:
        self.set_n_neuron(n_neuron)
        self.set_dt(dt)
        self.set_rise_time_const(rise_time_const)
        self.set_adapt_time_const(adapt_time_const)
        self.set_self_inhibit(self_inhibit)
        self.set_mutual_inhibit(mutual_inhibit)
        self.set_tonic_input(np.zeros((self.n_neuron,)))
        self.set_sensory_feedback(np.zeros((self.n_neuron,)))
        self.x = np.zeros((self.n_neuron * 2))
        self.t = 0

    def set_n_neuron(self, n_neuron: int) -> None:
        assert n_neuron >= 2
        self.n_neuron = n_neuron

    def set_dt(self, dt: float) -> None:
        assert dt > 0
        self.dt = dt

    def set_rise_time_const(self, rise_time_const: np.ndarray) -> None:
        assert rise_time_const.shape == (self.n_neuron,)
        self.rise_time_const = rise_time_const

    def set_adapt_time_const(self, adapt_time_const: np.ndarray) -> None:
        assert adapt_time_const.shape == (self.n_neuron,)
        self.adapt_time_const = adapt_time_const

    def set_self_inhibit(self, self_inhibit: np.ndarray) -> None:
        assert self_inhibit.shape == (self.n_neuron,)
        self.self_inhibit = self_inhibit

    def set_mutual_inhibit(self, mutual_inhibit: np.ndarray) -> None:
        assert mutual_inhibit.shape == (self.n_neuron, self.n_neuron)
        self.mutual_inhibit = mutual_inhibit

    def set_tonic_input(self, tonic_input: np.ndarray) -> None:
        assert tonic_input.shape == (self.n_neuron,)
        self.tonic_input = tonic_input

    def set_sensory_feedback(self, sensory_feedback: np.ndarray) -> None:
        assert sensory_feedback.shape == (self.n_neuron,)
        self.sensory_feedback = sensory_feedback

    @property
    def u(self) -> np.ndarray:
        n = self.n_neuron
        return self.x[:n]

    @property
    def v(self) -> np.ndarray:
        n = self.n_neuron
        return self.x[n:]

    @property
    def y(self) -> np.ndarray:
        return np.maximum(self.u, 0)

    def set_u(self, u: np.ndarray) -> None:
        n = self.n_neuron
        assert u.shape == (n,)
        self.x[:n] = u

    def set_v(self, v: np.ndarray) -> None:
        n = self.n_neuron
        assert v.shape == (n,)
        self.x[n:] = v

    def dxdt(self, _, x):
        n = self.n_neuron
        u, v = x[:n], x[n:]
        y = np.maximum(u, 0)
        du = (
            -u
            - self.self_inhibit * v
            - self.mutual_inhibit.dot(y)
            + self.tonic_input
            + self.sensory_feedback
        ) / self.rise_time_const
        dv = (-v + y) / self.adapt_time_const
        return np.concatenate((du, dv))

    def update(
        self,
        tonic_input: np.ndarray,
        sensory_feedback: np.ndarray | None = None,
    ):
        self.set_tonic_input(tonic_input)
        if sensory_feedback is not None:
            self.set_sensory_feedback(sensory_feedback)
        solver = solve_ivp(self.dxdt, (self.t, self.t + self.dt), self.x)
        self.x = solver.y[:, -1]
        self.t += self.dt


def matsuoka_oscillator_test1(output: str):
    logger = Logger(output)
    logger.set_labels("t", "y1", "y2", "u1", "u2", "v1", "v2")
    Tr = np.array([1, 1])
    Ta = np.array([12, 12])
    b = np.array([2.5, 2.5])
    W = np.array(
        [
            [0, 1.5],
            [1.5, 0],
        ]
    )
    s = np.array([5, 5])
    T = 100
    nw = MatsuokaOscillatorNetwork(2, 0.01, Tr, Ta, b, W)
    nw.set_u(np.array([1, 0]))
    while nw.t < T:
        nw.update(s)
        logger.store(nw.t, nw.y, nw.u, nw.v)
    logger.dump()


def configure_nn() -> MatsuokaOscillatorNetwork:
    n = 26
    Tr = np.full((n,), 1)
    Ta = np.full((n,), 10)
    b = np.full((n,), 2.5)
    W = np.array(
        [
            [0, 1.5, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1.5, 0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]
    )


def ctrl_input(t: float) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((13,)), np.zeros((13,))


def create_logger(output: str) -> Logger:
    logger = Logger(output)
    logger.set_labels(
        "t",
        [f"ca{i}" for i in range(13)],
        [f"cb{i}" for i in range(13)],
    )
    return logger


def mainloop(
    config: str,
    output: str | None = None,
    cfreq: float | None = None,
):
    actrl = AffCtrl(config_path=config, freq=cfreq)
    acom = AffComm(config_path=config)
    acom.create_command_socket()
    timer = Timer(rate=actrl.freq)
    logger = None
    if output is not None:
        logger = create_logger(output)

    print("Type Ctrl-C to stop moving.")
    timer.start()
    try:
        while True:
            t = timer.elapsed_time()
            u1, u2 = ctrl_input(t)
            ca, cb = actrl.update(t, u1, u2)
            acom.send_commands(ca, cb)
            if logger is not None:
                logger.store(t, ca, cb)
            timer.block()
    finally:
        acom.close_command_socket()
        if logger is not None:
            logger.dump()


def parse():
    parser = argparse.ArgumentParser(description="Get Affetto to move spontaneously")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument(
        "-f",
        "--control-freq",
        dest="cfreq",
        type=float,
        help="control frequency",
    )
    return parser.parse_args()


def main():
    args = parse()
    # mainloop(args.config, args.output, args.cfreq)
    matsuoka_oscillator_test1(args.output)


if __name__ == "__main__":
    main()
