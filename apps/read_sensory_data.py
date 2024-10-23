#!/usr/bin/env python
# ruff: noqa: ANN001,T201

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from affctrllib import AffComm, Logger, Timer

DEFAULT_CONFIG_PATH = Path(__file__) / "config.toml"
DOF = 13
LABELS = ["t"]
for i in range(DOF):
    LABELS.extend([f"q{i}", f"pa{i}", f"pb{i}"])
BUFSIZE = 1024


def report_statistics(received_time_series) -> None:
    if len(received_time_series) == 0:
        return

    time_series = np.array(received_time_series)
    time_delta = time_series[1:] - time_series[:-1]
    mean = np.mean(time_delta)
    var = np.var(time_delta)
    std = np.std(time_delta)
    print("Statistics:")
    print(f"  Number of collected samples: {len(received_time_series)}")
    print(f"     Mean of time differences: {mean:.5f}[s] ({mean*1000:.2f}[ms])")
    print(f"            Mean of frequency: {1.0 / mean:.2f}[Hz]")
    print(f"        Variance of time diff: {var:.4e}[s^2] ({var*1e6:.2e}[ms^2])")
    print(f"   Std deviation of time diff: {std:.6f}[s] ({std*1000:.3f}[ms])")


def mainloop(config, output, period) -> None:
    acom = AffComm(config)
    print(acom)
    if period == 0:
        print("To finish process, type Ctrl-C.")

    acom.create_sensory_socket()
    logger = Logger(output)
    logger.set_labels(LABELS)

    received_time_series: list[float] = []

    def cleanup() -> None:
        acom.close()
        logger.dump()
        report_statistics(received_time_series)

    timer = Timer()
    timer.start()
    t: float = 0
    try:
        while period == 0 or t < period:
            t = timer.elapsed_time()
            data: list = acom.receive_as_list()
            logger.store_data([t, *data])
            received_time_series.append(t)
            print(f"\rt = {t:.2f}", end="")
        print()

    except KeyboardInterrupt:
        print("\nFinishing process by KeyboardInterrupt.")
    cleanup()


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read sensory data forever.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="config file")
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument("-T", "--period", default=0, type=float, help="time [s] until program ends")
    return parser.parse_args()


def main() -> None:
    args = parse()
    mainloop(args.config, args.output, args.period)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "Ctrl env noqa pb usr"
# End:
