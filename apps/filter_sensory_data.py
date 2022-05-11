#!/usr/bin/env python

import argparse
import os

import affctrllib as acl
import numpy as np
from affctrllib import AffComm, AffState, Logger, Timer

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
DOF = 13
LABELS = ["t"]
# raw data
LABELS.extend([f"rq{i}" for i in range(DOF)])
LABELS.extend([f"rpa{i}" for i in range(DOF)])
LABELS.extend([f"rpb{i}" for i in range(DOF)])
# filtered data
LABELS.extend([f"q{i}" for i in range(DOF)])
LABELS.extend([f"dq{i}" for i in range(DOF)])
LABELS.extend([f"pa{i}" for i in range(DOF)])
LABELS.extend([f"pb{i}" for i in range(DOF)])
BUFSIZE = 1024


def report_statistics(received_time_series, freq):
    if len(received_time_series) == 0:
        return

    time_series = np.array(received_time_series)
    time_delta = time_series[1:] - time_series[:-1]  # type: ignore
    mean = np.mean(time_delta)
    var = np.var(time_delta)
    std = np.std(time_delta)
    print(f"Statistics:")
    print(f"            Assumed frequency: {freq}")
    print(f"  Number of collected samples: {len(received_time_series)}")
    print(f"     Mean of time differences: {mean:.5f}[s] ({mean*1000:.2f}[ms])")
    print(f"            Mean of frequency: {1.0 / mean:.2f}[Hz]")
    print(f"        Variance of time diff: {var:.4e}[s^2] ({var*1e6:.2e}[ms^2])")
    print(f"   Std deviation of time diff: {std:.6f}[s] ({std*1000:.3f}[ms])")


def logging(logger, t, astate):
    logger.store_data([t])
    logger.extend_data(astate.raw_q)
    logger.extend_data(astate.raw_pa)
    logger.extend_data(astate.raw_pb)
    logger.extend_data(astate.q)
    logger.extend_data(astate.dq)
    logger.extend_data(astate.pa)
    logger.extend_data(astate.pb)


def mainloop(config, output, freq, period):
    acom = AffComm(config)
    print(acom)
    if period == 0:
        print(f"To finish process, type Ctrl-C.")

    astate = AffState(config)
    if freq > 0:
        astate.freq = freq
    logger = Logger(output)
    logger.set_labels(LABELS)

    received_time_series = []

    def cleanup():
        acom.close()
        logger.dump()
        report_statistics(received_time_series, astate.freq)

    astate.idle(acom)
    timer = Timer(rate=astate.freq)
    timer.start()
    t = 0
    try:
        while period == 0 or t < period:
            t = timer.elapsed_time()
            data = acom.receive_as_list()
            astate.update(data)
            logging(logger, t, astate)
            received_time_series.append(t)
            print(f"\rt = {t:.2f}", end="")
        print()

    except KeyboardInterrupt:
        print(f"\nFinishing process by KeyboardInterrupt.")
    cleanup()


def parse():
    parser = argparse.ArgumentParser(description="Read sensory data forever.")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument(
        "-H", "--hz", dest="freq", default=0, type=float, help="frequency to read data"
    )
    parser.add_argument(
        "-T", "--period", default=0, type=float, help="time [s] until program ends"
    )
    return parser.parse_args()


def main():
    args = parse()
    mainloop(args.config, args.output, args.freq, args.period)


if __name__ == "__main__":
    main()
