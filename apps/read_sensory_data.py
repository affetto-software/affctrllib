#!/usr/bin/env python

import argparse
import os

import affctrllib as acl
import numpy as np
from affctrllib import AffComm, Logger, Timer

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
DOF = 13
LABELS = ["t"]
for i in range(DOF):
    LABELS.extend([f"q{i}", f"pa{i}", f"pb{i}"])
BUFSIZE = 1024


def report_statistics(received_time_series):
    if len(received_time_series) == 0:
        return

    time_series = np.array(received_time_series)
    time_delta = time_series[1:] - time_series[:-1]  # type: ignore
    mean = np.mean(time_delta)
    var = np.var(time_delta)
    std = np.std(time_delta)
    print(f"Statistics:")
    print(f"  Number of collected samples: {len(received_time_series)}")
    print(f"     Mean of time differences: {mean:.5f}[s] ({mean*1000:.2f}[ms])")
    print(f"            Mean of frequency: {1.0 / mean:.2f}[Hz]")
    print(f"        Variance of time diff: {var:.4e}[s^2] ({var*1e6:.2e}[ms^2])")
    print(f"   Std deviation of time diff: {std:.6f}[s] ({std*1000:.3f}[ms])")


def mainloop(config, output):
    acom = AffComm(config)
    print(acom)

    ssock = acom.create_sensory_socket()
    logger = Logger(output)
    logger.set_labels(LABELS)

    received_time_series = []
    timer = Timer()
    timer.start()
    try:
        while True:
            t = timer.elapsed_time()
            recv_bytes, _ = ssock.recvfrom(BUFSIZE)
            sarr = acl.split_received_msg(recv_bytes, function=int)
            logger.store_data([t] + sarr)  # type: ignore
            received_time_series.append(t)

            print(f"\rt = {t:.2f}", end="")
            # timer.block()

    except KeyboardInterrupt:
        ssock.close()
        print()
        print(f"Saving data in <{str(output)}>...")
        logger.dump()
        report_statistics(received_time_series)


def parse():
    parser = argparse.ArgumentParser(description="Read sensory data forever.")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument("-H", "--hz", help="frequency to read data")
    parser.add_argument("-T", "--period", help="time [s] until program ends")
    return parser.parse_args()


def main():
    args = parse()
    mainloop(args.config, args.output)


if __name__ == "__main__":
    main()
