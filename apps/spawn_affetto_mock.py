#!/usr/bin/env python

import argparse
import os

from affctrllib.affmock import AffettoMock

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")


def mainloop(config, freq, quiet):
    mock = AffettoMock(config)
    if freq is not None:
        mock.sensor_rate = freq
    mock.start(freq, quiet)


def parse():
    parser = argparse.ArgumentParser(description="Spawn mock Affetto.")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument(
        "-H",
        "--hz",
        dest="freq",
        type=float,
        help="frequency to send data",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="silence if specified"
    )
    return parser.parse_args()


def main():
    args = parse()
    mainloop(args.config, args.freq, args.quiet)


if __name__ == "__main__":
    main()
