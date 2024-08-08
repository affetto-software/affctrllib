#!/usr/bin/env python
# ruff: noqa: ANN001

from __future__ import annotations

import argparse
from pathlib import Path

from affctrllib.affmock import AffMock

DEFAULT_CONFIG_PATH = Path(__file__) / "config.toml"


def mainloop(config, freq, quiet) -> None:
    mock = AffMock(config)
    if freq is not None:
        mock.sensor_rate = freq
    mock.start(freq, quiet=quiet)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spawn mock Affetto.")
    parser.add_argument("-c", "--config", default=str(DEFAULT_CONFIG_PATH), help="config file")
    parser.add_argument(
        "-H",
        "--hz",
        dest="freq",
        type=float,
        help="frequency to send data",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="silence if specified")
    return parser.parse_args()


def main() -> None:
    args = parse()
    mainloop(args.config, args.freq, args.quiet)


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "env hz noqa usr"
# End:
