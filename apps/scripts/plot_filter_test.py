#!/usr/bin/env python
# ruff: noqa: ANN001,ANN003,PLR2004,T201

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from pyplotutil.datautil import Data

sfparam_tmpl = {
    "savefig": False,
    "basedir": "fig",
    "filename": None,
    "extensions": ["png"],
}


def savefig(fig, **sfparam) -> None:
    if not sfparam.get("savefig", False):
        return
    if "filename" not in sfparam:
        msg = "filename is required in sfparam."
        raise KeyError(msg)

    # Join basedir and filename.
    path = Path(sfparam.get("basedir", "fig")) / Path(sfparam["filename"])
    # Create directories if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save figures in specified formats.
    for ext in sfparam.get("extensions", ["png"]):
        if not ext.startswith("."):
            ext = f".{ext}"  # noqa: PLW2901
        fname = path.with_suffix(ext)
        fig.savefig(str(fname), bbox_inches="tight")


def plot(data, i, **sfparam) -> None:
    fig, ax = plt.subplots()
    ax.plot(data.i, getattr(data, f"in{i}"), label=f"in{i}")
    ax.plot(data.i, getattr(data, f"out{i}"), label=f"out{i}")
    ax.grid(axis="y")
    ax.legend()
    pparam = {
        "xlabel": "samples",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = f"filter-{i}"
    savefig(fig, **sfparam)


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot script for send_sinusoidal_command.py")
    parser.add_argument("data", help="path to data file")
    parser.add_argument("-i", "--index", default=[0, 1, 2], nargs="+", type=int, help="index to show")
    parser.add_argument("-d", "--basedir", default="fig", help="directory where figures will be saved")
    parser.add_argument(
        "-e",
        "--extension",
        default=["png"],
        nargs="+",
        help="extensions to save as figures",
    )
    parser.add_argument("-s", "--savefig", action="store_true", help="export figures if specified")
    parser.add_argument("-x", "--noshow", action="store_true", help="do not show figures if specified")
    return parser.parse_args()


def main() -> None:
    args = parse()
    sfparam = sfparam_tmpl.copy()
    sfparam["savefig"] = args.savefig
    sfparam["basedir"] = args.basedir
    sfparam["extensions"] = args.extension

    data = Data(args.data)
    for i in args.index:
        plot(data, i, **sfparam)
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()

# Local Variables:
# jinx-local-words: "basedir env noqa noshow png py savefig sfparam usr xlabel"
# End:
