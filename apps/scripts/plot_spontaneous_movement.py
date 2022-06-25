#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from pyplotutil.datautil import Data

sfparam_tmpl = {
    "time": None,
    "savefig": False,
    "basedir": "fig",
    "filename": None,
    "extensions": ["svg"],
}


def savefig(fig, **sfparam):
    if not sfparam.get("savefig", False):
        return
    if not "filename" in sfparam:
        raise KeyError(f"filename is required in sfparam.")

    # Join basedir and filename.
    path = Path(sfparam.get("basedir", "fig")) / Path(sfparam["filename"])
    # Create directories if needed.
    path.parent.mkdir(parents=True, exist_ok=True)
    # Save figures in specified formats.
    for ext in sfparam.get("extensions", ["svg"]):
        if not ext.startswith("."):
            ext = f".{ext}"
        fname = path.with_suffix(ext)
        fig.savefig(str(fname), bbox_inches="tight")


def make_mask(t, between=None):
    if between is None:
        return np.full(t.size, True)
    elif len(between) == 1:
        return t <= between[0]
    else:
        return (t >= between[0]) & (t <= between[1])


def plot_y(data, **sfparam):
    fig, ax = plt.subplots()
    mask = make_mask(data.t, sfparam["time"])
    ax.plot(data.t[mask], data.y1[mask], label=f"y1")
    ax.plot(data.t[mask], data.y2[mask], label=f"y2")
    ax.grid(axis="y")
    ax.legend(title="Output")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "output",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "command"
    savefig(fig, **sfparam)


def parse():
    parser = argparse.ArgumentParser(
        description="Plot script for spontaneous_movement.py"
    )
    parser.add_argument("data", help="path to data file")
    parser.add_argument(
        "-d", "--basedir", default="fig", help="directory where figures will be saved"
    )
    parser.add_argument(
        "-e",
        "--extension",
        default=["png"],
        nargs="+",
        help="extensions to save as figures",
    )
    parser.add_argument(
        "-t", "--time", nargs="+", type=float, help="time range to show in figure"
    )
    parser.add_argument(
        "-s", "--savefig", action="store_true", help="export figures if specified"
    )
    parser.add_argument(
        "-x", "--noshow", action="store_true", help="do not show figures if specified"
    )
    return parser.parse_args()


def main():
    args = parse()
    sfparam = sfparam_tmpl.copy()
    sfparam["time"] = args.time
    sfparam["savefig"] = args.savefig
    sfparam["basedir"] = args.basedir
    sfparam["extensions"] = args.extension

    data = Data(args.data)
    plot_y(data, **sfparam)
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
