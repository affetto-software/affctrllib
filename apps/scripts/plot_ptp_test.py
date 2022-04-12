#!/usr/bin/env python

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
    for ext in sfparam.get("extensions", ["png"]):
        if not ext.startswith("."):
            ext = f".{ext}"
        fname = path.with_suffix(ext)
        fig.savefig(str(fname))


def plot_q(data, **sfparam):
    fig, ax = plt.subplots()
    for i in [0, 1, 2]:
        ax.plot(data.t, getattr(data, f"q{i}"), label=f"q[{i}]")
    ax.grid(axis="y")
    ax.legend()
    ax.autoscale(tight=True)
    pparam = {
        "xlabel": "samples",
        "ylabel": "q",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "q"
    savefig(fig, **sfparam)


def plot_dq(data, **sfparam):
    fig, ax = plt.subplots()
    for i in [0, 1, 2]:
        ax.plot(data.t, getattr(data, f"dq{i}"), label=f"dq[{i}]")
    ax.grid(axis="y")
    ax.legend()
    ax.autoscale(tight=True)
    pparam = {
        "xlabel": "samples",
        "ylabel": "dq",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "dq"
    savefig(fig, **sfparam)


def plot_ddq(data, **sfparam):
    fig, ax = plt.subplots()
    for i in [0, 1, 2]:
        ax.plot(data.t, getattr(data, f"ddq{i}"), label=f"ddq[{i}]")
    ax.grid(axis="y")
    ax.legend()
    ax.autoscale(tight=True)
    pparam = {
        "xlabel": "samples",
        "ylabel": "ddq",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "ddq"
    savefig(fig, **sfparam)


def parse():
    parser = argparse.ArgumentParser(
        description="Plot script for send_sinusoidal_command.py"
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
        "-s", "--savefig", action="store_true", help="export figures if specified"
    )
    parser.add_argument(
        "-x", "--noshow", action="store_true", help="do not show figures if specified"
    )
    return parser.parse_args()


def main():
    args = parse()
    sfparam = sfparam_tmpl.copy()
    sfparam["savefig"] = args.savefig
    sfparam["basedir"] = args.basedir
    sfparam["extensions"] = args.extension

    data = Data(args.data)
    plot_q(data, **sfparam)
    plot_dq(data, **sfparam)
    plot_ddq(data, **sfparam)
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
