#!/usr/bin/env python

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from pyplotutil.datautil import Data

sfparam_tmpl = {
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
        fig.savefig(str(fname))


def plot_command(data, joints, **sfparam):
    fig, ax = plt.subplots()
    for i in joints:
        ax.plot(data.t, getattr(data, f"ca{i}"), label=f"ca[{i}]")
        ax.plot(data.t, getattr(data, f"cb{i}"), label=f"cb[{i}]")
    ax.grid(axis="y")
    ax.legend(title="Command")
    ax.autoscale(tight=True)
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "command [0-255]",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "command"
    savefig(fig, **sfparam)


def plot_pressure(data, joints, **sfparam):
    fig, ax = plt.subplots()
    for i in joints:
        ax.plot(data.t, getattr(data, f"pa{i}"), label=f"pa[{i}]")
        ax.plot(data.t, getattr(data, f"pb{i}"), label=f"pb[{i}]")
    ax.grid(axis="y")
    ax.legend(title="Measured pressure")
    ax.autoscale(tight=True)
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "pressure [0-255]",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "pressure"
    savefig(fig, **sfparam)


def plot_q(data, joints, **sfparam):
    fig, ax = plt.subplots()
    for i in joints:
        ax.plot(data.t, getattr(data, f"q{i}"), label=f"q[{i}]")
    ax.grid(axis="y")
    ax.legend(title="Joint angle")
    ax.autoscale(tight=True)
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "position [0-255]",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "q"
    savefig(fig, **sfparam)


def parse():
    parser = argparse.ArgumentParser(
        description="Plot script for send_sinusoidal_command.py"
    )
    parser.add_argument("data", help="path to data file")
    parser.add_argument(
        "-j",
        "--joint",
        nargs="+",
        type=int,
        help="joint indices to be shown in figure",
    )
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
    if args.joint is None:
        args.joint = [11]

    data = Data(args.data)
    plot_command(data, args.joint, **sfparam)
    plot_pressure(data, args.joint, **sfparam)
    plot_q(data, args.joint, **sfparam)
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
