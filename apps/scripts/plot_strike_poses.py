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


def plot_command(data, joints, **sfparam):
    fig, ax = plt.subplots()
    mask = make_mask(data.t, sfparam["time"])
    for i in joints:
        ax.plot(data.t[mask], getattr(data, f"ca{i}")[mask], label=f"ca[{i}]")
        ax.plot(data.t[mask], getattr(data, f"cb{i}")[mask], label=f"cb[{i}]")
    ax.grid(axis="y")
    ax.legend(title="Command")
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
    mask = make_mask(data.t, sfparam["time"])
    for i in joints:
        ax.plot(data.t[mask], getattr(data, f"pa{i}")[mask], label=f"pa[{i}]")
        ax.plot(data.t[mask], getattr(data, f"pb{i}")[mask], label=f"pb[{i}]")
    ax.grid(axis="y")
    ax.legend(title="Measured pressure")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "pressure [kPa]",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "pressure"
    savefig(fig, **sfparam)


def plot_pressure_command(data, joints, **sfparam):
    fig, ax = plt.subplots()
    mask = make_mask(data.t, sfparam["time"])
    for i in joints:
        (line,) = ax.plot(
            data.t[mask],
            getattr(data, f"ca{i}")[mask] * 600 / 255,
            ls="--",
            label=f"ca[{i}]",
        )
        (line,) = ax.plot(
            data.t[mask],
            getattr(data, f"pa{i}")[mask],
            c=line.get_color(),
            ls="-",
            label=f"pa[{i}]",
        )
        (line,) = ax.plot(
            data.t[mask],
            getattr(data, f"cb{i}")[mask] * 600 / 255,
            ls="--",
            label=f"cb[{i}]",
        )
        (line,) = ax.plot(
            data.t[mask],
            getattr(data, f"pb{i}")[mask],
            c=line.get_color(),
            ls="-",
            label=f"pb[{i}]",
        )
    ax.grid(axis="y")
    ax.legend(title="Pressure values")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "pressure [kPa]",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "pressure_command"
    savefig(fig, **sfparam)


def plot_q(data, joints, **sfparam):
    fig, ax = plt.subplots()
    mask = make_mask(data.t, sfparam["time"])
    for i in joints:
        ax.plot(data.t[mask], getattr(data, f"qdes{i}")[mask], label=f"qdes[{i}]")
        ax.plot(data.t[mask], getattr(data, f"q{i}")[mask], label=f"q[{i}]")
    ax.grid(axis="y")
    ax.legend(title="Joint angle")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "position [0-100]",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "q"
    savefig(fig, **sfparam)


def plot_dq(data, joints, **sfparam):
    fig, ax = plt.subplots()
    mask = make_mask(data.t, sfparam["time"])
    for i in joints:
        ax.plot(data.t[mask], getattr(data, f"dqdes{i}")[mask], label=f"dqdes[{i}]")
        ax.plot(data.t[mask], getattr(data, f"dq{i}")[mask], label=f"dq[{i}]")
    ax.grid(axis="y")
    ax.legend(title="Joint angle velocity")
    pparam = {
        "xlabel": "time [s]",
        "ylabel": "velocity [/s]",
    }
    ax.set(**pparam)
    if sfparam.get("filename", None) is None:
        sfparam["filename"] = "dq"
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
    if args.joint is None:
        args.joint = [11]

    data = Data(args.data)
    # plot_command(data, args.joint, **sfparam)
    # plot_pressure(data, args.joint, **sfparam)
    plot_pressure_command(data, args.joint, **sfparam)
    plot_q(data, args.joint, **sfparam)
    plot_dq(data, args.joint, **sfparam)
    if not args.noshow:
        plt.show()


if __name__ == "__main__":
    main()
