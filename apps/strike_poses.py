#!/usr/bin/env python

import argparse
import os

import numpy as np
import tomli
from affctrllib import PTP, AffComm, AffCtrl, AffState, Logger, Timer

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
# command data
LABELS.extend([f"qdes{i}" for i in range(DOF)])
LABELS.extend([f"dqdes{i}" for i in range(DOF)])
LABELS.extend([f"ca{i}" for i in range(DOF)])
LABELS.extend([f"cb{i}" for i in range(DOF)])
BUFSIZE = 1024


def logging(logger, t, astate, qdes, dqdes, ca, cb):
    logger.store_data([t])
    logger.extend_data(astate.raw_q)
    logger.extend_data(astate.raw_pa)
    logger.extend_data(astate.raw_pb)
    logger.extend_data(astate.q)
    logger.extend_data(astate.dq)
    logger.extend_data(astate.pa)
    logger.extend_data(astate.pb)
    logger.extend_data(qdes)
    logger.extend_data(dqdes)
    logger.extend_data(ca)
    logger.extend_data(cb)


def mainloop(config, output, freq, keyframes, initial=None, profile="tri"):
    acom = AffComm(config)
    print(acom)

    acom.create_sockets()
    logger = Logger(output)
    logger.set_labels(LABELS)

    astate = AffState(config)
    if freq > 0:
        astate.freq = freq
    for _ in range(10):
        sarr = acom.receive_as_list()
        astate.update(sarr)

    actrl = AffCtrl(config)
    timer = Timer(rate=astate.freq)

    def cleanup():
        acom.close()
        print(f"\nSaving data in <{str(output)}>...")
        logger.dump()

    def moveto(t0, q0, T, qF, profile, msg=None):
        if msg:
            print(msg, flush=True)
        t = t0
        ptp = PTP(q0, qF, T, t0, profile)
        timer.start()
        while t < t0 + T:
            t = t0 + timer.elapsed_time()
            sarr = acom.receive_as_list()
            astate.update(sarr)
            qdes = ptp.q(t)
            # dqdes = ptp.dq(t)
            dqdes = np.zeros(shape=(actrl.dof,))
            ca, cb = actrl.update(
                t, astate.q, astate.dq, astate.pa, astate.pb, qdes, dqdes
            )
            acom.send_commands(ca, cb)
            logging(logger, t, astate, qdes, dqdes, ca, cb)
            print(f"\rt = {t:.2f}", end="")
            # timer.block()

    q0 = acom.receive_as_2darray()[0]
    if initial:
        t0 = -5
        time = 5
        msg = f"\nMoving to initial pose (in {time} sec.) ..."
        moveto(t0, q0, time, np.array(initial), profile, msg)
        q0 = np.array(initial)

    t0 = 0
    for cnt, kf in enumerate(keyframes):
        time = kf[0]
        msg = f"\nMoving to keyframe {cnt} (in {time} sec.) ..."
        moveto(t0, q0, time, np.array(kf[1]), profile, msg)
        # t0 = t0 + time
        t0 = t0 + timer.elapsed_time()
        q0 = np.array(kf[1])
    cleanup()


def load_keyframe(keyframe):
    with open(keyframe, "rb") as f:
        d = tomli.load(f)
    k = d["keyframe"]
    profile = k["profile"]
    q0 = k["initial"]["q"]
    frames = []
    for kf in k["frames"]:
        frames.append((kf["T"], kf["q"]))
    return profile, q0, frames


def parse():
    parser = argparse.ArgumentParser(description="Strike poses")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-k", "--keyframe", required=True, help="keyframe file")
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument(
        "-H",
        "--hz",
        dest="freq",
        default=0,
        type=float,
        help="frequency to send and receive data",
    )
    parser.add_argument(
        "-p",
        "--profile",
        default=None,
        help="Point to Point interpolation profile",
    )
    return parser.parse_args()


def main():
    args = parse()
    profile, q0, keyframes = load_keyframe(args.keyframe)
    if args.profile:
        profile = args.profile
    mainloop(
        args.config,
        args.output,
        args.freq,
        keyframes,
        q0,
        profile,
    )


if __name__ == "__main__":
    main()
