#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import affctrllib as acl
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

# Default values
DEFAULT_FREQ = 30


def logging(logger, t, astate, qdes, dqdes, ca, cb):
    logger.store_data([t])
    logger.extend_data(astate.raw_q)  # type: ignore
    logger.extend_data(astate.raw_pa)  # type: ignore
    logger.extend_data(astate.raw_pb)  # type: ignore
    logger.extend_data(astate.q)  # type: ignore
    logger.extend_data(astate.dq)  # type: ignore
    logger.extend_data(astate.pa)  # type: ignore
    logger.extend_data(astate.pb)  # type: ignore
    logger.extend_data(qdes)
    logger.extend_data(dqdes)
    logger.extend_data(ca)
    logger.extend_data(cb)


def mainloop(config, output, freq, keyframes, initial=None, profile="tri"):
    acom = AffComm(config)
    print(acom)

    ssock = acom.create_sensory_socket()
    csock = acom.create_command_socket()
    logger = Logger(output)
    logger.set_labels(LABELS)

    astate = AffState(freq=freq)
    actrl = AffCtrl(config)

    def cleanup():
        csock.close()
        ssock.close()
        print(f"\nSaving data in <{str(output)}>...")
        logger.dump()

    t = 0
    timer = Timer()

    recv_bytes, _ = ssock.recvfrom(BUFSIZE)
    sarr = acl.split_received_msg(recv_bytes, function=int)
    data = acl.reshape_array_for_unzip(sarr, ncol=3)
    q0 = data[0]  # type: ignore
    time = 5
    timer.start()
    ptp = PTP(q0, initial, time, 0, profile)
    print(f"Moving to initial pose...")
    while t < time:
        t = timer.elapsed_time()
        recv_bytes, _ = ssock.recvfrom(BUFSIZE)
        sarr = acl.split_received_msg(recv_bytes, function=int)
        astate.update(sarr)
        qdes = ptp.q(t)
        dqdes = ptp.dq(t)
        ca, cb = actrl.update(t, astate.q, astate.dq, astate.pa, astate.pb, qdes, dqdes)
        carr = acl.zip_arrays(ca, cb)
        send_bytes = acl.convert_array_to_bytes(carr)
        csock.sendto(send_bytes, acom.remote_addr.addr)
        timer.block()

    t = 0
    timer.start()
    for cnt, kf in enumerate(keyframes):
        recv_bytes, _ = ssock.recvfrom(BUFSIZE)
        sarr = acl.split_received_msg(recv_bytes, function=int)
        data = acl.reshape_array_for_unzip(sarr, ncol=3)
        q0 = data[0]  # type: ignore
        time = kf[0]
        t0 = t
        qF = np.array(kf[1])
        ptp = PTP(q0, qF, time - t0, t0, profile)
        print(f"Moving to keyframe {cnt}...")
        while t < time:
            t = timer.elapsed_time()
            recv_bytes, _ = ssock.recvfrom(BUFSIZE)
            sarr = acl.split_received_msg(recv_bytes, function=int)
            astate.update(sarr)
            qdes = ptp.q(t)
            dqdes = ptp.dq(t)
            ca, cb = actrl.update(
                t, astate.q, astate.dq, astate.pa, astate.pb, qdes, dqdes
            )
            carr = acl.zip_arrays(ca, cb)
            send_bytes = acl.convert_array_to_bytes(carr)
            csock.sendto(send_bytes, acom.remote_addr.addr)
            logging(logger, t, astate, qdes, dqdes, ca, cb)
            print(f"\rt = {t:.2f}", end="")
            timer.block()
    cleanup()


def load_keyframe(keyframe):
    with open(keyframe, "rb") as f:
        d = tomli.load(f)
    k = d["keyframe"]
    freq = k["freq"]
    profile = k["profile"]
    q0 = k["initial"]["q"]
    frames = []
    for kf in k["frames"]:
        frames.append((kf["t"], kf["q"]))
    return freq, profile, q0, frames


def parse():
    parser = argparse.ArgumentParser(description="Strike poses")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-k", "--keyframe", default=None, help="keyframe file")
    parser.add_argument("-o", "--output", default=None, help="output filename")
    return parser.parse_args()


def main():
    args = parse()
    freq, profile, initial, keyframes = load_keyframe(args.keyframe)
    mainloop(
        args.config,
        args.output,
        freq,
        keyframes,
        initial,
        profile,
    )


if __name__ == "__main__":
    main()
