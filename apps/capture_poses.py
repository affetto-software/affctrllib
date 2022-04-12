#!/usr/bin/env python

import argparse
import os
from pathlib import Path

import affctrllib as acl
from affctrllib import AffComm, Timer

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
DOF = 13
BUFSIZE = 1024


def usage():
    print(f"Operations: <Key>")
    print(f"   capture: <RET>")
    print(f"   quit:    <q>")
    print(f"")


def make_export_string(frames):
    dof = 13
    profile = "triangular"

    string = f"[keyframe]\n"
    string += f"dof = {dof}\n"
    string += f'profile = "{profile}"\n'
    string += f"\n"
    if len(frames) > 0:
        string += f"[keyframe.initial]\n"
        string += f"q = {frames[0][1]}\n"
        string += f"\n"
    for frame in frames:
        t = round(frame[0], 1)
        q = frame[1]
        string += f"[[keyframe.frames]]\n"
        string += f"t = {t}\n"
        string += f"q = {q}\n"
        string += f"\n"
    return string


def export(frames, output):
    string = make_export_string(frames)
    if output is None:
        print(string)
    else:
        path = Path(output)
        with open(path.with_suffix(".toml"), mode="w") as fobj:
            fobj.write(string)
        print(f"Saved captured keyframes in <{str(path)}>")


def mainloop(config, output):
    acom = AffComm(config)
    print(acom)
    usage()

    timer = Timer()
    frames = []
    ssock = acom.create_sensory_socket()

    t = 0
    timer.start()
    while True:
        c = input("> ")
        if c == "q":
            print("Quitting...")
            break
        t = timer.elapsed_time()
        recv_bytes, _ = ssock.recvfrom(BUFSIZE)
        sarr = acl.split_received_msg(recv_bytes, function=int)
        q = acl.unzip_array(sarr)[0]
        frames.append((t, q))
        print("Successfully captured!")

    ssock.close()
    export(frames, output)


def parse():
    parser = argparse.ArgumentParser(description="Capture static poses.")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", default=None, help="output filename")
    return parser.parse_args()


def main():
    args = parse()
    mainloop(args.config, args.output)


if __name__ == "__main__":
    main()
