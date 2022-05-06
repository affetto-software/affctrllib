#!/usr/bin/env python

import argparse
import os
import re
import sys
import threading
import time
from pathlib import Path

import affctrllib as acl
from affctrllib import AffComm, Timer

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
DOF = 13
BUFSIZE = 1024


def usage():
    print(f" =========== Operations ========== : <Key>")
    print(f"             preview current angles: <RET>")
    print(f"             capture current angles: <c>")
    print(f"capture current angles after 2 sec.: <2c>")
    print(f"       save captured angles to file: <s>")
    print(f"                      quit and save: <q>")
    print(f"                quit without saving: <q!>")
    print(f"                  show help message: <h>")
    print(f"")


def make_toml_string(frames):
    dof = 13
    profile = "triangular"

    string = f"[keyframe]\n"
    string += f"dof = {dof}\n"
    string += f'profile = "{profile}"\n'
    string += f"\n"
    if len(frames) > 0:
        q = frames[0][1]
        string += f"[keyframe.initial]\n"
        string += f"q = {q}\n"
        string += f"\n"
    for frame in frames:
        T = frame[0]
        q = frame[1]
        string += f"[[keyframe.frames]]\n"
        string += f"T = {T}\n"
        string += f"q = {q}\n"
        string += f"\n"
    return string


def save(frames, output):
    string = make_toml_string(frames)
    if output is None:
        print(string)
    else:
        path = Path(output).with_suffix(".toml")
        with open(path, mode="w") as fobj:
            fobj.write(string)
        print(f"Saved captured keyframes in <{str(path)}>")


class ReceivedBytes(object):
    _bytes: bytes

    def __init__(self):
        self._bytes = bytes()
        self._lock = threading.Lock()

    def get(self) -> bytes:
        with self._lock:
            return self._bytes

    def set(self, _bytes: bytes) -> None:
        with self._lock:
            self._bytes = _bytes


def receive_continuously(ssock, recv_bytes: ReceivedBytes, event: threading.Event):
    while event.is_set():
        rb, _ = ssock.recvfrom(BUFSIZE)
        recv_bytes.set(rb)


def mainloop(config, output):
    acom = AffComm(config)
    print(acom)
    usage()

    # Create a thread to receive continuously.
    ssock = acom.create_sensory_socket()
    recv_bytes = ReceivedBytes()
    event = threading.Event()
    th = threading.Thread(target=receive_continuously, args=(ssock, recv_bytes, event))
    event.set()
    th.start()

    def cleanup():
        event.clear()
        th.join()
        acom.close()

    frames = []
    timer = Timer()
    t0 = 0
    timer.start()

    def capture():
        t = timer.elapsed_time()
        sarr = acl.split_received_msg(recv_bytes.get())
        q = list(map(int, acl.unzip_array(sarr)[0]))
        T = round(t - t0, 1)
        return T, q

    try:
        while True:
            c = input("> ")
            if c == "q":
                print("Quitting...")
                save(frames, output)
                break
            elif c == "q!":
                print("Quitting without saving...")
                break
            elif c == "c":
                T, q = capture()
                frames.append((T, q))
                print(f"T = {T}, q={q}")
                print("Successfully captured!")
                t0 = timer.elapsed_time()
            elif (len(c) > 1) and (
                (c[-1] == "c" and c[0].isdigit()) or (c[0] == "c" and c[-1].isdigit())
            ):
                wait_t = re.findall(r"\d+", c)[0]
                print(f"Waiting for {wait_t} sec...")
                time.sleep(int(wait_t))
                T, q = capture()
                frames.append((T, q))
                print(f"T = {T}, q={q}")
                print("Successfully captured!")
                t0 = timer.elapsed_time()
            elif c == "h":
                usage()
            elif c == "s":
                save(frames, output)
                print("Successfully saved!")
            else:
                T, q = capture()
                print(f"T = {T}, q={q}")
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt")
        cleanup()
        sys.exit(1)
    cleanup()


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
