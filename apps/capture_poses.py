#!/usr/bin/env python

import argparse
import re
import time
from pathlib import Path

from affctrllib import AffStateThread, Timer

DEFAULT_CONFIG_PATH = Path(__file__).parent.joinpath("config.toml")
MENU = [
    ("RET", "Preview current joint angles"),
    ("c", "Capture current joint angles"),
    ("2c", "Capture current angles after 2 sec"),
    ("s", "Save captured angles to file"),
    ("q", "Quit and save"),
    ("q!", "Quit without saving"),
    ("h", "Show help message"),
]


def usage():
    maxlen_key = 0
    maxlen_desc = 0
    for key, desc in MENU:
        maxlen_key = len(key) if len(key) > maxlen_key else maxlen_key
        maxlen_desc = len(desc) if len(desc) > maxlen_desc else maxlen_desc
    sep = ": "
    maxlen = maxlen_key + maxlen_desc + len(sep)
    title = f"Operations"
    header = (
        f"{' ' + title[:int(len(title)/2)]:=>{int(maxlen/2)}}"
        + f"{title[int(len(title)/2):] + ' ':=<{int(maxlen/2+maxlen%2)}}"
    )
    menu = header + "\n"
    for key, desc in MENU:
        menu += f"{key: >{maxlen_key}}{sep}{desc}\n"
    print(menu)


def make_toml_string(frames):
    dof = 13
    profile = "trapezoidal"

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


def capture(t0, timer: Timer, astate: AffStateThread):
    t = timer.elapsed_time()
    q = list(astate.q.astype(int))
    T = round(t - t0, 1)
    return t, T, q


def mainloop(config, output, freq=None):
    if freq is None:
        astate = AffStateThread(config)
    else:
        astate = AffStateThread(config, freq=freq)

    frames = []
    timer = Timer()
    t0 = 0
    astate.prepare()
    astate.start()
    timer.start()
    try:
        usage()
        while True:
            c = input("> ")
            if c == "q":
                save(frames, output)
                break
            elif c == "q!":
                break
            elif c == "h":
                usage()
            elif c == "s":
                save(frames, output)
            elif "c" in c:
                if len(c) > 1:
                    try:
                        wait_t = re.findall(r"\d+", c)[0]
                        print(f"Waiting for {wait_t} sec...")
                        time.sleep(int(wait_t))
                    except:
                        pass
                t0, T, q = capture(t0, timer, astate)
                frames.append((T, q))
                print(f"T = {T}, q = {q}")
                print("Successfully captured!")
            else:
                _, T, q = capture(t0, timer, astate)
                print(f"T = {T}, q = {q}")
    finally:
        print("quitting...")
        astate.join()


def parse():
    parser = argparse.ArgumentParser(description="Capture static poses.")
    parser.add_argument(
        "-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file"
    )
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument("-H", "--hz", type=float, help="sampling frequency")
    return parser.parse_args()


def main():
    args = parse()
    mainloop(args.config, args.output, args.hz)


if __name__ == "__main__":
    main()
