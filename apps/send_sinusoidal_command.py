#!/usr/bin/env python

import argparse
import copy
import os
from typing import Any

import numpy as np
import tomli
from affctrllib import AffComm, Logger, Timer

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")
DOF = 13
LABELS = ["t"]
for i in range(DOF):
    LABELS.extend([f"q{i}", f"pa{i}", f"pb{i}"])
LABELS.extend([f"ca{i}" for i in range(DOF)])
LABELS.extend([f"cb{i}" for i in range(DOF)])


# Default values
DEFAULT_RATE = 30
DEFAULT_TIME = 55
DEFAULT_PROFILES: dict[str, dict[str, Any]] = {
    "constant": {
        "value": 0,
    },
    "sinusoidal": {
        "amplitude": 127,
        "period": 10,
        "base": 127,
        "idletime": 5,
        "phase": 0,
    },
}
DEFAULT_PROFILE = "constant"


def sinusoidal(t, amplitude=1.0, period=1.0, base=0.0, idletime=0.0, phase=0.0) -> int:
    omega = 2.0 * np.pi / period
    if isinstance(phase, str):
        phase = eval(phase)
    if t < idletime:
        x = 0.0
    else:
        x = omega * (t - idletime) - phase
    return round(amplitude * np.sin(x) + base)


def constant(_, value=0.0) -> int:
    return round(value)


def print_parameters(default_params, specific_params_list):
    print(f"Default parameters:")
    print(f'  profile: {default_params["profile"]}')
    print(f'  params: {default_params["params"]}')
    if len(specific_params_list) > 0:
        print(f"Joint-specific parameters:")
    for p in specific_params_list:
        print(f"  Joint {p[0]}:")
        print(f'    profile: {p[1]["profile"]}')
        print(f'    params: {p[1]["params"]}')
        for side in ("ca", "cb"):
            if side in p[1]:
                print(f"    A-side of joint {p[0]}:")
                print(f'      profile: {p[1][side]["profile"]}')
                print(f'      params: {p[1][side]["params"]}')


def load_parameters(config):
    with open(config, "rb") as f:
        config_dict = tomli.load(f)
    try:
        command = config_dict["command"]
    except KeyError:
        raise KeyError(f"No 'command' in {str(config)}")

    # Load default values for each profile.
    default_profiles: dict[str, dict[str, Any]] = copy.deepcopy(DEFAULT_PROFILES)
    for key, params in command.get("profile", {}).items():
        if key in default_profiles:
            default_profiles[key].update(params)
        else:
            default_profiles[key] = params.copy()

    # Load default values for Affetto.
    try:
        affetto = command.get("affetto", {})
    except KeyError:
        raise KeyError(f"No definitions for 'affetto' in {str(config)}")

    default_params: dict[str, str | dict[str, Any]] = {}
    if "profile" in affetto:
        default_params["profile"] = affetto["profile"]
        default_params["params"] = default_profiles[affetto["profile"]].copy()
    else:
        default_params["profile"] = DEFAULT_PROFILE
        default_params["params"] = default_profiles[DEFAULT_PROFILE].copy()
    default_params["params"].update(affetto.get("params", {}))

    # Load specific values for each joint.
    specific_params_list: list[tuple[int, dict[str, str | dict[str, Any]]]] = []
    for ith in affetto:
        # note: ith is str.
        if not ith.isnumeric():
            continue

        ith_params: dict[str, str | dict[str, Any]] = {}
        if "profile" in affetto[ith]:
            ith_params["profile"] = affetto[ith]["profile"]
            if ith_params["profile"] == default_params["profile"]:
                ith_params["params"] = default_params["params"].copy()
            else:
                ith_params["params"] = default_profiles[affetto[ith]["profile"]].copy()
        else:
            ith_params["profile"] = default_params["profile"]
            ith_params["params"] = default_params["params"].copy()
        ith_params["params"].update(affetto[ith].get("params", {}))

        for side in ("ca", "cb"):
            if side in affetto[ith]:
                ith_params_side: dict[str, str | dict[str, Any]] = {}
                if "profile" in affetto[ith][side]:
                    ith_params_side["profile"] = affetto[ith][side]["profile"]
                    if ith_params_side["profile"] == ith_params["profile"]:
                        ith_params_side["params"] = ith_params["params"].copy()
                    else:
                        ith_params_side["params"] = default_profiles[affetto[ith][side]["profile"]].copy()
                else:
                    ith_params_side["profile"] = ith_params["profile"]
                    ith_params_side["params"] = ith_params["params"].copy()
                ith_params_side["params"].update(affetto[ith][side].get("params", {}))
                ith_params[side] = ith_params_side.copy()
        specific_params_list.append((int(ith), ith_params.copy()))
    print_parameters(default_params, specific_params_list)  # for debug
    return default_params, specific_params_list


PROFILE_TO_FUNC_MAP = {
    "constant": constant,
    "sinusoidal": sinusoidal,
}


def generate_commands(t, default_params, specific_params_list) -> tuple[np.ndarray, np.ndarray]:
    # Set default values.
    val = PROFILE_TO_FUNC_MAP[default_params["profile"]](t, **default_params["params"])
    ca = np.full((DOF,), val)
    cb = np.full((DOF,), val)
    # Set joint-specific values.
    for i, params in specific_params_list:
        val = PROFILE_TO_FUNC_MAP[params["profile"]](t, **params["params"])
        ca[i] = val
        cb[i] = val
        if "ca" in params:
            ca[i] = PROFILE_TO_FUNC_MAP[params["ca"]["profile"]](t, **params["ca"]["params"])
        if "cb" in params:
            cb[i] = PROFILE_TO_FUNC_MAP[params["cb"]["profile"]](t, **params["cb"]["params"])
    return (ca, cb)


def mainloop(config, output, freq, time, default_params, specific_params_list):
    acom = AffComm(config)
    print(acom)

    acom.create_sockets()
    logger = Logger(output)
    logger.set_labels(LABELS)

    def cleanup():
        acom.close()
        logger.dump()

    t = 0
    timer = Timer(rate=freq)
    timer.start()
    while t < time:
        # Get the current time elapsed.
        t = timer.elapsed_time()

        # Receive sensory data.
        sdata = acom.receive_as_list()

        # Generate pressure values to send.
        ca, cb = generate_commands(t, default_params, specific_params_list)

        # Send commands.
        acom.send_commands(ca, cb)

        # Logging.
        logger.store_data([t] + sdata + list(ca) + list(cb))
        print(f"\rt = {t:.2f}", end="")

        # Block process for a certain period.
        timer.block()
    print()

    cleanup()


def parse():
    parser = argparse.ArgumentParser(description="Send sinusoidal actuation commands.")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG_PATH, help="config file")
    parser.add_argument("-o", "--output", default=None, help="output filename")
    parser.add_argument(
        "-C",
        "--command",
        default=None,
        help="command profile configuration",
    )
    parser.add_argument(
        "-H",
        "--hz",
        dest="freq",
        default=DEFAULT_RATE,
        type=float,
        help="frequency to send and receive data",
    )
    parser.add_argument(
        "-t",
        "--time",
        default=DEFAULT_TIME,
        type=float,
        help="time [s] until program ends",
    )
    # parser.add_argument(
    #     "-a",
    #     "--amplitude",
    #     default=0,
    #     type=float,
    #     help="amplitude (int) for sinusoidal wave",
    # )
    # parser.add_argument(
    #     "-T",
    #     "--period",
    #     default=0,
    #     type=float,
    #     help="time period [s] for sinusoidal wave",
    # )
    return parser.parse_args()


def main():
    # Parse arguments.
    args = parse()

    # Load parameters.
    default_params = {
        "profile": DEFAULT_PROFILE,
        "params": DEFAULT_PROFILES[DEFAULT_PROFILE],
    }
    specific_params_list = []
    if args.command is not None:
        default_params, specific_params_list = load_parameters(args.command)

    # Run mainloop.
    mainloop(
        args.config,
        args.output,
        args.freq,
        args.time,
        default_params,
        specific_params_list,
    )


if __name__ == "__main__":
    main()
