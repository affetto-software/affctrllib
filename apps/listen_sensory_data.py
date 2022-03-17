#!/usr/bin/env python3

from affctrllib.affcomm import AffComm

if __name__ == "__main__":
    acom = AffComm()
    acom.load_config("tests/config/default.toml")
    acom.listen()
