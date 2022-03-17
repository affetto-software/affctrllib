#!/usr/bin/env python3

from affctrllib.affmock import AffettoMock

if __name__ == "__main__":
    mock = AffettoMock()
    mock.load_config("tests/config/mock.toml")
    mock.sensor_rate = 10
    mock.start()
