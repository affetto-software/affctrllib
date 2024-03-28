"""Const moudle.

This module defines constants that used through this library.
"""

from __future__ import annotations

from numpy import pi

# The acceptible amount of error.
TOL = 1e-6

# The circle ratio.
PI = pi
PIx2 = 2.0 * PI
PI_2 = 0.5 * PI

# The ration of radian per degree (and vice versa).
RAD_PER_DEG = PI / 180.0
DEG_PER_RAD = 180.0 / PI
