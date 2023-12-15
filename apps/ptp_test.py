#!/usr/bin/env python

import numpy as np

from affctrllib import PTP, Logger

q0 = np.array([0, 0, 127])
qF = np.array([127, 255, 0])
vmax = np.array([16, 32, 16])
tb = np.array([2, 3, 4])
T = 10
t0 = 0
profile = "trapez"  # or "5th" or "tri" or "sin"
N = 1000

if profile in ["tri", "5th", "sin"]:
    ptp = PTP(q0, qF, T, t0, profile_name=profile)
else:
    ptp = PTP(q0, qF, T, t0, vmax=vmax, profile_name=profile)
    # ptp = PTP(q0, qF, T, t0, tb=tb, profile_name=profile)
logger = Logger()
logger.set_labels(["t", "q0", "q1", "q2", "dq0", "dq1", "dq2", "ddq0", "ddq1", "ddq2"])
dt = (t0 + T) / N
dq_prev = 0
for i in range(N):
    t = i * dt
    q = ptp.q(t)
    dq = ptp.dq(t)
    ddq = (ptp.dq(t) - dq_prev) / dt
    dq_prev = dq
    logger.store_data(np.concatenate(([t], q, dq, ddq)))
logger.dump()
