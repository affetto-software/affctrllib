#!/usr/bin/env python

import numpy as np
from affctrllib import PTP, Logger

q0 = np.array([0, 0, 127])
qF = np.array([127, 255, 0])
T = 10
t0 = 0
profile = "tri"  # or "5th"
N = 1000

ptp = PTP(q0, qF, T, t0, profile_name=profile)
logger = Logger()
logger.set_labels(["t", "q0", "q1", "q2", "dq0", "dq1", "dq2", "ddq0", "ddq1", "ddq2"])
dt = (t0 + T) / N
dq_prev = 0
for i in range(N):
    t = i * dt
    q = ptp.q(t)
    dq = ptp.dq(t)
    ddq = (ptp.dq(t) - dq_prev) / dt  # type: ignore
    dq_prev = dq
    logger.store_data(np.concatenate(([t], q, dq, ddq)))  # type: ignore
logger.dump()
