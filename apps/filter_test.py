#!/usr/bin/env python

import numpy as np

from affctrllib import Filter, Logger

N = 1000  # number of samples
M = 5  # number of points for averaging
dim = 3
rec = [(200, 400), (400, 600), (600, 800)]  # rectangular vertices
input_signal = np.zeros((N, dim))
for i in range(dim):
    input_signal[rec[i][0] : rec[i][1], i] = 1.0
input_signal = input_signal + np.random.normal(0, 0.01, size=(N, dim))

logger = Logger()
logger.set_labels(["i"])
logger.extend_labels([f"in{i}" for i in range(dim)])
logger.extend_labels([f"out{i}" for i in range(dim)])

filt = Filter(n_points=M)
output_signal = np.empty((N, dim))
for i in range(N):
    output_signal[i] = filt.update(input_signal[i])
    logger.store_data(np.concatenate(([i], input_signal[i], output_signal[i])))
logger.dump()
