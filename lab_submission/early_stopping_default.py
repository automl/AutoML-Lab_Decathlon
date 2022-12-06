import math
import time

import numpy as np


class EarlyStopping:
    def __init__(self, metadata, run_time):
        self.patience = self._get_patience(metadata)
        self.run_time = run_time
        self.start_time = time.time()
        self.lowest_cost = np.inf
        self.epochs_since_new_min = 0

    def _get_patience(self, metadata):
        n_datapoints = metadata.size()
        timeseries, channel, row, col = metadata.get_tensor_shape()
        n_features = timeseries * channel * row * col
        if n_datapoints < 3000:
            patience = 20.
        elif n_datapoints < 10000:
            patience = 12.
        elif n_datapoints > 200000:
            patience = 4.
        else:
            patience = 8.

        if n_features > 10000:
            patience -= 2.
        elif n_features < 1000:
            patience += 2.

        return math.ceil(patience * 2.5)

    def running(self, cost):
        running = True
        if cost < self.lowest_cost:
            self.lowest_cost = cost
            self.epochs_since_new_min = 0
        else:
            self.epochs_since_new_min += 1
        # runtime check
        if time.time() - self.start_time > self.run_time:
            running = False
        # cost check
        if self.epochs_since_new_min > self.patience:
            running = False
        return running
