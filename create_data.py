#!/usr/bin/env python3

import numpy as np

# Class that creates data based on simple 2D motion models (e.g. constant velocity)

class DATA:
    def __init__(self, targets=3, duration=10, dt=0.01):
        self.duration = duration
        self.dt = dt
        self.t = np.zeros((self.duration/self.dt))
        self.targets = targets

    def update(self):
        pass

