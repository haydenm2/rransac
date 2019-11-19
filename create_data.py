#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Class that creates data based on simple 1D motion models (e.g. constant velocity)

class DATA:
    def __init__(self, targets=3, duration=10, dt=0.1):
        self.duration = duration
        self.dt = dt
        self.num_iterations = int(self.duration/self.dt)
        self.t = np.zeros(self.num_iterations)
        self.num_targets = targets
        self.num_noise_channels = 1
        self.num_channels = self.num_noise_channels + self.num_targets
        self.position_bound = 100
        self.x = np.zeros((self.num_channels, self.num_iterations))
        self.params = (np.random.rand(self.num_targets) * 2 - 1) * 5  # slope bounds between -5 and 5
        self.noise_cov = 0.5

    def create_points(self):
        self.x[:, 0] = np.random.rand(self.num_channels) * self.position_bound
        for i in range(1, self.num_iterations):
            self.t[i] = self.t[i-1] + self.dt
            self.propogate(i)
            self.insert_noise_data(i)
        pass

    def propogate(self, iter):
        self.x[0:self.num_targets, iter] = self.x[0:self.num_targets, iter-1] + self.params*self.dt + np.random.randn(self.num_targets) * np.sqrt(self.noise_cov)

    def insert_noise_data(self, iter):
        self.x[self.num_targets:self.num_channels, iter] = np.random.rand(self.num_noise_channels) * self.position_bound

    def visualize(self):
        for j in range(self.num_channels):
            plt.plot(self.t, self.x[j], 'r.')
        plt.title('Created Data Points')
        plt.xlabel('time (s)')
        plt.show()

