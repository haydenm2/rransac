#!/usr/bin/env python3

import numpy as np

# Class implementing the Recursive Random Sampling and Consensus (RRANSAC) algorithm which takes in iterations of noisy data and extracts viable motion models.

class RRANSAC:
    def __init__(self, sensor_covariance, model_type=1):
        self.model_type = model_type
        self.points = []
        self.t = []
        self.init = True
        self.ransac_iterations = 100
        self.error_threshold = 4 * np.sqrt(sensor_covariance)  #
        self.stopping_criteria = 0.7
        self.ransac_window = 10
        self.minimum_subset = 5
        self.points_per_iteration = 0
        self.consensus_set = []

    def RANSAC(self, xpoints, ypoints):
        best_consensus = []
        for j in range(self.ransac_iterations):
            # Generate subset
            subset = np.zeros((1, 0))
            for i in range(len(ypoints[0])):
                np.random.shuffle(ypoints[:, i])
            subset = ypoints[0, :]

            # Generate Hypothesis
            y = subset.reshape((-1, 1))
            A = np.hstack((xpoints.reshape((-1, 1)), np.ones((len(xpoints), 1))))
            A_T = A.transpose()
            c = np.linalg.inv(A_T @ A) @ A_T @ y

            # Compute consensus set
            consensus = []
            for k in range(len(ypoints[0])):
                e_ind = np.argmin(np.abs((c[0]*xpoints[k] + c[1]) - ypoints[:, k]))
                e = np.abs((c[0]*xpoints[k] + c[1]) - ypoints[e_ind, k])
                if e < self.error_threshold:
                    consensus = np.hstack((consensus, ypoints[e_ind, k]))
            if len(consensus) > len(best_consensus):
                best_consensus = consensus
                if len(best_consensus) > self.stopping_criteria * len(xpoints):
                    break
        self.consensus_set = best_consensus

    def AppendData(self, time, data):
        if self.init:
            self.points = data.reshape(-1, 1)
            self.init = False
            self.t = time
            self.points_per_iteration = len(data)
        else:
            self.points = np.hstack((self.points, data.reshape(-1, 1)))
            self.t = np.hstack((self.t, time))

    def Update(self, time, data):
        # Update point set
        self.AppendData(time, data)
        if len(self.points[0]) >= self.ransac_window:
            self.RANSAC(self.t, self.points)




