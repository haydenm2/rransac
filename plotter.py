#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Class that plots incoming noisy data points alongside incoming best fit line estimates

class PLOTTER:
    def __init__(self):
        self.plots = []
        self.type = 0  # 0: LINE | 1: SCATTER
        self.iter = 0
        pass

    def register_plot(self, title, x_label, y_label, plot_types, number_channels, bounds):
        fig, ax = plt.subplots()
        channels = [ax]
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(bounds[0:2])
        ax.set_ylim(bounds[2:4])
        for j in range(len(plot_types)):
            if plot_types[j] == 0:
                line, = ax.plot([], [])
                channels.append(line)
            elif plot_types[j] == 1:
                scatt = ax.scatter([], [], color='red', s=2)
                channels.append(scatt)
            else:
                print("Invalid plot type")
        self.plots.append(channels)
        self.iter += 1
        pass

    def access_plot(self, plot_number, channel_number):
        return self.plots[plot_number][channel_number+1]  # skip the plot axis (or access with an input of -1

    def visualize(self):
        plt.legend()
        plt.grid(True)
        plt.pause(0.01)
        # plt.show()



