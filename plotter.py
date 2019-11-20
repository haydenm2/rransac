#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Class that plots incoming noisy data points alongside incoming best fit line estimates

class PLOTTER:
    def __init__(self):
        self.plots = []
        self.type = 0  # 0: LINE, 1: SCATTER
        self.iter = 0
        pass

    def register_plot(self, title, x_label, y_label, plot_type, number_channels):
        if plot_type == 0:
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            channels = []
            for i in range(number_channels):
                line, = ax.plot([], [])
                channels.append(line)
            self.plots.append(channels)
            self.iter += 1
        elif plot_type == 1:
            fig, ax = plt.subplots()
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            channels = []
            for i in range(number_channels):
                scatt = ax.scatter([], [], color='red', s=2)
                channels.append(scatt)

            self.plots.append(channels)
            self.iter += 1
        else:
            print("Invalid plot type")
        pass

    def access_plot(self, plot_number, channel_number):
        return self.plots[plot_number][channel_number]

    def visualize(self):
        plt.legend()
        plt.grid(True)
        plt.show()



