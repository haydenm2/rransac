#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from create_data import DATA
from rransac import RRANSAC
from plotter import PLOTTER


if __name__ == "__main__":
    data = DATA()
    rransac = RRANSAC()
    plotter = PLOTTER()
    data.create_points()

    bound_extension = 0.1
    bounds = np.array((0, data.t[-1], 0-data.position_bound*(bound_extension), data.position_bound*(1 + bound_extension)))
    plotter.register_plot('Noisy Data', 'time (s)', '', 1, data.num_channels, bounds)
    plt.title('Created Data Points')
    plt.xlabel('time (s)')
    plotter.access_plot(0, 0).set_label('data points')
    for i in range(data.num_channels):
        try:
            plotter.access_plot(0, i).set_ydata(data.x[i, :])
            plotter.access_plot(0, i).set_xdata(data.t)
        except:
            plotter.access_plot(0, i).set_offsets(np.hstack((data.t.reshape(-1, 1), data.x[i, :].reshape(-1, 1))))
    plotter.visualize()

    data.visualize()

