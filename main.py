#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from create_data import DATA
from rransac import RRANSAC
from plotter import PLOTTER


if __name__ == "__main__":
    live = False  # live plotting flag

    data = DATA()
    rransac = RRANSAC(data.noise_cov, model_type=1, ransac_update_type=0)
    plotter = PLOTTER()
    data.create_points()

    bound_extension = 0.2
    bounds = np.array((0, data.t[-1], 0-data.position_bound*(bound_extension), data.position_bound*(1 + bound_extension)))
    plot_types = np.hstack((np.ones((data.num_channels)), np.zeros(2)))
    plotter.register_plot('Noisy Data', 'time (s)', '', plot_types, data.num_channels + 1, bounds)
    plt.title('RANSAC Model Estimation')
    plt.xlabel('time (s)')
    plotter.access_plot(0, 0).set_label('data points')
    plotter.access_plot(0, data.num_channels).set_label('RANSAC Estimate')
    plotter.access_plot(0, data.num_channels+1).set_label('Full RANSAC Estimate')

    # # ----------------------- Visualize live point propogation -----------------------
    # for j in range(len(data.t)):
    #     for i in range(data.num_channels):
    #         try:
    #             plotter.access_plot(0, i).set_ydata(data.x[i, 0:j])
    #             plotter.access_plot(0, i).set_xdata(data.t[0:j])
    #         except:
    #             plotter.access_plot(0, i).set_offsets(np.hstack((data.t[0:j].reshape(-1, 1), data.x[i, 0:j].reshape(-1, 1))))
    #     plotter.visualize()

    # # ----------------------- Visualize final point set -----------------------
    # for i in range(data.num_channels):
    #     try:
    #         plotter.access_plot(0, i).set_ydata(data.x[i, :])
    #         plotter.access_plot(0, i).set_xdata(data.t)
    #     except:
    #         plotter.access_plot(0, i).set_offsets(np.hstack((data.t.reshape(-1, 1), data.x[i, :].reshape(-1, 1))))
    # plotter.visualize()
    # plt.show()

    for k in range(len(data.t)):
        rransac.Update(data.get_time(), data.get_next_points())
        if live:
            for l in range(data.num_channels):
                plotter.access_plot(0, l).set_offsets(np.hstack((rransac.t.reshape(-1, 1), rransac.points[l, :].reshape(-1, 1))))
            plotter.access_plot(0, data.num_channels).set_ydata(rransac.model[:])
            plotter.access_plot(0, data.num_channels).set_xdata(rransac.model_t[:])
            plotter.visualize()
    if not live:
        for l in range(data.num_channels):
            plotter.access_plot(0, l).set_offsets(np.hstack((rransac.t.reshape(-1, 1), rransac.points[l, :].reshape(-1, 1))))
        plotter.access_plot(0, data.num_channels).set_ydata(rransac.model[:])
        plotter.access_plot(0, data.num_channels).set_xdata(rransac.model_t[:])
    c = rransac.GenerateHypothesis(rransac.model_t, rransac.model, rransac.model_type)
    y_est = c[0]*rransac.model_t + c[1]
    plotter.access_plot(0, data.num_channels+1).set_ydata(y_est)
    plotter.access_plot(0, data.num_channels+1).set_xdata(rransac.model_t[:])
    plotter.visualize()
    plt.show()




