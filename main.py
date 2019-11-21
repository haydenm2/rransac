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
    plot_types = np.hstack((np.ones((data.num_channels)), np.zeros(3)))
    plotter.register_plot('Noisy Data', 'time (s)', '', plot_types, bounds)
    plt.title('RANSAC Model Estimation')
    plt.xlabel('time (s)')
    plotter.access_plot(0, 0).set_label('data points')
    plotter.access_plot(0, data.num_channels).set_label('RANSAC Point Estimate')
    plotter.access_plot(0, data.num_channels+1).set_label('Full RANSAC Model Estimate')
    plotter.access_plot(0, data.num_channels+2).set_label('Actual Model')

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

    # plot overall linear model estimate
    c = rransac.GenerateHypothesis(rransac.model_t, rransac.model, rransac.model_type)
    y_est = c[0]*rransac.model_t + c[1]
    plotter.access_plot(0, data.num_channels+1).set_ydata(y_est)
    plotter.access_plot(0, data.num_channels+1).set_xdata(rransac.model_t[:])
    plotter.visualize()

    # plot true linear model
    y_true = data.params[0] * rransac.model_t + data.x[0, 0]
    plotter.access_plot(0, data.num_channels + 2).set_ydata(y_true)
    plotter.access_plot(0, data.num_channels + 2).set_xdata(rransac.model_t[:])
    plotter.visualize()

    # plot error
    y_err = y_est - y_true
    yp_err = rransac.model[:] - y_true
    err_bounds = np.array((0, data.t[-1], -3, 3))
    plotter.register_plot('Error', 'time (s)', '', [0, 0], err_bounds)
    plt.title('RANSAC Model Estimation Error')
    plt.xlabel('time (s)')
    plotter.access_plot(1, 0).set_label('RANSAC Full Model Error')
    plotter.access_plot(1, 0).set_ydata(y_err)
    plotter.access_plot(1, 0).set_xdata(rransac.model_t[:])
    plotter.access_plot(1, 1).set_label('RANSAC Point Model Error')
    plotter.access_plot(1, 1).set_ydata(yp_err)
    plotter.access_plot(1, 1).set_xdata(rransac.model_t[:])
    plotter.visualize()

    # plot parameter estimates vs truth
    p0_ransac = data.params[0] * np.ones(np.shape(rransac.model_t[:]))
    p0_true = c[0] * np.ones(np.shape(rransac.model_t[:]))
    param1_bounds = np.array((0, data.t[-1], -5, 5))
    plotter.register_plot('', 'time (s)', '', [0, 0, 0, 0], param1_bounds)
    plt.title('RANSAC Model Parameter 1 Estimate vs Truth')
    plt.xlabel('time (s)')
    plotter.access_plot(2, 0).set_label('RANSAC Model Param 1')
    plotter.access_plot(2, 0).set_ydata(p0_ransac)
    plotter.access_plot(2, 0).set_xdata(rransac.model_t[:])
    plotter.access_plot(2, 1).set_label('True Model Param 1')
    plotter.access_plot(2, 1).set_ydata(p0_true)
    plotter.access_plot(2, 1).set_xdata(rransac.model_t[:])
    plotter.visualize()

    # plot parameter estimates vs truth
    p1_ransac = data.x[0, 0] * np.ones(np.shape(rransac.model_t[:]))
    p1_true = c[1] * np.ones(np.shape(rransac.model_t[:]))
    param2_bounds = np.array((0, data.t[-1], data.x[0, 0]-5, data.x[0, 0]+5))
    plotter.register_plot('', 'time (s)', '', [0, 0, 0, 0], param2_bounds)
    plt.title('RANSAC Model Parameter 2 Estimate vs Truth')
    plt.xlabel('time (s)')
    plotter.access_plot(3, 0).set_label('RANSAC Model Param 2')
    plotter.access_plot(3, 0).set_ydata(p1_ransac)
    plotter.access_plot(3, 0).set_xdata(rransac.model_t[:])
    plotter.access_plot(3, 1).set_label('True Model Param 2')
    plotter.access_plot(3, 1).set_ydata(p1_true)
    plotter.access_plot(3, 1).set_xdata(rransac.model_t[:])
    plotter.visualize()

    # plot param error
    p0_err = c[0] - data.params[0]
    p1_err = c[1] - data.x[0, 0]
    param_err_bounds = np.array((0, data.t[-1], -1, 1))
    plotter.register_plot('Error', 'time (s)', '', [0, 0], param_err_bounds)
    plt.title('RANSAC Model Parameter Estimation Error')
    plt.xlabel('time (s)')
    plotter.access_plot(4, 0).set_label('RANSAC Model Param 1 Error')
    plotter.access_plot(4, 0).set_ydata(p0_err)
    plotter.access_plot(4, 0).set_xdata(rransac.model_t[:])
    plotter.access_plot(4, 1).set_label('RANSAC Model Param 2 Error')
    plotter.access_plot(4, 1).set_ydata(p1_err)
    plotter.access_plot(4, 1).set_xdata(rransac.model_t[:])
    plotter.visualize()

    plt.show()




