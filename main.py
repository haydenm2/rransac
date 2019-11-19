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
    data.visualize()

