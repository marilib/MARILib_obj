#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Thomas Ligeois, Thierry DRUOT
"""

import numpy as np
from scipy.optimize import fsolve

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

import unit, utils
from physical_data import PhysicalData

# Using 4 digits NACA airfoil series

# Meshing data
theta_start = unit.rad_deg(20)
theta_stop = unit.rad_deg(180)
n_val = 100

# Airfoil data
chord = 1.
er = 0.14       # Thickness to chord ratio
der = 0.01      # Delta t/c

er_ext = er + der      # Relative thickness for upper surface
er_int = er - der      # Relative thickness for lower surface

theta = np.linspace(theta_start, theta_stop, n_val)
delta = np.cos(theta_start) - np.cos(theta_stop)
x_list = (1 + np.cos(theta)) / delta

x_vec = np.array(x_list)
y_ext = 5.0*er_ext*(0.2969*x_vec**0.5 - 0.1260*x_vec - 0.3516*x_vec**2 + 0.2843*x_vec**3 - 0.1015*x_vec**4)

x_vec_flip = np.flip(x_vec[:-1])    # Remove last point and flip
y_int = -5.0*er_int*(0.2969*x_vec_flip**0.5 - 0.1260*x_vec_flip - 0.3516*x_vec_flip**2 + 0.2843*x_vec_flip**3 - 0.1015*x_vec_flip**4)

# Catenating and scaling data
x = chord * np.hstack((x_vec, x_vec_flip))
y = chord * np.hstack((y_ext,y_int))

# Packaging data for CAD
n = 2*n_val - 1
table = np.vstack([np.ones(n), 1+np.arange(n), x, y, np.zeros(n)]).T

# Print airfoil data
for i in range(n):
    print("{:1.0f} {:3.0f} {:.8f} {:.8f} {:.8f}".format(*table[i,:]))

# Store in a file
file_name = "airfoil_data.txt"
with open(file_name, 'w') as f:
    for i in range(n):
        f.write("{:1.0f} {:3.0f} {:.8f} {:.8f} {:.8f}".format(*table[i,:]))
        f.write('\n')
f.close()

# Plot airfoil data
plt.plot(x,y)
plt.show()

