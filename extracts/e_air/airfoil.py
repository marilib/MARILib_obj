#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Thomas Ligeois, Thierry DRUOT
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, interp2d

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
x_airfoil = chord * np.hstack((x_vec, x_vec_flip))
y_airfoil = chord * np.hstack((y_ext,y_int))

# Packaging data for CAD
n = 2*n_val - 1
airfoil = np.vstack([np.ones(n), 1+np.arange(n), x_airfoil, y_airfoil, np.zeros(n)]).T

# Store in a file
file_name = "airfoil_data.txt"
with open(file_name, 'w') as f:
    for i in range(n):
        f.write("{:1.0f} {:3.0f} {:.8f} {:.8f} {:.8f}".format(*airfoil[i,:]))
        f.write('\n')
f.close()

# # Print airfoil data
# for i in range(n):
#     print("{:1.0f} {:3.0f} {:.8f} {:.8f} {:.8f}".format(*airfoil[i,:]))
#
# # Plot airfoil data
# plt.plot(x_airfoil,y_airfoil)
# plt.show()


# Packaging data for thermal transfer model
x_vec_2 = np.flip(x_vec)
y_ext_2 = np.flip(y_ext)
y_int_2 = -5.0*er_int*(0.2969*x_vec_2**0.5 - 0.1260*x_vec_2 - 0.3516*x_vec_2**2 + 0.2843*x_vec_2**3 - 0.1015*x_vec_2**4)

s_ext_2 = [0]
s_int_2 = [0]
for i in range(n_val-1):
    s_ext_2.append(s_ext_2[-1] + np.sqrt((x_vec_2[i+1]-x_vec_2[i])**2 + (y_ext_2[i+1]-y_ext_2[i])**2))
    s_int_2.append(s_int_2[-1] + np.sqrt((x_vec_2[i+1]-x_vec_2[i])**2 + (y_int_2[i+1]-y_int_2[i])**2))

table = np.vstack([chord*x_vec_2, chord*y_ext_2, chord*y_int_2, chord*np.array(s_ext_2), chord*np.array(s_int_2)]).T

df1 = pd.DataFrame(table, columns=["x", "y_ext", "y_int", "s_ext", "s_int"])

# print(tabulate(table))


file_name = "htc_mac_0p183.csv"
df = pd.read_csv(file_name, delimiter = ";")

h_ext_0 = interp1d(df["x_ext_aoa_0"], df["h_ext_aoa_0"], kind="cubic", fill_value='extrapolate')
h_int_0 = interp1d(df["x_int_aoa_0"], df["h_int_aoa_0"], kind="cubic", fill_value='extrapolate')

h_ext_2p5 = interp1d(df["x_ext_aoa_2p5"], df["h_ext_aoa_2p5"], kind="cubic", fill_value='extrapolate')
h_int_2p5 = interp1d(df["x_int_aoa_2p5"], df["h_int_aoa_2p5"], kind="cubic", fill_value='extrapolate')

h_ext_5 = interp1d(df["x_ext_aoa_5"], df["h_ext_aoa_5"], kind="cubic", fill_value='extrapolate')
h_int_5 = interp1d(df["x_int_aoa_5"], df["h_int_aoa_5"], kind="cubic", fill_value='extrapolate')

h_ext_8 = interp1d(df["x_ext_aoa_8"], df["h_ext_aoa_8"], kind="cubic", fill_value='extrapolate')
h_int_8 = interp1d(df["x_int_aoa_8"], df["h_int_aoa_8"], kind="cubic", fill_value='extrapolate')

aoa_vec = unit.convert_from("deg", np.array([0, 2.5, 5, 8]))

x_abs = df1["x"]
s_abs_ext = df1["s_ext"]
s_abs_int = df1["s_int"]

h_ext = np.array([[float(h_ext_0(x)) for x in x_abs],
                  [float(h_ext_2p5(x)) for x in x_abs],
                  [float(h_ext_5(x)) for x in x_abs],
                  [float(h_ext_8(x)) for x in x_abs]])

h_int = np.array([[float(h_int_0(x)) for x in x_abs],
                  [float(h_int_2p5(x)) for x in x_abs],
                  [float(h_int_5(x)) for x in x_abs],
                  [float(h_int_8(x)) for x in x_abs]])


h_ext_fct = interp2d(s_abs_ext, aoa_vec, h_ext, kind="cubic")
h_int_fct = interp2d(s_abs_int, aoa_vec, h_int, kind="cubic")


