#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot, Weichang LYU
"""

import numpy as np
from scipy.optimize import fsolve, least_squares

import pandas
import matplotlib.pyplot as plt

from context import unit, math
from network.pico_design.design_model import Aircraft


# ======================================================================================================
# Packaging data for structural relation calibration
# ------------------------------------------------------------------------------------------------------
file = "../input_data/Aircraft_general_data_v2.csv"

# Loading csv file
# ------------------------------------------------------------------------------------------------------
data_frame = pandas.read_csv(file, delimiter = ";", skipinitialspace=True, header=None)

# Extracting info
# ------------------------------------------------------------------------------------------------------
label = [el.strip() for el in data_frame.iloc[0,1:].values]     # Variable names
unit_ = [el.strip() for el in data_frame.iloc[1,1:].values]     # Figure units
name = [el.strip() for el in data_frame.iloc[2:,0].values]      # Aircraft names
data = data_frame.iloc[2:,1:].values                            # Data matrix

# Packaging figures into a dictionary
# ------------------------------------------------------------------------------------------------------
data_dict = {}

n = len(name)   # the number of aircraft

# Data matrix is spread over a dictionary column by column
# Dictionary allows to address each column by the name of the corresponding variable
# for instance dict["Npax"] retrieves an array with all the realizations of Npax
for j in range(len(label)):
    data_dict[label[j]] = data[:,j].astype(np.float)

# ======================================================================================================
# Identifing structure model
# ------------------------------------------------------------------------------------------------------
param = data_dict["MTOW"]

A = np.vstack([param**2,param, np.ones(n)]).T                # Need to transpose the stacked matrix

B = data_dict["OWE"]

(C, res, rnk, s) = np.linalg.lstsq(A, B, rcond=None)

# Main results
#------------------------------------------------------------------------------------------------------
print(C)

AC = np.dot(A,C)

res = np.sqrt(np.sum((AC-B)**2))

print("%.0f"%res)

# Graph
#------------------------------------------------------------------------------------------------------
plt.plot(B, AC, 'o', color="red", markersize=2)
plt.plot(B, B, 'green')
plt.grid(True)
plt.suptitle('Structure : residual = '+"%.0f"%res, fontsize=14)
plt.ylabel('Approximations (kg)')
plt.xlabel('Experiments (kg)')
plt.savefig("calibration_structure_graph",dpi=500,bbox_inches = 'tight')
plt.show()

# Result
#------------------------------------------------------------------------------------------------------
def operating_empty_weight(mtow):
    owe = (-1.478e-07*mtow + 5.459e-01)*mtow + 8.40e+02
    return owe




# ======================================================================================================
# Packaging data for design relation calibration
# ------------------------------------------------------------------------------------------------------
file = "../input_data/Aircraft_general_data_v2.csv"

# Loading csv file
# ------------------------------------------------------------------------------------------------------
data_frame = pandas.read_csv(file, delimiter = ";", skipinitialspace=True, header=None)

# Extracting info
# ------------------------------------------------------------------------------------------------------
label = [el.strip() for el in data_frame.iloc[0,1:].values]     # Variable names
unit_ = [el.strip() for el in data_frame.iloc[1,1:].values]     # Figure units
name = [el.strip() for el in data_frame.iloc[2:,0].values]      # Aircraft names
data = data_frame.iloc[2:,1:].values                            # Data matrix

# Packaging figures into a dictionary
# ------------------------------------------------------------------------------------------------------
data_dict = {}

n = len(name)   # the number of aircraft

# Data matrix is spread over a dictionary column by column
# Dictionary allows to address each column by the name of the corresponding variable
# for instance dict["Npax"] retrieves an array with all the realizations of Npax
for j in range(len(label)):
    data_dict[label[j]] = data[:,j].astype(np.float)

# ======================================================================================================
# Identifing design model
# ------------------------------------------------------------------------------------------------------
ac = Aircraft()

B = data_dict["MTOW"]
n = len(B)

def model(x_in):
    R = []
    for j in range(n):
        ac.npax = data_dict["Npax"][j]
        ac.range = data_dict["Range"][j]*1000.
        ac.cruise_mach = data_dict["Speed"][j]
        #ac.design_aircraft(effr=x_in[0], mpax=x_in[1], kr=x_in[2])
        ac.design_aircraft()
        R.append(ac.mtow)
    return R

def residual(x_in):
    return model(x_in)-B

def objective(x_in):
    return 1.e4/np.sqrt(np.sum(model(x_in)-B)**2)

x0 = ac.lod / unit.convert_to("kg/daN/h", ac.sfc)

# x_out = x0

# x_out, y_out, rc = math.maximize_1d(x0, 0.1, [objective])

# Main results
#------------------------------------------------------------------------------------------------------
print("x0 = ",x0)

R = model(x0)

res = np.sqrt(np.sum((R-B)**2))

print("%.0f"%res)

# Graph
#------------------------------------------------------------------------------------------------------
plt.plot(B, R, 'o', color="red", markersize=2)
plt.plot(B, B, 'green')
plt.grid(True)
plt.suptitle('Design : residual = '+"%.0f"%res, fontsize=14)
plt.ylabel('Approximations (kg)')
plt.xlabel('Experiments (kg)')
plt.savefig("calibration_design_graph",dpi=500,bbox_inches = 'tight')
plt.show()

