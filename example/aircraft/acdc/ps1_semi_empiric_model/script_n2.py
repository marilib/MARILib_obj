#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 20 20:20:20 2020
@author: DRUOT Thierry
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

file = "Biblio/Aircraft_data_v2.csv"

# Loading csv file
#------------------------------------------------------------------------------------------------------
data_frame = pandas.read_csv(file, delimiter = ";",skipinitialspace=True, header=None)

# Extracting info
#------------------------------------------------------------------------------------------------------
label = [el.strip() for el in data_frame.iloc[0,1:].values]     # Variable names
unit_ = [el.strip() for el in data_frame.iloc[1,1:].values]     # Figure units
name = [el.strip() for el in data_frame.iloc[2:,0].values]      # Aircraft names
data = data_frame.iloc[2:,1:].values                            # Data matrix

# Packaging figures into a dictionary
#------------------------------------------------------------------------------------------------------
dict = {}

n = len(name)   # the number of aircraft

# Data matrix is spread over a dictionary column by column
# Dictionary allows to address each column by the name of the corresponding variable
# for instance dict["Npax"] retrieves an array with all the realizations of Npax
for j in range(len(label)):
    dict[label[j]] = data[:,j].astype(np.float)

# First function
#------------------------------------------------------------------------------------------------------
param = dict["Npax"]*dict["Range"]

B = dict["Price"]

def residual(x):
    return x[0]*param**x[1] + x[2] - B

x0 = np.array([0.5, 0.9, 0.])

out = least_squares(residual, x0)


# Main results
#------------------------------------------------------------------------------------------------------
print(out.x)

AC = out.x[0]*param**out.x[1] + out.x[2]

res = np.sqrt(np.sum((AC-B)**2))

print("%.0f"%res)

# Graph
#------------------------------------------------------------------------------------------------------
plt.plot(B, AC, 'o', color="red", markersize=2)
plt.plot(B, B, 'green')
plt.grid(True)
plt.suptitle('Script_n2 : residual = '+"%.0f"%res, fontsize=14)
plt.ylabel('Approximations (M$)')
plt.xlabel('Experiments (M$)')
plt.show()

