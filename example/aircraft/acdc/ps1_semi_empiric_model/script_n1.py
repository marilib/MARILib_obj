#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 20 20:20:20 2020
@author: DRUOT Thierry
"""

import pandas
import numpy as np
import matplotlib.pyplot as plt

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
# param = ...

A = np.vstack([param, np.ones(n)]).T                # Need to transpose the stacked matrix

B = dict["Price"]

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
plt.suptitle('Script_n1 : residual = '+"%.0f"%res, fontsize=14)
plt.ylabel('Approximations (M$)')
plt.xlabel('Experiments (M$)')
plt.show()

