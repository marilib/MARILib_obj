#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot, Weichang LYU
"""

import numpy as np

import pandas
import matplotlib.pyplot as plt

from marilib.context import unit, math
from marilib.network.pico_design.design_model import Aircraft


def get_data(file_name):
    data_frame = pandas.read_csv(file_name, delimiter = ";", skipinitialspace=True, header=None)

    label = [el.strip() for el in data_frame.iloc[0,1:].values]     # Variable names
    unit_ = [el.strip() for el in data_frame.iloc[1,1:].values]     # Figure units
    name = [el.strip() for el in data_frame.iloc[2:,0].values]      # Aircraft names
    data = data_frame.iloc[2:,1:].values                            # Data matrix

    data_dict = {"Name":name}
    for j in range(len(label)):
        data_dict[label[j]] = unit.convert_from(unit_[j], data[:, j].astype(np.float))
    return data_dict


if __name__ == '__main__':

    # ======================================================================================================
    # Identify structure model
    # ------------------------------------------------------------------------------------------------------
    data_dict = get_data("../input_data/Aircraft_general_data_v2.csv")

    param = data_dict["MTOW"]

    n = len(param)   # the number of aircraft

    A = np.vstack([param**2,param, np.ones(n)]).T                # Need to transpose the stacked matrix

    B = data_dict["OWE"]

    (C, res, rnk, s) = np.linalg.lstsq(A, B, rcond=None)

    # Main results
    #------------------------------------------------------------------------------------------------------
    print("")
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
    # Identify design model
    # ------------------------------------------------------------------------------------------------------
    ac = Aircraft()

    B = data_dict["MTOW"]
    n = len(B)

    def model(x_in):
        R = []
        for j in range(n):
            ac.npax = data_dict["Npax"][j]
            ac.range = x_in*data_dict["WikiRange"][j]
            ac.cruise_mach = data_dict["Speed"][j]
            ac.design_aircraft()
            R.append(ac.mtow)
        return R

    def residual(x_in):
        return model(x_in)-B

    def objective(x_in):
        return 1.e4/np.sqrt(np.sum(model(x_in)-B)**2)

    x0 = 1.

    # x_out = x0

    x_out, y_out, rc = math.maximize_1d(x0, 0.05, [objective])

    # Main results
    #------------------------------------------------------------------------------------------------------
    print("")
    print("x0 = ",x0)
    print("x_out = ",x_out)

    R = model(x_out)

    res = np.sqrt(np.sum((R-B)**2))

    print("%.0f"%res)

    # for j in range(n):
    #     print("%.0f"%B[j], "   ", "%.0f"%R[j] )

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



    # ======================================================================================================
    # Identify design range
    # ------------------------------------------------------------------------------------------------------
    ac = Aircraft()

    B = data_dict["MTOW"]
    n = len(B)

    def fct(dist):
        ac.range = dist
        ac.design_aircraft()
        res = B[j] - ac.mtow
        return 1./(1.+res**2)

    print("")
    for j in range(n):
        ac.npax = data_dict["Npax"][j]
        ac.range = data_dict["WikiRange"][j] * x_out
        ac.cruise_mach = data_dict["Speed"][j]
        ac.design_aircraft()

        x0 = ac.range

        range, y_out, rc = math.maximize_1d(x0, 1.e4, [fct])

        print(j,"    ","%.0f"%(data_dict["WikiRange"][j]*x_out/1000.),"    ","%.0f"%(range/1000.),"  ",data_dict["Name"][j])



