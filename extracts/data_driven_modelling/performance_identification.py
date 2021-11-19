#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import matplotlib.pyplot as plt

import unit

from analyse_data import coloration, read_db, lin_lst_reg, draw_reg, subplots_by_varname,\
    draw_colored_cloud_on_axis, get_error, do_regression


#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis functions
#
#-----------------------------------------------------------------------------------------------------------------------

# Set font size
plt.rc('axes',labelsize=12,titlesize=20)
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.rc('legend',fontsize=12)




#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis
#
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "../../data/All_Data_v5.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    un1 = un.copy()

    # perform regressions
    #-------------------------------------------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "OWE"

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    # df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t

    order = [2, 1]
    dict_owe = do_regression(df1, un1, abs, ord, coloration, order)


    #----------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "total_power"                           # Name of the new column

    df[ord] = df['max_power']*df['n_engine']      # Add the new column to the dataframe
    un[ord] = un['max_power']                     # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1]
    dict = do_regression(df, un, abs, ord, coloration, order)


    # build polar
    #-------------------------------------------------------------------------------------------------------------------
    # nap = df.shape[0]
    #
    # for n in range(nap):
    #     aspect_ratio = df["wing_span"][n]**2 / df["wing_area"][n]
    #
    #     kind = (1.05 + (df["fuselage_width"][n] / df["wing_span"][n])**2)  / (np.pi * aspect_ratio)
    #
    #     lod = 17
    #     sfc = unit.convert_from("kg/daN/h", 0.6)
    #
    #     speed = df["cruise_speed"][n]

        # if speed < 1:

        # dist =



        # print(df["name"][n], "         ", speed)
