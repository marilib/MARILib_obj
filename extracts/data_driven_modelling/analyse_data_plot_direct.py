#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, least_squares

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

import unit
import utils
from physical_data import PhysicalData
from models import DDM

from analyse_data import coloration, read_db, draw_reg, draw_hist, do_regression


#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis
#
#-----------------------------------------------------------------------------------------------------------------------


# Read data
#-------------------------------------------------------------------------------------------------------------------
path_to_data_base = "All_Data_v4.xlsx"

df,un = read_db(path_to_data_base)

# Remove A380-800 row and reset index
df = df[df['name']!='A380-800'].reset_index(drop=True)


#----------------------------------------------------------------------------------
abs = "n_pax"
ord = "fuselage_width"

dict = draw_reg(df, un, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "wing_span"
ord = "VTP_area"

dict = draw_reg(df, un, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "wing_span"
ord = "wing_area/wing_span"

df[ord] = df['wing_area']/df['wing_span']    # Add the new column to the dataframe
un[ord] = "m"                                # Add its unit

dict = draw_reg(df, un, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "wing_sweep25"
ord = "cruise_speed"

df1 = df[df['cruise_speed']<1].reset_index(drop=True).copy()    # Remove
un1 = un.copy()
un1[ord] = "mach"

dict = draw_reg(df1, un1, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "cruise_speed"
ord = "max_speed"

df1 = df[df['cruise_speed']>1].reset_index(drop=True).copy()    # Remove
un1 = un.copy()

dict = draw_reg(df1, un1, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "cruise_speed"
ord = "max_speed"

df1 = df[df['cruise_speed']<1].reset_index(drop=True).copy()    # Remove
un1 = un.copy()

un1[abs] = "mach"
un1[ord] = "mach"

dict = draw_reg(df1, un1, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "total_length*fuselage_width"
ord = "n_pax"

df[abs] = df['total_length']*df['fuselage_width']   # Add the new column to the dataframe
un[abs] = "m2"                                      # Add its unit

dict = draw_reg(df, un, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "total_length"
ord = "total_height"

dict = draw_reg(df, un, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "MTOW"
ord = "OWE"

dict = draw_reg(df, un, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "OWE"
ord = "OWE+n_pax*100"

df[ord] = df['OWE'] + df['n_pax'].multiply(100.)   # Add the new column to the dataframe
un[ord] = "m2"                                      # Add its unit

dict = draw_reg(df, un, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "n_pax*nominal_range(km)"
ord = "max_fuel"

df[abs] = df['n_pax'] * df['nominal_range'].multiply(1e-3)   # Add the new column to the dataframe
un[abs] = "seat.km"                                      # Add its unit

dict = draw_reg(df, un, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "rotor_diameter+fuselage_width"
ord = "engine_y_arm"

df[abs] = df['rotor_diameter'] + df['fuselage_width']   # Add the new column to the dataframe
un[abs] = "m"                                      # Add its unit

dict = draw_reg(df, un, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "MTOW"
ord = "max_thrust*n_engine"

df1 = df[df['max_thrust'].notnull()].reset_index(drop=True).copy()    # Remove
un1 = un.copy()
df1[ord] = df1['max_thrust'] * df['n_engine']   # Add the new column to the dataframe
un1[ord] = "kN"

dict = draw_reg(df1, un1, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "MTOW"
ord = "max_power*n_engine"

df1 = df[df['max_thrust'].isnull()].reset_index(drop=True).copy()    # Remove
un1 = un.copy()
df1[ord] = df1['max_power'] * df1['n_engine']   # Add the new column to the dataframe
un1[ord] = "kW"

dict = draw_reg(df1, un1, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "MTOW"
ord = "generic_max_thrust*n_engine"

df[ord] = df['max_power'] * df['n_engine']   # Add the new column to the dataframe
un[ord] = "kW"

dict = draw_reg(df, un, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "approach_speed"
ord = "lfl"

dict = draw_reg(df, un, abs, ord, [[],[]], coloration, leg_loc="upper left")

#----------------------------------------------------------------------------------
abs = "MLW/wing_area"
ord = "approach_speed"

df[abs] = (df['MLW'] / df['wing_area'])   # Add the new column to the dataframe
un[abs] = "kg/m2"                                      # Add its unit

dict = draw_reg(df, un, abs, ord, [[],[]], coloration)

#----------------------------------------------------------------------------------
abs = "MTOW**2/(max_power*wing_area)"
ord = "tofl"

df[abs] = df['MTOW']**2 / (df['max_power'] * df['n_engine'] * df['wing_area'])   # Add the new column to the dataframe
un[abs] = "std"                                      # Add its unit

dict = draw_reg(df, un, abs, ord, [[],[]], coloration)

