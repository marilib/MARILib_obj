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



def compare_owe_base_and_model(df, ddm, coloration):
    """Compare OWE from data base with OWE computed through nominal mission simulation
    Results are drawn on graphs
    """
    owe = []
    rer = []
    power_system = ddm.default_power_system
    for i in df.index:
        npax = float(df['n_pax'][i])
        mtow = float(df['MTOW'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        max_power = float(df['max_power'][i])
        airplane_type = df['airplane_type'][i]
        power_system["engine_type"] = df['engine_type'][i]

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        energy_storage = 0.
        dict = ddm.owe_structure(mtow, max_power, energy_storage, power_system)

        owe_ref = float(df['OWE'][i])
        rer.append((dict["owe"]-owe_ref)/owe_ref)   # Store relative error
        owe.append(dict["owe"])

    df['OWE_ref'] = df['OWE']
    un['OWE_ref'] = un['OWE']

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    draw_reg(df, un, 'OWE_ref', 'OWE_mod', [[0,max(df['OWE_ref'])], [0,max(df['OWE_ref'])]], coloration)
    draw_hist(rer, 'OWE model - OWE reference')


# factor = {'fuselage': 5,
#           'wing': 32.,
#           'htp': 22.,
#           'vtp': 25,
#           'ldg': 0.2}
#
# def breakdown(df,factor):
#     fuselage_mass = ((df['fuselage_width'].multiply(np.pi))*df['total_length']).multiply(factor['fuselage'])
#     wing_mass = df['wing_area'].multiply(factor['wing'])
#     htp_mass = df['HTP_area'].multiply(factor['htp'])
#     vtp_mass = df['VTP_area'].multiply(factor['vtp'])
#     ldg_mass = df['MTOW'].multiply(factor['ldg'])
#     df[]
#     if initial_power_system["engine_type"]==self.piston:
#         initial_engine_mass = max_power / self.piston_eng_pw_density
#         initial_engine_mass += max_power / self.propeller_pw_density
#     elif initial_power_system["engine_type"]==self.turboprop:
#         initial_engine_mass = max_power / self.turboprop_pw_density
#         initial_engine_mass += max_power / self.propeller_pw_density
#     elif initial_power_system["engine_type"]==self.turbofan:
#         initial_engine_mass = max_power / self.turbofan_pw_density
#     else:
#         raise Exception("initial power system - engine type is not allowed")


# Read data
#-------------------------------------------------------------------------------------------------------------------
path_to_data_base = "All_Data_v4.xlsx"

df,un = read_db(path_to_data_base)

# Remove A380-800 row and reset index
df = df[df['name']!='A380-800'].reset_index(drop=True)

df = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
un = un.copy()


#-------------------------------------------------------------------------------------------------------------------
abs = "MTOW"
ord = "OWE"

# print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
# df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t

# order = [1]
order = [2, 1]
dict_owe = do_regression(df, un, abs, ord, coloration, order)


#-------------------------------------------------------------------------------------------------------------------
phd = PhysicalData()
ddm = DDM(phd)

compare_owe_base_and_model(df, ddm, coloration)
