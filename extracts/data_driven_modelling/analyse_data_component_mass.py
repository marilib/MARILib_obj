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
    sqr_err = []
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

        if not np.isnan(dict["owe"]):
            sqr_err.append((dict["owe"]-owe_ref)**2)    # Store square of the errors

    df['OWE_ref'] = df['OWE']
    un['OWE_ref'] = un['OWE']

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    draw_reg(df, un, 'OWE_ref', 'OWE_mod', [[0,max(df['OWE_ref'])], [0,max(df['OWE_ref'])]], coloration)
    draw_hist(rer, 'OWE model - OWE reference')

    return sqr_err


def compare_owe_base_and_breakdown(df, ddm, factor, coloration, graph=True):
    """Compare OWE from data base with OWE computed through nominal mission simulation
    Results are drawn on graphs
    """
    owe_brk = []
    sqr_err = []
    rer = []
    power_system = ddm.default_power_system
    for i in df.index:
        airplane_name = df['name'][i]
        airplane_type = df['airplane_type'][i]
        engine_type = df['engine_type'][i]
        n_engine = float(df['n_engine'][i])
        max_power = float(df['max_power'][i])
        n_pax = float(df['n_pax'][i])
        nominal_range = float(df['nominal_range'][i])
        fuselage_width = float(df['fuselage_width'][i])
        total_length = float(df['total_length'][i])
        wing_area = float(df['wing_area'][i])
        wing_span = float(df['wing_span'][i])
        wing_sweep25 = float(df['wing_sweep25'][i])
        htp_area = float(df['HTP_area'][i])
        vtp_area = float(df['VTP_area'][i])
        mtow = df['MTOW'][i]
        mlw = df['MLW'][i]

        mfac = ddm.get_structure_mass_factor(nominal_range)

        fuselage_mass = factor['fuselage'] * mfac * 5.80*(np.pi*fuselage_width*total_length)**1.20   # Statistical regression versus fuselage built surface
        furnishing_mass = factor['furnishing'] * 10.*n_pax                                     # Furnishings mass
        op_item_mass = factor['op_item'] * 5.2*(n_pax*nominal_range*1e-6)                   # Operator items mass

        wing_ar = wing_span**2/wing_area
        A = 32*wing_area**1.1
        B = 3.5*wing_span**2 * mtow
        C = 1.1e-6*(1.+2.*wing_ar)/(1.+wing_ar)
        D = 0.127 * (wing_area/wing_span)
        E = np.cos(wing_sweep25)**2
        wing_mass = factor['wing'] * mfac * (A + (B*C)/(D*E))   # Shevell formula + high lift device regression

        htp_mass = factor['htp'] * mfac * 22.*htp_area
        vtp_mass = factor['vtp'] * mfac * 25.*vtp_area
        ldg_mass = factor['ldg'] * 0.015*mtow**1.03 + 0.012*mlw    # Landing gears
        system_mass = factor['system'] * 0.545*mtow**0.8

        power_density = {"piston":1000.,
                         "turboprop":3000.,
                         "turbofan":2000./ddm.get_engine_mass_factor(max_power)
                         }.get(engine_type)    # WARNING : INSTALLED POWER DENSITY
        engine_mass = factor['engine'] * (max_power*n_engine) / power_density

        # print(airplane_name, engine_mass)
        owe =  fuselage_mass + furnishing_mass + op_item_mass \
             + wing_mass + htp_mass + vtp_mass \
             + ldg_mass + system_mass \
             + engine_mass

        owe_ref = float(df['OWE'][i])
        rer.append((owe-owe_ref)/owe_ref)   # Store relative errors
        owe_brk.append(owe)

        if not np.isnan(owe):
            sqr_err.append((owe-owe_ref)**2)    # Store square of the errors

    df['OWE_ref'] = df['OWE']
    un['OWE_ref'] = un['OWE']

    df['OWE_brk'] = owe_brk
    un['OWE_brk'] = un['OWE']

    if graph:
        draw_reg(df, un, 'OWE_ref', 'OWE_brk', [[0,max(df['OWE_ref'])], [0,max(df['OWE_ref'])]], coloration)
        draw_hist(rer, 'OWE model - OWE reference')

    return sqr_err



# Read data
#-------------------------------------------------------------------------------------------------------------------
path_to_data_base = "All_Data_v4.xlsx"

df,un = read_db(path_to_data_base)

# Remove A380-800 row and reset index
df = df[df['name']!='A380-800'].reset_index(drop=True)

df = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
un = un.copy()


# #-------------------------------------------------------------------------------------------------------------------
# abs = "MTOW"
# ord = "OWE"
#
# # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
# # df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t
#
# # order = [1]
# order = [2, 1]
# dict_owe = do_regression(df, un, abs, ord, coloration, order)
#
#
# #-------------------------------------------------------------------------------------------------------------------
# phd = PhysicalData()
# ddm = DDM(phd)
#
# res = compare_owe_base_and_model(df, ddm, coloration)
# print("global model : ", res)
#

#-------------------------------------------------------------------------------------------------------------------
phd = PhysicalData()
ddm = DDM(phd)

factor = {'fuselage': 1.,
          'furnishing': 1.,
          'op_item': 1.,
          'wing': 1.,
          'htp': 1.,
          'vtp': 1.,
          'ldg': 1.,
          'system': 1.,
          'engine': 1.}


res = compare_owe_base_and_breakdown(df, ddm, factor, coloration, graph=True)
print(int(np.sqrt(sum(res))))

