#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot
"""

import numpy as np

from scipy.optimize import fsolve, least_squares

import matplotlib.pyplot as plt

import pickle

#import network.flight_data.data_explorer as data_explorer

from context import unit, math

from network.pico_design.design_model import Aircraft, Fleet



def load_matrix_from_file(matrix_file_name):
    with open(matrix_file_name, 'rb') as f:
        data_matrix = pickle.load(f)
    return data_matrix

# ======================================================================================================
# Test
# ------------------------------------------------------------------------------------------------------
# ac = Aircraft(npax=150., range=unit.m_NM(3000.), mach=0.78)
#
# ac.payload_range()

# ======================================================================================================
# Fleet evaluation
# ------------------------------------------------------------------------------------------------------
ac0 = Aircraft(npax=100., range=unit.m_km(4000.), mach=0.76)    # Short
ac1 = Aircraft(npax=150., range=unit.m_km(5000.), mach=0.78)    # Short
ac2 = Aircraft(npax=250., range=unit.m_km(8000.), mach=0.82)    # Medium
ac3 = Aircraft(npax=350., range=unit.m_km(12000.), mach=0.85)   # Long
ac4 = Aircraft(npax=500., range=unit.m_km(14000.), mach=0.85)   # Very long

fleet = Fleet([ac0, ac1, ac2, ac3, ac4])

# ac0 = Aircraft(npax=117., range=unit.m_km(6000.), mach=0.78)    # A318
# ac1 = Aircraft(npax=134., range=unit.m_km(6500.), mach=0.78)    # A319
# ac2 = Aircraft(npax=164., range=unit.m_km(6000.), mach=0.78)    # A320
# ac3 = Aircraft(npax=199., range=unit.m_km(5600.), mach=0.78)    # A321
# ac4 = Aircraft(npax=140., range=unit.m_km(6600.), mach=0.78)    # A319neo
# ac5 = Aircraft(npax=165., range=unit.m_km(6200.), mach=0.78)    # A320neo
# ac6 = Aircraft(npax=190., range=unit.m_km(7000.), mach=0.78)    # A321neo
# ac7 = Aircraft(npax=246., range=unit.m_km(11500.), mach=0.82)   # A330-200
# ac8 = Aircraft(npax=277., range=unit.m_km(12000.), mach=0.82)   # A330-300
# ac9 = Aircraft(npax=257., range=unit.m_km(13600.), mach=0.82)   # A330-800neo
# ac10 = Aircraft(npax=287., range=unit.m_km(12000.), mach=0.82)   # A330-900neo
# ac11 = Aircraft(npax=280., range=unit.m_km(13000.), mach=0.85)   # A350-800neo
# ac12 = Aircraft(npax=326., range=unit.m_km(13500.), mach=0.85)   # A350-900neo
# ac13 = Aircraft(npax=400., range=unit.m_km(13000.), mach=0.85)   # A350-1000neo
# ac14 = Aircraft(npax=700., range=unit.m_km(14000.), mach=0.85)   # A380-800
#
# fleet = Fleet([ac0, ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8, ac9, ac10, ac11, ac12, ac13, ac14])
#
# ac20 = Aircraft(npax=130., range=unit.m_km(5500.), mach=0.785)    # 737-700
# ac21 = Aircraft(npax=160., range=unit.m_km(5500.), mach=0.785)    # 737-800
# ac22 = Aircraft(npax=177., range=unit.m_km(5500.), mach=0.785)    # 737-900ER
# ac23 = Aircraft(npax=138., range=unit.m_km(5800.), mach=0.79)    # 737 MAX 7
# ac24 = Aircraft(npax=162., range=unit.m_km(6100.), mach=0.79)    # 737 MAX 8
# ac25 = Aircraft(npax=178., range=unit.m_km(6200.), mach=0.79)   # 737 MAX 9
# ac26 = Aircraft(npax=188., range=unit.m_km(6700.), mach=0.79)   # 737 MAX 10
# ac27 = Aircraft(npax=261., range=unit.m_km(9500.), mach=0.80)   # 767-300ER
# ac28 = Aircraft(npax=242., range=unit.m_km(13000.), mach=0.85)   # 787-8
# ac29 = Aircraft(npax=290., range=unit.m_km(12700.), mach=0.85)   # 787-9
# ac30 = Aircraft(npax=330., range=unit.m_km(10700.), mach=0.85)   # 787-10
# ac31 = Aircraft(npax=300., range=unit.m_km(14500.), mach=0.84)   # 777-8
# ac32 = Aircraft(npax=400., range=unit.m_km(13500.), mach=0.84)   # 777-9
# ac33 = Aircraft(npax=317., range=unit.m_km(14200.), mach=0.84)   # 777-200LR
# ac34 = Aircraft(npax=350., range=unit.m_km(13000.), mach=0.84)   # 777-300ER
# ac35 = Aircraft(npax=313., range=unit.m_km(12500.), mach=0.84)   # 777-200ER
# ac36 = Aircraft(npax=467., range=unit.m_km(14000.), mach=0.84)   # 747-8

# fleet = Fleet([ac20, ac21, ac22, ac23, ac24, ac25, ac26, ac27, ac28, ac29, ac30, ac31, ac32, ac33, ac34, ac35, ac36])

# fleet = Fleet([ac0, ac1, ac2, ac3, ac4, ac5, ac6, ac7, ac8, ac9, ac10, ac11, ac12, ac13, ac14, ac20, ac21, ac22, ac23, ac24, ac25, ac26, ac27, ac28, ac29, ac30, ac31, ac32, ac33, ac34, ac35, ac36])

matrix_file = "../flight_data/all_flights_2016_matrix.bin"

data_matrix = load_matrix_from_file(matrix_file)



out_dict = fleet.fleet_analysis(data_matrix)

print("Fleet yearly trip = ","%.0f" % (out_dict["trip"]*1e-6)," Mtrip")
print("Fleet yearly fuel = ","%.2f" % (out_dict["fuel"]*1.e-9)," Mt")
print("Fleet yearly efficiency = ","%.3f" % ((out_dict["fuel"]/(out_dict["paxkm"]/100.))/0.803)," l/pax/100km")
print("Fleet total passengers = ","%.0f" % (out_dict["npax"]*1.e-6)," Mpax")
print("Fleet total capacity = ","%.0f" % (out_dict["capa"]*1.e-6)," Mpax")
print("Fleet mean fill rate = ","%.2f" % (out_dict["npax"]/out_dict["capa"]))
print("Fleet passager x km = ","%.2f" % (out_dict["paxkm"]*1.e-9)," 1e9 paxkm")
print("Fleet ton x km = ","%.2f" % (out_dict["tonkm"]*1.e-9)," 1e9 tonkm")
