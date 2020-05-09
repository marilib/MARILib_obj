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
ac1 = Aircraft(npax=100., range=unit.m_NM(3000.), mach=0.76)
ac2 = Aircraft(npax=150., range=unit.m_NM(3500.), mach=0.78)
ac3 = Aircraft(npax=300., range=unit.m_NM(6000.), mach=0.82)
ac4 = Aircraft(npax=400., range=unit.m_NM(8000.), mach=0.85)
ac5 = Aircraft(npax=700., range=unit.m_NM(8200.), mach=0.85)



fleet = Fleet([ac1, ac2, ac3, ac4, ac5])

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
