#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot
"""

import pickle

#import network.flight_data.data_explorer as data_explorer

from marilib.utils import unit

from marilib.network.pico_design.calibration import get_data

from marilib.network.pico_design.design_model import Aircraft, Fleet

from marilib.network.flight_data.data_explorer import load_data_from_file



# ======================================================================================================
# Test
# ------------------------------------------------------------------------------------------------------
data_dict = get_data("../input_data/Aircraft_general_data_v2.csv")
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
ac4 = Aircraft(npax=450., range=unit.m_km(12000.), mach=0.85)   # Long
ac5 = Aircraft(npax=500., range=unit.m_km(14000.), mach=0.85)   # Very long

ac_list = [ac0, ac1, ac2, ac3, ac4, ac5]

Airbus_list = []
for i,name in enumerate(data_dict["Name"]):
    if name[0]=="A":
        Airbus_list.append(Aircraft(npax=data_dict["Npax"][i], range=data_dict["Range"][i], mach=data_dict["Speed"][i]))

Boeing_list = []
Full_list = Airbus_list
for i,name in enumerate(data_dict["Name"]):
    if name[0]!="A":
        Boeing_list.append(Aircraft(npax=data_dict["Npax"][i], range=data_dict["Range"][i], mach=data_dict["Speed"][i]))
        Full_list.append(Boeing_list[-1])


matrix_file = "../flight_data/all_flights_2016_matrix.bin"

data_matrix = load_data_from_file(matrix_file)

fleet = Fleet(ac_list)

out_dict = fleet.fleet_analysis(data_matrix)

print("Fleet yearly trip = ","%.0f" % (out_dict["trip"]*1e-6)," Mtrip")
print("Fleet yearly fuel = ","%.2f" % (out_dict["fuel"]*1.e-9)," Mt")
print("Fleet yearly efficiency = ","%.3f" % ((out_dict["fuel"]/(out_dict["paxkm"]/100.))/0.803)," l/pax/100km")
print("Fleet total passengers = ","%.0f" % (out_dict["npax"]*1.e-6)," Mpax")
print("Fleet total capacity = ","%.0f" % (out_dict["capa"]*1.e-6)," Mpax")
print("Fleet mean fill rate = ","%.2f" % (out_dict["npax"]/out_dict["capa"]))
print("Fleet passager x km = ","%.2f" % (out_dict["paxkm"]*1.e-9)," 1e9 paxkm")
print("Fleet ton x km = ","%.2f" % (out_dict["tonkm"]*1.e-9)," 1e9 tonkm")
print("")
for j in range(len(fleet.fleet_plane)):
    print(" Type ",j,"  n trip = ",fleet.fleet_trip[j],"  n plane = ",fleet.fleet_plane[j])
print("")
print("Total = ",sum(fleet.fleet_plane))
