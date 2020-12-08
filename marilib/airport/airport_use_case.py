#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 20 20:20:20 2020
@author: Cong Tam DO, Thierry DRUOT
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as pat

import unit

from physical_data import PhysicalData
from aircraft import AirplaneCategories
from airport import Airport

from power_plant import PvPowerPlantE




def operate(phd,ap, airport_capacity_ratio, fleet_profile):

    mean_distance = ap.town_distance + unit.m_km(20.)

    # Offer of public transport, the ratio représents the proportion of all the possible destinations that can be reached
    transport_offer = {"taxi":1., "drop_off_car":1., "parked_car":1., "bus":0.4, "train":0.4}       # TODO : Check coeffs for bus and train

    # Relative cost of transportation means, reference is taxi, WARNING drop_off_car cost is perceived cost, not full cost
    transport_cost = {"taxi":1., "drop_off_car":0.1, "parked_car":1., "bus":0.2, "train":0.2}      # TODO : Check coeffs

    # Relative convenience of transportation means
    transport_convenience = {"taxi":1., "drop_off_car":1., "parked_car":5., "bus":0.8, "train":0.8}      # TODO : Check coeffs

    # Energy consumption of each transportation mean
    transport_energy = {"taxi":(7. / 1.0)*1.e-8*phd.fuel_density("gasoline")*phd.fuel_heat("gasoline"),
                        "drop_off_car":(7. / 1.1)*1.e-8*phd.fuel_density("gasoline")*phd.fuel_heat("gasoline"),
                        "parked_car":(7. / 1.1)*1.e-8*phd.fuel_density("gasoline")*phd.fuel_heat("gasoline"),
                        "bus":(32. / 50.)*1.e-8*phd.fuel_density("gasoline")*phd.fuel_heat("gasoline"),
                        "train":unit.Jpm_kWhpkm(20. / 100.)}

    ac_list = []
    for seg,dict in fleet_profile.items():
        ac_list.append({"ratio":dict["ratio"], "npax":dict["npax"]})

    data_dict = ap.get_capacity(airport_capacity_ratio, ac_list)

    pax_flow = data_dict["pax_flow"]

    # Effective car transport offer is related to the number of park spaces
    transport_offer["parked_car"] *= (ap.car_parks.space_count / pax_flow)

    # Compute ground transport attractivity
    transport_attractivity = {}
    transport_flows = {}
    total = 0.
    for k in transport_offer.keys():
        transport_attractivity[k] = transport_convenience[k] * transport_offer[k] / transport_cost[k]
        total += transport_attractivity[k]
    # Renormalize
    for k in transport_attractivity.keys():
        transport_attractivity[k] /= total
        transport_flows[k] = pax_flow * transport_attractivity[k]

    # Compute ground energy for each transportation mean
    ground_energy = {}
    for k in transport_attractivity.keys():
        ground_energy[k] = 2. * pax_flow * mean_distance * transport_attractivity[k] * transport_energy[k]

    return pax_flow,transport_flows,ground_energy




# Tool objects
#-----------------------------------------------------------------------------------------------------------------------
phd = PhysicalData()
cat = AirplaneCategories()


# Fleet definition
#-----------------------------------------------------------------------------------------------------------------------
fleet_profile = {    "regional":{"ratio":0.30, "npax":70. , "range":unit.m_NM(500.) , "mach":0.50},
                  "short_range":{"ratio":0.50, "npax":150., "range":unit.m_NM(3000.), "mach":0.78},
                 "medium_range":{"ratio":0.15, "npax":300., "range":unit.m_NM(5000.), "mach":0.85},
                   "long_range":{"ratio":0.05, "npax":400., "range":unit.m_NM(7000.), "mach":0.85}}



# Airport
#-----------------------------------------------------------------------------------------------------------------------
runway_count = 3
app_dist = unit.m_NM(7.)
open_slot = [unit.s_h(6.), unit.s_h(23.)]
town_airport_dist = unit.m_km(8.)

# Instantiate an airport component
ap = Airport(cat, runway_count, open_slot, app_dist)

# Design the airport
ap.design(town_airport_dist, fleet_profile)


# On site production
#-----------------------------------------------------------------------------------------------------------------------
sol_pw = 250.    # W/m2, Mean solar irradiation
reg_factor = 0.7 # Regulation factor, 0.:no storage, 1.:regulation period is 24h

req_yearly_enrg = ap.ref_daily_energy * 365.

pv = PvPowerPlantE(req_yearly_enrg, sol_pw, reg_factor=reg_factor)

ap.print()
pv.print()
# ap.draw()


# Operation
#-----------------------------------------------------------------------------------------------------------------------
airport_capacity_ratio = 0.85

pax_flow,transport_flows,ground_energy = operate(phd,ap, airport_capacity_ratio, fleet_profile)

print("")
print("Ground transport energy")
print("==============================================================================")
print("Total number of passengers = ", "%.0f"%pax_flow)
for k in transport_flows.keys():
    print(k," : ", "%.0f"%transport_flows[k], " users, ", "%.2f"%unit.MWh_J(ground_energy[k]), " MWh")

