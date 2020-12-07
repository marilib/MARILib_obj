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

ap.draw()


