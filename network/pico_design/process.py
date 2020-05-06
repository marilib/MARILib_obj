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

import pandas
import matplotlib.pyplot as plt

from context import unit, math

from network.pico_design.design_model import Aircraft, Fleet

import network.flight_data.data_explorer as data_explorer


# ======================================================================================================
# Test
# ------------------------------------------------------------------------------------------------------
# ac = Aircraft(npax=150., range=unit.m_NM(3000.), mach=0.78)
#
# ac.payload_range()

# ======================================================================================================
# Fleet evaluation
# ------------------------------------------------------------------------------------------------------
ac1 = Aircraft(npax=100., range=unit.m_NM(2000.), mach=0.72)
ac2 = Aircraft(npax=150., range=unit.m_NM(3000.), mach=0.78)
ac3 = Aircraft(npax=300., range=unit.m_NM(6000.), mach=0.82)
ac4 = Aircraft(npax=400., range=unit.m_NM(8000.), mach=0.85)
ac5 = Aircraft(npax=500., range=unit.m_NM(8000.), mach=0.85)

fleet = Fleet([ac1, ac2, ac3, ac4, ac5])

matrix_file = "../flight_data/all_flights_2019_matrix.bin"

data_matrix = data_explorer.load_matrix_from_file(matrix_file)

total_fuel = fleet.fleet_fuel(data_matrix)

print("Fleet yearly fuel = ","%.0f" % total_fuel)
