#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import unit, process

from component_airplane import Airplane





ap = Airplane()

n_pax = 186
range = unit.m_NM(3000)

hld_type = 10
wing_area = 132
engine_slst = unit.N_kN(142.)

mtow = 80000
mlw = 65000
mzfw = 60000
d_owe = 0

nominal_fuel = 20000




# Analysis functions
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax, range)

geometry = ap.compute_geometry(cabin_width, cabin_length, hld_type, wing_area, engine_slst)

owe = ap.compute_owe(geometry, mtow, mzfw, mlw, d_owe)

mtow, mzfw, mlw, mfw = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)

nominal_fuel, nominal_reserve = ap.compute_nominal_mission(mtow, range)

cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)



# Graphics
#-----------------------------------------
ap.view_3d()

ap.print_airplane_data()

ap.missions.payload_range_diagram()



