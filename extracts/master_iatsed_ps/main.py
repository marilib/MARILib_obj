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

import unit

from component_airplane import Airplane





ap = Airplane()

n_pax = 150
range = unit.m_NM(3000)

wing_area = 122
engine_slst = unit.N_kN(120.)

mtow = 80000
mlw = 65000
mzfw = 60000

nominal_fuel = 20000

# Analysis functions
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax)

geometry = ap.compute_geometry(cabin_width, cabin_length, wing_area, engine_slst)

owe = ap.compute_owe(geometry, mtow, mzfw, mlw)

mtow, mzfw, mlw, mfw = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)

nominal_fuel, nominal_reserve = ap.compute_nominal_mission(mtow, range)

cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)




# MDA
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax)

geometry = ap.compute_geometry(cabin_width, cabin_length, wing_area, engine_slst)

# Mass-Mission adaptation
#-----------------------------------------------------------------------------------------------------------------------
def fct_mma(x):
mtow_i, mzfw_i, mlw_i = x
owe = ap.compute_owe(geometry, mtow_i, mzfw_i, mlw_i)
nominal_fuel, fuel_reserve = ap.compute_nominal_mission(mtow, range)
mtow_o, mzfw_o, mlw_o, mfw_o = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)
return [mtow_i-mtow_o, mzfw_i-mzfw_o, mlw_i-mlw_o]

xini = [mtow, mzfw, mlw]
output_dict = fsolve(fct_mma, x0=xini, args=(), full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

fct_mma(output_dict[0])

# Finalize MDA
#-----------------------------------------------------------------------------------------------------------------------
cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)





# MDF
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax)

# Performance optimization
#-----------------------------------------------------------------------------------------------------------------------
def fct_mda(x):

geometry = ap.compute_geometry(cabin_width, cabin_length, wing_area, engine_slst)

# Mass-Mission adaptation
#-----------------------------------------------------------------------------------------------------------------------
def fct_mma(x):
mtow_i, mzfw_i, mlw_i = x
owe = ap.compute_owe(geometry, mtow_i, mzfw_i, mlw_i)
nominal_fuel, fuel_reserve = ap.compute_nominal_mission(mtow, range)
mtow_o, mzfw_o, mlw_o, mfw_o = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)
return [mtow_i-mtow_o, mzfw_i-mzfw_o, mlw_i-mlw_o]

xini = [mtow, mzfw, mlw]
output_dict = fsolve(fct_mma, x0=xini, args=(), full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

fct_mma(output_dict[0])

# Finalize MDA
#-----------------------------------------------------------------------------------------------------------------------
cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)



# Graphics
#-----------------------------------------
ap.view_3d()

ap.print_airplane_data()

ap.missions.payload_range_diagram()




