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

import unit, process
from component_airplane import Airplane



# DESIGN SETTINGS
#-----------------------------------------------------------------------------------------------------------------------
n_pax = 150
range = unit.m_NM(2500)

hld_type = 9

d_owe = 0

# INITIALIZATIONS
#-----------------------------------------------------------------------------------------------------------------------
wing_area = 130
engine_slst = unit.N_kN(140.)

mtow_i = 80000
mlw_i = 65000
mzfw_i = 60000
nominal_fuel = 20000

# OPERATIONAL REQUIREMENTS AND AIRPLANE INSTANTIATION
#-----------------------------------------------------------------------------------------------------------------------
ap = Airplane()



# ANALYSIS FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax, range)

geometry = ap.compute_geometry(cabin_width, cabin_length, hld_type, wing_area, engine_slst)

owe = ap.compute_owe(geometry, mtow_i, mzfw_i, mlw_i, d_owe)

mtow, mzfw, mlw, mfw = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)

nominal_fuel, nominal_reserve = ap.compute_nominal_mission(mtow_i, range)

cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)




# MDA process
#-----------------------------------------------------------------------------------------------------------------------


# INSERT HERE WEAKLY COUPLED FUNCTIONS THAT CAN BE RUN PRIOR TO ANY OTHERS



# Mass-Mission adaptation
#-----------------------------------------------------------------------------------------------------------------------
def fct_mma(x):
    mtow_i, mzfw_i, mlw_i = x
    #-------------------------------------------------------------------------------------------------------------------


# INSERT HERE STRONGLY COUPLED FUNCTIONS IN THE RIGHT ORDER


    #-------------------------------------------------------------------------------------------------------------------
    return [mtow_i-mtow, mzfw_i-mzfw, mlw_i-mlw]

xini = [ap.mass.mtow, ap.mass.mzfw, ap.mass.mlw]
output_dict = fsolve(fct_mma, x0=xini, args=(), full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

mtow = output_dict[0][0]
mzfw = output_dict[0][1]
mlw = output_dict[0][2]

fct_mma([mtow, mzfw, mlw])

mfw = ap.mass.mfw

# Finalize MDA
#-----------------------------------------------------------------------------------------------------------------------



# INSERT HERE WEAKLY COUPLED FUNCTIONS THAT CAN BE RUN AFTER THE OTHERS



#-----------------------------------------------------------------------------------------------------------------------
# End of MDA process




# Graphics and printings
#-----------------------------------------
ap.print_airplane_data()

ap.view_3d()

ap.missions.payload_range_diagram()


