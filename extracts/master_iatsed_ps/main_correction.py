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
range = unit.m_NM(2400)

hld_type = 9

d_owe = 0

# INITIALIZATIONS
#-----------------------------------------------------------------------------------------------------------------------
wing_area = 120
engine_slst = unit.N_kN(130.)

mtow_i = 80000
mlw_i = 65000
mzfw_i = 60000
nominal_fuel = 20000

# OPERATIONAL REQUIREMENTS AND AIRPLANE INSTANTIATION
#-----------------------------------------------------------------------------------------------------------------------
ap = Airplane(tofl_req = 2100,
              app_speed_req = unit.mps_kt(137),
              vz_mcl_req = unit.mps_ftpmin(300),
              vz_mcr_req = unit.mps_ftpmin(0),
              oei_path_req = 0.011)



# ANALYSIS FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax, range)

geometry = ap.compute_geometry(cabin_width, cabin_length, hld_type, wing_area, engine_slst)

owe = ap.compute_owe(geometry, mtow_i, mzfw_i, mlw_i, d_owe)

mtow, mzfw, mlw, mfw = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)

nominal_fuel, nominal_reserve = ap.compute_nominal_mission(mtow_i, range)

cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)




# MDA
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax, range)

geometry = ap.compute_geometry(cabin_width, cabin_length, hld_type, wing_area, engine_slst)

# Mass-Mission adaptation
#-----------------------------------------------------------------------------------------------------------------------
def fct_mma(x):
    mtow_i, mzfw_i, mlw_i = x
    owe = ap.compute_owe(geometry, mtow_i, mzfw_i, mlw_i, d_owe)
    nominal_fuel, fuel_reserve = ap.compute_nominal_mission(mtow_i, range)
    mtow, mzfw, mlw_, mfw = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)
    return [mtow_i-mtow, mzfw_i-mzfw, mlw_i-mlw]

xini = [ap.mass.mtow, ap.mass.mzfw, ap.mass.mlw]
output_dict = fsolve(fct_mma, x0=xini, args=(), full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

mtow = output_dict[0][0]
mzfw = output_dict[0][1]
mlw = output_dict[0][2]

fct_mma([mtow, mzfw, mlw])

# Finalize MDA
#-----------------------------------------------------------------------------------------------------------------------
cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)






# MDF
#-----------------------------------------------------------------------------------------------------------------------
cabin_width, cabin_length, nominal_pl, max_pl = ap.compute_cabin(n_pax, range)

# Performance optimization
#-----------------------------------------------------------------------------------------------------------------------
def fct_mda(xx,ap):
    wing_area, engine_slst = xx

    geometry = ap.compute_geometry(cabin_width, cabin_length, hld_type, wing_area, engine_slst)

    # Mass-Mission adaptation
    #-----------------------------------------------------------------------------------------------------------------------
    def fct_mma(x):
        mtow_i, mzfw_i, mlw_i = x
        owe = ap.compute_owe(geometry, mtow_i, mzfw_i, mlw_i, d_owe)
        nominal_fuel, fuel_reserve = ap.compute_nominal_mission(mtow_i, range)
        mtow, mzfw, mlw, mfw = ap.compute_characteristic_weights(owe, nominal_pl, max_pl, nominal_fuel)
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
    cost_fuel_block, cost_time_bloc = ap.compute_other_missions(max_pl, mtow, mfw)

    cash_op_cost, direct_op_cost = ap.compute_other_performances(mtow, mlw, cost_fuel_block, cost_time_bloc)

    cst = [float(ap.operations.take_off.tofl_req - ap.operations.take_off.tofl_eff) / ap.operations.take_off.tofl_req,
           float(ap.operations.approach.app_speed_req - ap.operations.approach.app_speed_eff) / ap.operations.approach.app_speed_req,
           float(ap.operations.mcl_ceiling.vz_eff - ap.operations.mcl_ceiling.vz_req),
           float(ap.operations.mcr_ceiling.vz_eff - ap.operations.mcr_ceiling.vz_req),
           float(ap.operations.oei_ceiling.path_eff - ap.operations.oei_ceiling.path_req) / ap.operations.oei_ceiling.path_req,
           float(ap.mass.mfw - ap.missions.nominal.fuel_total) /ap.mass.mfw]

    crt = mtow

    return crt, cst


# opt = process.Optimizer()
# x_ini = np.array([wing_area, engine_slst])
# bnd = [[50., 300.], [unit.N_kN(50.), unit.N_kN(300.)]]       # Design space area where to look for an optimum solution
#
# x_res = opt.optimize([fct_mda,ap], x_ini, bnd)
#
# fct_mda(x_res, ap)


# Graphics
#-----------------------------------------
ap.view_3d()

ap.print_airplane_data()

ap.missions.payload_range_diagram()

ap.explore_design_space(fct_mda)


