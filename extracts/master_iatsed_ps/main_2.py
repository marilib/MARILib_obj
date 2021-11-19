#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
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
wing_area = 130
engine_slst = unit.N_kN(140.)

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



# MDF Process
#-----------------------------------------------------------------------------------------------------------------------
# Start of MDA process
#-----------------------------------------------------------------------------------------------------------------------


# INSERT HERE THE PART OF THE MDA SEQUENCE THAT IS INDEPENDANT FROM THE DESIGN VARIABLES



# Performance optimization
#-----------------------------------------------------------------------------------------------------------------------
def fct_mda(xx,ap):
    wing_area, engine_slst = xx
    #-------------------------------------------------------------------------------------------------------------------


# INSERT HERE THE REST OF THE MDA SEQUENCE



    #-----------------------------------------------------------------------------------------------------------------------
    # End of MDA process
    #-------------------------------------------------------------------------------------------------------------------
    cst = [float(ap.operations.take_off.tofl_req - ap.operations.take_off.tofl_eff) / ap.operations.take_off.tofl_req,
           float(ap.operations.approach.app_speed_req - ap.operations.approach.app_speed_eff) / ap.operations.approach.app_speed_req,
           float(ap.operations.mcl_ceiling.vz_eff - ap.operations.mcl_ceiling.vz_req),
           float(ap.operations.mcr_ceiling.vz_eff - ap.operations.mcr_ceiling.vz_req),
           float(ap.operations.oei_ceiling.path_eff - ap.operations.oei_ceiling.path_req) / ap.operations.oei_ceiling.path_req,
           float(ap.mass.mfw - ap.missions.nominal.fuel_total) /ap.mass.mfw]

    crt = mtow

    return crt, cst


# Call to the optimizer
#-----------------------------------------------------------------------------------------------------------------------
opt = process.Optimizer()
x_ini = np.array([wing_area, engine_slst])
bnd = [[50., 300.], [unit.N_kN(50.), unit.N_kN(300.)]]       # Design space area where to look for an optimum solution

x_res = opt.optimize([fct_mda,ap], x_ini, bnd)

fct_mda(x_res, ap)
#-----------------------------------------------------------------------------------------------------------------------
# End of MDF process




# Graphics
#-----------------------------------------
ap.print_airplane_data()

ap.view_3d()

ap.missions.payload_range_diagram()

ap.explore_design_space(fct_mda)


