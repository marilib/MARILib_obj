#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

import numpy as np

from marilib.utils import earth, unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process


# Configure airplane arrangement
# ---------------------------------------------------------------------------------------------------------------------
agmt = Arrangement(body_type = "fuselage",            # "fuselage" or "blended"
                   wing_type = "classic",             # "classic" or "blended"
                   wing_attachment = "low",           # "low" or "high"
                   stab_architecture = "classic",     # "classic", "t_tail" or "h_tail"
                   tank_architecture = "wing_box",    # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",         # "twin", "quadri" or "hexa"
                   nacelle_attachment = "wing",       # "wing", "rear" or "pods"
                   power_architecture = "pte",      # "tf", "tp", "ef", "ep", "pte", "pte", "extf", "exef"
                   power_source = "fuel",             # "fuel", "battery", "fuel_cell"
                   fuel_type = "kerosene")            # "kerosene", "liquid_h2", "Compressed_h2", "battery"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.))

ac = Aircraft("This_plane")     # Instantiate an Aircraft object

ac.factory(agmt, reqs)          # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line

# Eventual update of some values
#------------------------------------------------------------------------------------------------------
ac.power_system.reference_thrust = unit.N_kN(155.)
ac.airframe.wing.area = 130.

ac.airframe.system.battery = "no"
ac.airframe.system.battery_energy_density = unit.J_kWh(0.2) # J/kg, # Battery energy density
ac.airframe.system.cruise_energy = unit.J_kWh(140)          # J, energy stored in the battery dedicated to the cruise

ac.airframe.system.chain_power = unit.W_MW(1.)

ac.airframe.tail_nacelle.bli_effect = "yes"         # Include BLI effect in thrust computation

ac.airframe.system.generator_efficiency = 0.95
ac.airframe.system.rectifier_efficiency = 0.98
ac.airframe.system.wiring_efficiency = 0.995
ac.airframe.system.cooling_efficiency = 0.99

ac.airframe.tail_nacelle.controller_efficiency = 0.99
ac.airframe.tail_nacelle.motor_efficiency = 0.95


# Configure optimization problem
# ---------------------------------------------------------------------------------------------------------------------
var = ["aircraft.power_system.reference_thrust",
       "aircraft.airframe.wing.area"]               # Main design variables

var_bnd = [[unit.N_kN(80.), unit.N_kN(200.)],       # Design space area where to look for an optimum solution
           [100., 200.]]

# Operational constraints definition
cst = ["aircraft.performance.take_off.tofl_req - aircraft.performance.take_off.tofl_eff",
       "aircraft.performance.approach.app_speed_req - aircraft.performance.approach.app_speed_eff",
       "aircraft.performance.mcr_ceiling.vz_eff - aircraft.performance.mcr_ceiling.vz_req",
       "aircraft.performance.time_to_climb.ttc_req - aircraft.performance.time_to_climb.ttc_eff"]

# Magnitude used to scale constraints
cst_mag = ["aircraft.performance.take_off.tofl_req",
           "aircraft.performance.approach.app_speed_req",
           "unit.mps_ftpmin(100.)",
           "aircraft.performance.time_to_climb.ttc_req"]

# Optimization criteria
crt = "aircraft.weight_cg.mtow"


# Configure loop
# ---------------------------------------------------------------------------------------------------------------------
result = np.array([["e-fan power               (kW)"],
                   ["TF ref thrust            (daN)"],
                   ["Wing area                 (m2)"],
                   ["Wing span                  (m)"],
                   ["Fuselage length            (m)"],
                   ["Fuselage width             (m)"],
                   ["MTOW                      (kg)"],
                   ["MLW                       (kg)"],
                   ["OWE                       (kg)"],
                   ["MWE                       (kg)"],
                   ["TF nacelle mass           (kg)"],
                   ["Electric nacelle mass     (kg)"],
                   ["Electric chain mass       (kg)"],
                   ["Cruise true air speed     (kt)"],
                   ["Cruise thrust            (daN)"],
                   ["Cruise SFC          (kg/daN/h)"],
                   ["Cruise L/D            (no_dim)"],
                   ["Take Off Field Length      (m)"],
                   ["Approach speed            (kt)"],
                   ["Vz TOC MCL rating     (ft/min)"],
                   ["Time to climb            (min)"],
                   ["Design mission block fuel (kg)"],
                   ["Cost mission block fuel   (kg)"],
                   ["Cost mission block CO2    (kg)"],
                   ["Cash Op Cost          ($/trip)"],
                   ["CO2 metric (10e-4kg/km/m0.48)"]])

for chain_power in (0.15e6, 0.20e6, 0.25e6, 0.30e6, 0.35e6, 0.40e6, 0.45e6, 0.50e6, 0.75e6, 1.0e6, 1.25e6, 1.50e6, 1.75e6, 2.00e6):

    ac.airframe.system.chain_power = chain_power

    # Perform an MDF optimization process
    process.mdf(ac, var,var_bnd, cst,cst_mag, crt)

    print("-------------------------------------------")
    print("Optimization : done")

    # Save some characteristics
    #------------------------------------------------------------------------------------------------------
    res = np.array([["%8.0f"%(chain_power/1000.)],
                    ["%8.0f"%(ac.power_system.reference_thrust/10.)],
                    ["%8.1f"%ac.airframe.wing.area],
                    ["%8.1f"%ac.airframe.wing.span],
                    ["%8.1f"%ac.airframe.body.length],
                    ["%8.1f"%ac.airframe.body.width],
                    ["%8.0f"%ac.weight_cg.mtow],
                    ["%8.0f"%ac.weight_cg.mlw],
                    ["%8.0f"%ac.weight_cg.owe],
                    ["%8.0f"%ac.weight_cg.mwe],
                    ["%8.0f"%ac.airframe.nacelle.mass],
                    ["%8.0f"%ac.airframe.tail_nacelle.mass],
                    ["%8.0f"%ac.airframe.system.power_chain_mass],
                    ["%8.4f"%unit.kt_mps(ac.performance.mission.crz_tas)],
                    ["%8.4f"%(ac.performance.mission.crz_thrust/10.)],
                    ["%8.4f"%(ac.performance.mission.crz_tsfc*36000)],
                    ["%8.4f"%(ac.performance.mission.crz_lod)],
                    ["%8.0f"%ac.performance.take_off.tofl_eff],
                    ["%8.1f"%unit.kt_mps(ac.performance.approach.app_speed_eff)],
                    ["%8.0f"%unit.ftpmin_mps(ac.performance.mcl_ceiling.vz_eff)],
                    ["%8.1f"%unit.min_s(ac.performance.time_to_climb.ttc_eff)],
                    ["%8.0f"%ac.performance.mission.nominal.fuel_block],
                    ["%8.0f"%ac.performance.mission.cost.fuel_block],
                    ["%8.0f"%(ac.performance.mission.cost.fuel_block*3.14)],
                    ["%8.0f"%ac.economics.cash_op_cost],
                    ["%8.0f"%(ac.environment.CO2_metric*1.e7)]])

    result = np.hstack([result,res])

np.savetxt("scan_result.txt",result,delimiter=" ;",fmt='%s')


