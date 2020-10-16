#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from tabulate import tabulate

from marilib.utils import unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process

from marilib.aircraft.step_mission import StepMission

# Configure airplane arrangement
# ---------------------------------------------------------------------------------------------------------------------
agmt = Arrangement(body_type = "fuselage",           # "fuselage" or "blended"
                   wing_type = "classic",            # "classic" or "blended"
                   wing_attachment = "low",          # "low" or "high"
                   stab_architecture = "classic",    # "classic", "t_tail" or "h_tail"
                   tank_architecture = "wing_box",   # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",        # "twin", "quadri" or "hexa"
                   nacelle_attachment = "wing",      # "wing", "rear" or "pods"
                   power_architecture = "tf",        # "tf", "tp", "ef", "ep", "pte", "pte", "extf", "exef"
                   power_source = "fuel",            # "fuel", "battery", "fuel_cell"
                   fuel_type = "kerosene")           # "kerosene", "liquid_h2", "Compressed_h2", "battery"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.))

ac = Aircraft("This_plane")     # Instantiate an Aircraft object

ac.factory(agmt, reqs)          # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line

# overwrite default values for design space graph centering (see below)
ac.power_system.reference_thrust = unit.N_kN(170.)
ac.airframe.wing.area = 130.


process.mda(ac)                 # Run an MDA on the object (All internal constraints will be solved)



# miss = StepMission(ac)
#
# disa = 0.
#
# range = unit.m_NM(3000.)
#
# tow = 77866.
# owe = 45000.
#
# altp1 = unit.m_ft(1500.)
# cas1 = unit.mps_kt(250.)
#
# altp2 = unit.m_ft(10000.)
# cas2 = unit.mps_kt(300.)
#
# cruise_mach = 0.78
#
# vz_mcr = unit.mps_ftpmin(0.)
# vz_mcl = unit.mps_ftpmin(300.)
#
# heading = "east"
#
# miss.fly_mission(disa,range,tow,owe,altp1,cas1,altp2,cas2,cruise_mach,vz_mcr,vz_mcl,heading)
#
# print("------------------------------")
# table1 = np.array([[t1,unit.NM_m(x1),unit.ft_m(z1),m1] for t1,x1,z1,m1 in zip(miss.flight_profile["data"][:,0],
#                                                                               miss.flight_profile["data"][:,1],
#                                                                               miss.flight_profile["data"][:,2],
#                                                                               miss.flight_profile["data"][:,3])])
# k_list = [k for k,t in enumerate(table1[1:,0]) if t==table1[k,0]]
# table2 = np.delete(table1,k_list,axis=0)
# print(tabulate(table2))
#
# # miss.draw_flight_profile()
#
# profile = miss.flight_profile["data"][:,[0,1,2,7]]
#
# flight_path = miss.fly_this_profile(disa,tow,profile)
#
# print("------------------------------")
# print(miss.flight_profile["data"][0,3] - miss.flight_profile["data"][-1,3])
# print(flight_path["data"][0,3]-flight_path["data"][-1,3])
#
# table4 = [[t1,t2,z1,m1,m2,dm] for t1,t2,z1,m1,m2,dm in zip(table2[:,0],flight_path["data"][:,0],
#                                                      table2[:,2],
#                                                      table2[:,3],flight_path["data"][:,3],
#                                                      table2[:,3]-flight_path["data"][:,3])]
#
# # for z1,m1,m2,f1,f2 in zip(miss.flight_profile["data"][:,2],
# #                        miss.flight_profile["data"][:,3],flight_path["data"][:,3],
# #                        miss.flight_profile["data"][:,9],flight_path["data"][:,9]):
# #     print(unit.ft_m(z1),m1,m2,f1,f2)
#
# print(tabulate(table4))
#
# raise Exception("ici")


# Configure optimization problem
# ---------------------------------------------------------------------------------------------------------------------
var = ["aircraft.power_system.reference_thrust",
       "aircraft.airframe.wing.area"]               # Main design variables

var_bnd = [[unit.N_kN(80.), unit.N_kN(200.)],       # Design space area where to look for an optimum solution
           [100., 200.]]

# Operational constraints definition
cst = ["aircraft.performance.take_off.tofl_req - aircraft.performance.take_off.tofl_eff",
       "aircraft.performance.approach.app_speed_req - aircraft.performance.approach.app_speed_eff",
       "aircraft.performance.mcl_ceiling.vz_eff - aircraft.performance.mcl_ceiling.vz_req",
       "aircraft.performance.mcr_ceiling.vz_eff - aircraft.performance.mcr_ceiling.vz_req",
       "aircraft.performance.oei_ceiling.path_eff - aircraft.performance.oei_ceiling.path_req",
       "aircraft.performance.time_to_climb.ttc_req - aircraft.performance.time_to_climb.ttc_eff",
       "aircraft.airframe.tank.mfw_volume_limited - aircraft.performance.mission.nominal.fuel_total"]

# Magnitude used to scale constraints
cst_mag = ["aircraft.performance.take_off.tofl_req",
           "aircraft.performance.approach.app_speed_req",
           "unit.mps_ftpmin(100.)",
           "unit.mps_ftpmin(100.)",
           "aircraft.performance.oei_ceiling.path_req",
           "aircraft.performance.time_to_climb.ttc_req",
           "aircraft.airframe.tank.mfw_volume_limited"]

# Optimization criteria
crt = "aircraft.weight_cg.mtow"

# Perform an MDF optimization process
# opt = process.Optimizer()
# opt.mdf(ac, var,var_bnd, cst,cst_mag, crt,method='custom')
# algo_points = opt.computed_points
algo_points = None


# Main output
# ---------------------------------------------------------------------------------------------------------------------
ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
ac.draw.payload_range("This_plot")                      # Draw a payload range diagram

io = MarilibIO()
json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format
# dico = io.from_string(json)

io.to_binary_file(ac,'aircraft_binary_object')          # Write the complete Aircraft object into a binary file
# ac2 = io.from_binary_file('test.pkl')                 # Read the complete Aircraft object from a file


# Configure design space exploration
# ---------------------------------------------------------------------------------------------------------------------
step = [0.05,
        0.05]    # Relative grid step

data = [["Thrust", "daN", "%8.1f", var[0]+"/10."],
        ["Wing_area", "m2", "%8.1f", var[1]],
        ["Wing_span", "m", "%8.1f", "aircraft.airframe.wing.span"],
        ["MTOW", "kg", "%8.1f", "aircraft.weight_cg.mtow"],
        ["MLW", "kg", "%8.1f", "aircraft.weight_cg.mlw"],
        ["OWE", "kg", "%8.1f", "aircraft.weight_cg.owe"],
        ["MWE", "kg", "%8.1f", "aircraft.weight_cg.mwe"],
        ["Cruise_LoD", "no_dim", "%8.1f", "aircraft.performance.mission.crz_lod"],
        ["Cruise_SFC", "kg/daN/h", "%8.4f", "aircraft.performance.mission.crz_tsfc"],
        ["TOFL", "m", "%8.1f", "aircraft.performance.take_off.tofl_eff"],
        ["App_speed", "kt", "%8.1f", "unit.kt_mps(aircraft.performance.approach.app_speed_eff)"],
        ["OEI_path", "%", "%8.1f", "aircraft.performance.oei_ceiling.path_eff*100"],
        ["Vz_MCL", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.performance.mcl_ceiling.vz_eff)"],
        ["Vz_MCR", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.performance.mcr_ceiling.vz_eff)"],
        ["TTC", "min", "%8.1f", "unit.min_s(aircraft.performance.time_to_climb.ttc_eff)"],
        ["FUEL", "kg", "%8.1f", "aircraft.airframe.tank.mfw_volume_limited"],
        ["Cost_Block_fuel", "kg", "%8.1f", "aircraft.performance.mission.cost.fuel_block"],
        ["Std_op_cost", "$/trip", "%8.1f", "aircraft.economics.std_op_cost"],
        ["Cash_op_cost", "$/trip", "%8.1f", "aircraft.economics.cash_op_cost"],
        ["Direct_op_cost", "$/trip", "%8.1f", "aircraft.economics.direct_op_cost"],
        ["CO2_metric", "kg/km/m0.48", "%8.4f", "unit.convert_to('kg/km/m0.48',aircraft.environment.CO2_metric)"]]

file = "aircraft_explore_design.txt"

# res = process.eval_this(ac,var)                                  # This function allows to get the values of a list of addresses in the Aircraft
res = process.explore_design_space(ac, var, step, data, file)      # Build a set of experiments using above config data and store it in a file

field = 'MTOW'                                                                  # Optimization criteria, keys are from data
const = ['TOFL', 'App_speed', 'OEI_path', 'Vz_MCL', 'Vz_MCR', 'TTC', 'FUEL']    # Constrained performances, keys are from data
bound = np.array(["ub", "ub", "lb", "lb", "lb", "ub", "lb"])                    # ub: upper bound, lb: lower bound
color = ['red', 'blue', 'violet', 'orange', 'brown', 'yellow', 'black']         # Constraint color in the graph
limit = [ac.requirement.take_off.tofl_req,
         unit.kt_mps(ac.requirement.approach.app_speed_req),
         unit.pc_no_dim(ac.requirement.oei_ceiling.path_req),
         unit.ftpmin_mps(ac.requirement.mcl_ceiling.vz_req),
         unit.ftpmin_mps(ac.requirement.mcr_ceiling.vz_req),
         unit.min_s(ac.requirement.time_to_climb.ttc_req),
         ac.performance.mission.nominal.fuel_total]              # Limit values

process.draw_design_space(file, res, field, const, color, limit, bound, optim_points=algo_points) # Used stored result to build a graph of the design space


