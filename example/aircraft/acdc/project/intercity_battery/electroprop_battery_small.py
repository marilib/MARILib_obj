#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

import numpy as np

from marilib.utils import unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process

from marilib.aircraft.model_config_small_plane import ModelConfiguration

# Configure airplane arrangement
# ---------------------------------------------------------------------------------------------------------------------
agmt = Arrangement(body_type = "fuselage",           # "fuselage" or "blended"
                   wing_type = "classic",            # "classic" or "blended"
                   wing_attachment = "high",       # "low" or "high"
                   stab_architecture = "t_tail",   # "classic", "t_tail" or "h_tail"
                   tank_architecture = "floor",      # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",        # "twin", "quadri" or "hexa"
                   nacelle_attachment = "wing",      # "wing", "rear" or "pods"
                   power_architecture = "ep",      # "tf", "tp", "ef", "ep", "pte", "pte", "extf", "exef"
                   power_source = "battery",            # "fuel", "battery", "fuel_cell"
                   fuel_type = "battery")           # "kerosene", "liquid_h2", "Compressed_h2", "battery"

reqs = Requirement(n_pax_ref = 19.,
                   design_range = unit.m_NM(20.),
                   cruise_mach = 0.30,
                   cruise_altp = unit.m_ft(25000.),
                   model_config = ModelConfiguration)

ac = Aircraft("This_plane")     # Instantiate an Aircraft object


ac.factory(agmt, reqs)          # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line

# overwrite eventually default values for operational requirements
#-----------------------------------------------------------------------------------------------------------------------
# Take off
ac.requirement.take_off.tofl_req = 420.

# Approach
ac.requirement.approach.app_speed_req = unit.convert_from("kt",72.)
# Climb
ac.requirement.mcl_ceiling.altp = unit.convert_from("ft",16000.)
ac.requirement.mcl_ceiling.mach = 0.2
ac.requirement.mcl_ceiling.vz_req = unit.convert_from("ft/min",1400.)

ac.requirement.mcr_ceiling.altp = unit.convert_from("ft",16000.)
ac.requirement.mcr_ceiling.mach = 0.2
ac.requirement.mcr_ceiling.vz_req = unit.convert_from("ft/min",900.)

ac.requirement.oei_ceiling.altp = unit.convert_from("ft",16000.)

ac.requirement.time_to_climb.cas1 = unit.convert_from("kt",80.)
ac.requirement.time_to_climb.altp1 = unit.convert_from("ft",1500.)
ac.requirement.time_to_climb.cas2 = unit.convert_from("kt",80.)
ac.requirement.time_to_climb.altp2 = unit.convert_from("ft",10000.)
ac.requirement.time_to_climb.altp = unit.convert_from("ft",16000.)
ac.requirement.time_to_climb.ttc_req = unit.convert_from("min",10.)

ac.airframe.wing.taper_ratio = 0.4


# overwrite default values for design parameters
ac.power_system.reference_power = unit.W_kW(1000.)

ac.airframe.wing.hld_type = 4.     # (max : 4)
ac.airframe.wing.aspect_ratio = 10. # (max : 12)
ac.airframe.wing.area = 60.


# overwrite default values for sensitivity analysis
ac.airframe.system.battery_energy_density = unit.J_Wh(400)  # Wh/kg


# Evaluation process
proc = "mda"

eval("process."+proc+"(ac)")  # Run MDA


# Configure optimization problem
# ---------------------------------------------------------------------------------------------------------------------
var = ["aircraft.power_system.reference_power",
       "aircraft.airframe.wing.area"]               # Main design variables

var_bnd = [[unit.W_kW(200.), unit.W_kW(600.)],       # Design space area where to look for an optimum solution
           [40., 100.]]

# Operational constraints definition
cst = ["aircraft.performance.take_off.tofl_req - aircraft.performance.take_off.tofl_eff",
       "aircraft.performance.approach.app_speed_req - aircraft.performance.approach.app_speed_eff",
       "aircraft.performance.mcl_ceiling.vz_eff - aircraft.performance.mcl_ceiling.vz_req",
       "aircraft.performance.mcr_ceiling.vz_eff - aircraft.performance.mcr_ceiling.vz_req",
       "aircraft.performance.oei_ceiling.path_eff - aircraft.performance.oei_ceiling.path_req",
       "aircraft.performance.time_to_climb.ttc_req - aircraft.performance.time_to_climb.ttc_eff",
       "aircraft.weight_cg.mfw - aircraft.performance.mission.nominal.fuel_total"]

# Magnitude used to scale constraints
cst_mag = ["aircraft.performance.take_off.tofl_req",
           "aircraft.performance.approach.app_speed_req",
           "unit.mps_ftpmin(100.)",
           "unit.mps_ftpmin(100.)",
           "aircraft.performance.oei_ceiling.path_req",
           "aircraft.performance.time_to_climb.ttc_req",
           "aircraft.weight_cg.mfw"]

# Optimization criteria
crt = "aircraft.weight_cg.mtow"

# Perform an MDF optimization process
# opt = process.Optimizer()
# opt.mdf(ac, var,var_bnd, cst,cst_mag, crt,method='optim2d_poly',proc=proc)
# algo_points= opt.computed_points

# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()
json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format
# dico = io.from_string(json)

ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
ac.draw.payload_range("This_plot")                      # Draw a payload range diagram

io.to_binary_file(ac,'aircraft_binary_object')          # Write the complete Aircraft object into a binary file
# ac2 = io.from_binary_file('test.pkl')                 # Read the complete Aircraft object from a file


# Configure design space exploration
# ---------------------------------------------------------------------------------------------------------------------
step = [0.05,
        0.05]    # Relative grid step

data = [["Power", "kW", "%8.1f", var[0]+"/1000."],
        ["Wing_area", "m2", "%8.1f", var[1]],
        ["Wing_span", "m", "%8.1f", "aircraft.airframe.wing.span"],
        ["MTOW", "kg", "%8.1f", "aircraft.weight_cg.mtow"],
        ["MLW", "kg", "%8.1f", "aircraft.weight_cg.mlw"],
        ["OWE", "kg", "%8.1f", "aircraft.weight_cg.owe"],
        ["MWE", "kg", "%8.1f", "aircraft.weight_cg.mwe"],
        ["Cruise_LoD", "no_dim", "%8.1f", "aircraft.performance.mission.crz_lod"],
        ["Cruise_SEC", "kW/daN", "%8.4f", "aircraft.performance.mission.crz_sec"],
        ["TOFL", "m", "%8.1f", "aircraft.performance.take_off.tofl_eff"],
        ["App_speed", "kt", "%8.1f", "unit.kt_mps(aircraft.performance.approach.app_speed_eff)"],
        ["OEI_path", "%", "%8.1f", "aircraft.performance.oei_ceiling.path_eff*100"],
        ["Vz_MCL", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.performance.mcl_ceiling.vz_eff)"],
        ["Vz_MCR", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.performance.mcr_ceiling.vz_eff)"],
        ["TTC", "min", "%8.1f", "unit.min_s(aircraft.performance.time_to_climb.ttc_eff)"],
        ["BATTERY", "kg", "%8.1f", "aircraft.weight_cg.mfw"],
        ["Cost_Block_enrg", "MW", "%8.1f", "aircraft.performance.mission.cost.enrg_block"],
        ["Std_op_cost", "$/trip", "%8.1f", "aircraft.economics.std_op_cost"],
        ["Cash_op_cost", "$/trip", "%8.1f", "aircraft.economics.cash_op_cost"],
        ["Direct_op_cost", "$/trip", "%8.1f", "aircraft.economics.direct_op_cost"],
        ["CO2_metric", "kg/km/m0.48", "%8.4f", "unit.convert_to('kg/km/m0.48',aircraft.environment.CO2_metric)"]]

file = "aircraft_explore_design.txt"

# res = process.eval_this(ac,var)                                  # This function allows to get the values of a list of addresses in the Aircraft
res = process.explore_design_space(ac, var, step, data, file, proc=proc)      # Build a set of experiments using above config data and store it in a file

field = 'MTOW'                                                                  # Optimization criteria, keys are from data
other = ['MLW']                                                                 # Additional useful data to show
const = ['TOFL', 'App_speed', 'OEI_path', 'Vz_MCL', 'Vz_MCR', 'TTC', 'BATTERY']    # Constrained performances, keys are from data
bound = np.array(["ub", "ub", "lb", "lb", "lb", "ub", "lb"])                    # ub: upper bound, lb: lower bound
color = ['red', 'blue', 'violet', 'orange', 'brown', 'yellow', 'black']         # Constraint color in the graph
limit = [ac.requirement.take_off.tofl_req,
         unit.kt_mps(ac.requirement.approach.app_speed_req),
         unit.pc_no_dim(ac.requirement.oei_ceiling.path_req),
         unit.ftpmin_mps(ac.requirement.mcl_ceiling.vz_req),
         unit.ftpmin_mps(ac.requirement.mcr_ceiling.vz_req),
         unit.min_s(ac.requirement.time_to_climb.ttc_req),
         ac.performance.mission.nominal.battery_mass]              # Limit values

process.draw_design_space(file, res, other, field, const, color, limit, bound) # Used stored result to build a graph of the design space


