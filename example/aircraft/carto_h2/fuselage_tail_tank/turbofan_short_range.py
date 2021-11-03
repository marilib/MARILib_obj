#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from tabulate import tabulate

from marilib.utils import unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process

from marilib.aircraft.step_mission import StepMission

# Architecture parameters
# ---------------------------------------------------------------------------------------------------------------------
agmt = Arrangement(body_type = "fuselage",           # "fuselage" or "blended"
                   wing_type = "classic",            # "classic" or "blended"
                   wing_attachment = "low",          # "low" or "high"
                   stab_architecture = "classic",    # "classic", "t_tail" or "h_tail"
                   tank_architecture = "rear",   # "wing_box", "rear", "piggy_back" or "pods"
                   number_of_engine = "twin",        # "twin", "quadri" or "hexa"
                   nacelle_attachment = "wing",      # "wing", "rear" or "pods"
                   power_architecture = "tf",        # "tf", "tp", "ef", "ep", "pte", , "extf", "exef"
                   power_source = "fuel",            # "fuel", "battery", "fuel_cell"
                   fuel_type = "liquid_h2")           # "kerosene", "liquid_h2", "Compressed_h2", "battery"

# Design parameters
#-----------------------------------------------------------------------------------------------------------------------
airplane_type = "A220-100"
ga_type = "rear_tank"

n_pax_ref = 135
design_range = unit.m_NM(990.)
cruise_mach = 0.78
cruise_altp = unit.m_ft(35000.)


# Build airplane object
#-----------------------------------------------------------------------------------------------------------------------
reqs = Requirement(n_pax_ref = n_pax_ref,
                   design_range = design_range,
                   cruise_mach = cruise_mach,
                   cruise_altp = cruise_altp)

ac = Aircraft(airplane_type)     # Instantiate an Aircraft object

ac.factory(agmt, reqs)          # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line


# Operational requirements
#-----------------------------------------------------------------------------------------------------------------------
# Max fuel mission
ac.requirement.max_fuel_range_factor = 1.25

# Take off
ac.requirement.take_off.altp = 0
ac.requirement.take_off.disa = 15
ac.requirement.take_off.tofl_req = 2000

# Approach
ac.requirement.approach.altp = 0
ac.requirement.approach.disa = 0
ac.requirement.approach.app_speed_req = unit.mps_kt(135)

# Climb
ac.requirement.mcl_ceiling.altp = cruise_altp
ac.requirement.mcl_ceiling.mach = cruise_mach
ac.requirement.mcl_ceiling.vz_req = unit.mps_ftpmin(300)

ac.requirement.mcr_ceiling.altp = cruise_altp
ac.requirement.mcr_ceiling.mach = cruise_mach
ac.requirement.mcr_ceiling.vz_req = unit.mps_ftpmin(0)

ac.requirement.oei_ceiling.altp = unit.m_ft(14000)

ac.requirement.time_to_climb.altp1 = unit.m_ft(1500)
ac.requirement.time_to_climb.cas1 = unit.mps_kt(180)
ac.requirement.time_to_climb.altp2 = unit.m_ft(10000)
ac.requirement.time_to_climb.cas2 = unit.mps_kt(250)
ac.requirement.time_to_climb.altp = cruise_altp
ac.requirement.time_to_climb.ttc_req = unit.s_min(25)

# Technological parameters
#-----------------------------------------------------------------------------------------------------------------------
ac.airframe.horizontal_stab.volume_factor = 0.94
ac.airframe.vertical_stab.wing_volume_factor = 0.07
ac.airframe.vertical_stab.thrust_volume_factor = 0.4

ac.airframe.tank.volumetric_index = 0.606
ac.airframe.tank.gravimetric_index = 0.1

# Design variables
#-----------------------------------------------------------------------------------------------------------------------
ac.power_system.reference_thrust = unit.N_kN(133.5)
ac.airframe.wing.area = 167.4

ac.airframe.wing.aspect_ratio = 11

ac.airframe.tank.ref_length = 15
ac.airframe.tank.mfw_factor = 1


# design_range = unit.m_NM(990.)
# n_pax_ref = 135
case_type = "max_range_soa"
ac.airframe.cabin.n_pax_front = 6
ac.airframe.tank.gravimetric_index = 0.1
ac.airframe.tank.volumetric_index = 0.606
ac.power_system.reference_thrust = unit.N_kN(215)
ac.airframe.wing.area = 276.9
ac.weight_cg.mtow = 139888
# fuselage ratio = 13.3  (limite à 13.4)

# design_range = unit.m_NM(1130.)
# n_pax_ref = 99
# case_type = "pax_range_trade_soa"
# ac.airframe.cabin.n_pax_front = 6
# ac.airframe.tank.gravimetric_index = 0.1
# ac.airframe.tank.volumetric_index = 0.606
# ac.power_system.reference_thrust = unit.N_kN(215)
# ac.airframe.wing.area = 276.9
# ac.weight_cg.mtow = 139888
# fuselage ratio = 13.3  (limite à 13.4)

# design_range = unit.m_NM(600.)
# n_pax_ref = 135
# case_type = "ref_600NM_soa"
# ac.airframe.cabin.n_pax_front = 6
# ac.airframe.tank.gravimetric_index = 0.1
# ac.airframe.tank.volumetric_index = 0.606
# ac.power_system.reference_thrust = unit.N_kN(139.8)
# ac.airframe.wing.area = 183.5
# ac.weight_cg.mtow = 91814
# fuselage ratio = 11.3  (limite à 13.4)



# design_range = unit.m_NM(2100.)
# n_pax_ref = 135
# case_type = "req_range_2030"
# ac.airframe.cabin.n_pax_front = 6
# ac.airframe.tank.gravimetric_index = 0.3
# ac.airframe.tank.volumetric_index = 0.845
# ac.power_system.reference_thrust = unit.N_kN(133.5)
# ac.airframe.wing.area = 167.4
# ac.weight_cg.mtow = 85797
# fuselage ratio = 12.9  (limite à 13.4)

# design_range = unit.m_NM(2730.)
# n_pax_ref = 99
# case_type = "pax_range_trade_2030"
# ac.airframe.cabin.n_pax_front = 6
# ac.airframe.tank.gravimetric_index = 0.3
# ac.airframe.tank.volumetric_index = 0.845
# ac.power_system.reference_thrust = unit.N_kN(133.5)
# ac.airframe.wing.area = 167.4
# ac.weight_cg.mtow = 85797
# fuselage ratio = 12.9  (limite à 13.4)

proc = "mda_plus"

eval("process."+proc+"(ac)")  # Run MDA


# Configure optimization problem
# ---------------------------------------------------------------------------------------------------------------------
var = ["aircraft.power_system.reference_thrust",
       "aircraft.airframe.wing.area"]               # Main design variables

var_bnd = [[unit.N_kN(80.), unit.N_kN(300.)],       # Design space area where to look for an optimum solution
           [100., 400.]]

# Operational constraints definition
cst = ["aircraft.performance.take_off.tofl_req - aircraft.performance.take_off.tofl_eff",
       "aircraft.performance.approach.app_speed_req - aircraft.performance.approach.app_speed_eff",
       "aircraft.performance.mcl_ceiling.vz_eff - aircraft.performance.mcl_ceiling.vz_req",
       "aircraft.performance.mcr_ceiling.vz_eff - aircraft.performance.mcr_ceiling.vz_req",
       "aircraft.performance.oei_ceiling.path_eff - aircraft.performance.oei_ceiling.path_req",
       "aircraft.performance.time_to_climb.ttc_req - aircraft.performance.time_to_climb.ttc_eff",
       "aircraft.weight_cg.mfw - aircraft.performance.mission.nominal.fuel_total",
       "aircraft.requirement.max_body_aspect_ratio - aircraft.airframe.body.aspect_ratio"]

# Magnitude used to scale constraints
cst_mag = ["aircraft.performance.take_off.tofl_req",
           "aircraft.performance.approach.app_speed_req",
           "unit.mps_ftpmin(100.)",
           "unit.mps_ftpmin(100.)",
           "aircraft.performance.oei_ceiling.path_req",
           "aircraft.performance.time_to_climb.ttc_req",
           "aircraft.weight_cg.mfw",
           "aircraft.requirement.max_body_aspect_ratio"]

# Optimization criteria
crt = "aircraft.weight_cg.mtow"

# Perform an MDF optimization process
opt = process.Optimizer()
# opt.mdf(ac, var,var_bnd, cst[0:-2],cst_mag[0:-2], crt,method='optim2d_poly',proc=proc)
# opt.mdf(ac, var,var_bnd, cst,cst_mag, crt,method='trust-constr')
# opt.mdf(ac, var,var_bnd, cst,cst_mag, crt)
# algo_points = opt.computed_points
# algo_points = None

print("MTOW = ", "%8.0f"%ac.weight_cg.mtow)

# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()
folder = "../files/"
file_name = airplane_type+"_"+ga_type+"_"+case_type
json = io.to_json_file(ac, folder+file_name)                # Write all output data into a json readable format
# dico = io.from_string(json)

io.to_binary_file(ac,'aircraft_binary_object')              # Write the complete Aircraft object into a binary file
# ac2 = io.from_binary_file('test.pkl')                     # Read the complete Aircraft object from a file

ac.draw.view_3d(file_name, folder=folder)    # Draw a 3D view diagram
ac.draw.payload_range("This_plot")           # Draw a payload range diagram


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
        ["FUEL", "kg", "%8.1f", "aircraft.weight_cg.mfw"],
        ["Body_AR", "no_dim", "%8.3f", "aircraft.airframe.body.aspect_ratio"],
        ["Cost_Block_fuel", "kg", "%8.1f", "aircraft.performance.mission.cost.fuel_block"],
        ["Std_op_cost", "$/trip", "%8.1f", "aircraft.economics.std_op_cost"],
        ["Cash_op_cost", "$/trip", "%8.1f", "aircraft.economics.cash_op_cost"],
        ["Direct_op_cost", "$/trip", "%8.1f", "aircraft.economics.direct_op_cost"],
        ["CO2_metric", "kg/km/m0.48", "%8.4f", "unit.convert_to('kg/km/m0.48',aircraft.environment.CO2_metric)"]]

file = "aircraft_explore_design.txt"

# res = process.eval_this(ac,var)                                  # This function allows to get the values of a list of addresses in the Aircraft
res = process.explore_design_space(ac, var, step, data, file, proc=proc)      # Build a set of experiments using above config data and store it in a file

field = 'MTOW'                                                                  # Optimization criteria, keys are from data
other = ['MLW']                                                                 # Additional useful data to show
const = ['TOFL', 'App_speed', 'OEI_path', 'Vz_MCL', 'Vz_MCR', 'TTC', 'FUEL', 'Body_AR']    # Constrained performances, keys are from data
bound = np.array(["ub", "ub", "lb", "lb", "lb", "ub", "lb", 'ub'])                    # ub: upper bound, lb: lower bound
color = ['red', 'blue', 'violet', 'orange', 'brown', 'yellow', 'black', 'grey']         # Constraint color in the graph
limit = [ac.requirement.take_off.tofl_req,
         unit.kt_mps(ac.requirement.approach.app_speed_req),
         unit.pc_no_dim(ac.requirement.oei_ceiling.path_req),
         unit.ftpmin_mps(ac.requirement.mcl_ceiling.vz_req),
         unit.ftpmin_mps(ac.requirement.mcr_ceiling.vz_req),
         unit.min_s(ac.requirement.time_to_climb.ttc_req),
         ac.performance.mission.nominal.fuel_total,
         ac.requirement.max_body_aspect_ratio]              # Limit values

process.draw_design_space(file, res, other, field, const, color, limit, bound) # Used stored result to build a graph of the design space


