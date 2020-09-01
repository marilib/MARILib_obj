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


# Eventual update of some values
#------------------------------------------------------------------------------------------------------
ac.power_system.reference_thrust = unit.N_kN(160.)
ac.airframe.wing.area = 125.

ac.airframe.wing.wing_morphing = "aspect_ratio_driven"  # "aspect_ratio_driven" or "span_driven"

ac.airframe.wing.aspect_ratio = 9.
ac.airframe.wing.span = 32.


# Run MDA analysis
#------------------------------------------------------------------------------------------------------
process.mda(ac, mass_mission_matching=True)    # Run an MDA on the object, with mass - mission adaptation


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


# Perform an MDF optimization process
process.mdf(ac, var,var_bnd, cst,cst_mag, crt)


# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()
json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format

ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
ac.draw.payload_range("This_plot")                      # Draw a payload range diagram


# Print some relevant data
#------------------------------------------------------------------------------------------------------
print("")
print("Engine thrust = ","%.1f"%(ac.power_system.reference_thrust/10)," daN")
print("Engine mass = ","%.1f"%(ac.airframe.nacelle.mass)," kg")
print("SFC cruise = ","%.3f"%(ac.performance.mission.crz_tsfc*36000)," kg/daN/h")
print("")
print("Wing area = ","%.1f"%ac.airframe.wing.area," m2")
print("Wing span = ","%.1f"%ac.airframe.wing.span," m")
print("Wing aspect ratio = ","%.1f"%ac.airframe.wing.aspect_ratio," no_dim")
print("Wing mass = ","%.1f"%ac.airframe.wing.mass," kg")
print("LoD cruise (LoD max) = ","%.2f"%ac.performance.mission.crz_lod," no_dim")
print("")
print("MTOW = ","%.0f"%ac.weight_cg.mtow," kg")
print("OWE = ","%.0f"%ac.weight_cg.owe," kg")

print("")
print("Cash Operating Cost = ","%.1f"%ac.economics.cash_op_cost," $/trip")
print("Cost mission block fuel = ","%.1f"%(ac.performance.mission.cost.fuel_block)," kg/trip")
print("Carbon dioxide emission = ","%.1f"%(ac.performance.mission.cost.fuel_block*3.14)," kg/trip")

print("")
print("Take off field length required = "+"%.1f"%ac.performance.take_off.tofl_req+" m")
print("Take off field length effective = "+"%.1f"%ac.performance.take_off.tofl_eff+" m")
print("")
print("Approach speed required = "+"%.1f"%unit.kt_mps(ac.performance.approach.app_speed_req)+" kt")
print("Approach speed effective = "+"%.1f"%unit.kt_mps(ac.performance.approach.app_speed_eff)+" kt")
print("")
print("Vertical speed required = "+"%.1f"%unit.ftpmin_mps(ac.performance.mcl_ceiling.vz_req)+" ft/min")
print("Vertical speed effective = "+"%.1f"%unit.ftpmin_mps(ac.performance.mcl_ceiling.vz_eff)+" ft/min")
print("")
print("Time to climb required = "+"%.1f"%unit.min_s(ac.performance.time_to_climb.ttc_req)+" min")
print("Time to climb effective = "+"%.1f"%unit.min_s(ac.performance.time_to_climb.ttc_eff)+" min")


# Configure design space mapping
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
const = ['TOFL', 'App_speed', 'Vz_MCR', 'TTC']    # Constrained performances, keys are from data
bound = np.array(["ub", "ub", "lb", "ub"])                    # ub: upper bound, lb: lower bound
color = ['red', 'blue', 'brown', 'yellow']         # Constraint color in the graph
limit = [ac.requirement.take_off.tofl_req,
         unit.kt_mps(ac.requirement.approach.app_speed_req),
         unit.ftpmin_mps(ac.requirement.mcl_ceiling.vz_req),
         unit.min_s(ac.requirement.time_to_climb.ttc_req)]              # Limit values

process.draw_design_space(file, res, field, const, color, limit, bound) # Used stored result to build a graph of the design space

