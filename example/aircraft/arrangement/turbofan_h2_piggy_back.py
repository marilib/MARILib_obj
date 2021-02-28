#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np

from marilib.utils import unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process


# Configure airplane arrangement
# ---------------------------------------------------------------------------------------------------------------------
agmt = Arrangement(body_type = "fuselage",            # "fuselage" or "blended"
                   wing_type = "classic",             # "classic" or "blended"
                   wing_attachment = "low",           # "low" or "high"
                   stab_architecture = "h_tail",    # "classic", "t_tail" or "h_tail"
                   tank_architecture = "piggy_back",# "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",         # "twin", "quadri" or "hexa"
                   nacelle_attachment = "wing",       # "wing", "rear", "pods", "body_cones"
                   power_architecture = "tf",         # "tf", "tp", "ef", "ep", "pte", "pte", "extf", "exef"
                   power_source = "fuel",             # "fuel", "battery", "fuel_cell"
                   fuel_type = "liquid_h2")         # "kerosene", "liquid_h2", "Compressed_h2", "battery"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(2000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.))

ac = Aircraft("This_plane")     # Instantiate an Aircraft object

ac.factory(agmt, reqs)          # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line

# overwrite default values for design space graph centering (see below)
ac.power_system.reference_thrust = unit.N_kN(160.)
ac.airframe.wing.area = 180.


process.mda(ac)                 # Run an MDA on the object (All internal constraints will be solved)


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
#opt = process.Optimizer()
#opt.mdf(ac, var,var_bnd, cst,cst_mag, crt,method='custom')
#algo_points= opt.computed_points

# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()



def get_section(length, toc):

    nose2 = np.array([[ 0.0000 , 0.0000 ,  0.0000 ] ,
                      [ 0.0050 , 0.0335 , -0.0335 ] ,
                      [ 0.0191 , 0.0646 , -0.0646 ] ,
                      [ 0.0624 , 0.1196 , -0.1196 ] ,
                      [ 0.1355 , 0.1878 , -0.1878 ] ,
                      [ 0.1922 , 0.2297 , -0.2297 ] ,
                      [ 0.2773 , 0.2859 , -0.2859 ] ,
                      [ 0.4191 , 0.3624 , -0.3624 ] ,
                      [ 0.5610 , 0.4211 , -0.4211 ] ,
                      [ 0.7738 , 0.4761 , -0.4761 ] ,
                      [ 0.9156 , 0.4976 , -0.4976 ] ,
                      [ 1.0000 , 0.5000 , -0.5000 ]])

    cone2 = np.array([[ 0.0000 , 0.5000 , -0.5000 ] ,
                      [ 0.0213 , 0.5000 , -0.5000 ] ,
                      [ 0.0638 , 0.4956 , -0.4956 ] ,
                      [ 0.1064 , 0.4875 , -0.4875 ] ,
                      [ 0.1489 , 0.4794 , -0.4794 ] ,
                      [ 0.1915 , 0.4720 , -0.4720 ] ,
                      [ 0.2766 , 0.4566 , -0.4566 ] ,
                      [ 0.3617 , 0.4330 , -0.4330 ] ,
                      [ 0.4894 , 0.3822 , -0.3822 ] ,
                      [ 0.6170 , 0.3240 , -0.3240 ] ,
                      [ 0.7447 , 0.2577 , -0.2577 ] ,
                      [ 0.8723 , 0.1834 , -0.1834 ] ,
                      [ 0.8936 , 0.1679 , -0.1679 ] ,
                      [ 0.9149 , 0.1524 , -0.1524 ] ,
                      [ 0.9362 , 0.1333 , -0.1333 ] ,
                      [ 0.9574 , 0.1097 , -0.1097 ] ,
                      [ 0.9787 , 0.0788 , -0.0788 ] ,
                      [ 0.9894 , 0.0589 , -0.0589 ] ,
                      [ 1.0000 , 0.0162 , -0.0162 ]])

    r_nose = 0.15       # Leading edga evolutive part
    r_cone = 0.35       # Trailing edge evolutive part

    width = length * toc

    leading_edge_xy = np.stack([nose2[0:,0]*length*r_nose , nose2[0:,1]*width , nose2[0:,2]*width], axis=1)
    trailing_edge_xy = np.stack([(1-r_cone)*length + cone2[0:,0]*length*r_cone , cone2[0:,1]*width , cone2[0:,2]*width], axis=1)
    section_xy = np.vstack([leading_edge_xy , trailing_edge_xy])

    return section_xy

curves = ac.draw.get_3d_curves()
io.to_binary_file(curves,'aircraft_curves.pkl')
comp = io.from_binary_file('aircraft_curves.pkl')

print(comp["name"])

print("-----------------------------------------")
print("nombre de surfaces = ", len(comp["surface"]))
for surf in comp["surface"]:
    print("-----------------------------------------")
    print("Leading edge")
    print(surf["le"])
    print("Trailing edge")
    print(surf["te"])
    print("Thickness over chord")
    print(surf["toc"])

print("-----------------------------------------")
print("nombre de bodies = ", len(comp["body"]))
for body in comp["body"]:
    print("-----------------------------------------")
    print("XZ curves")
    print(body["xz"])
    print("XY curves")
    print(body["xy"])

print("-----------------------------------------")
print("nombre de nacelles = ", len(comp["nacelle"]))
for nac in comp["nacelle"]:
    print("-----------------------------------------")
    print("Leading edge")
    print(nac["le"])
    print("Trailing edge")
    print(nac["te"])
    print("Thickness over chord")
    print(nac["toc"])

print("-----------------------------------------")
print("Example of section call")
sec = get_section(2.5, 0.12)
print(sec)


ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
ac.draw.payload_range("This_plot")                      # Draw a payload range diagram

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
        ["FUEL", "kg", "%8.1f", "aircraft.weight_cg.mfw"],
        ["Cost_Block_fuel", "kg", "%8.1f", "aircraft.performance.mission.cost.fuel_block"],
        ["Std_op_cost", "$/trip", "%8.1f", "aircraft.economics.std_op_cost"],
        ["Cash_op_cost", "$/trip", "%8.1f", "aircraft.economics.cash_op_cost"],
        ["Direct_op_cost", "$/trip", "%8.1f", "aircraft.economics.direct_op_cost"],
        ["CO2_metric", "kg/km/m0.48", "%8.4f", "unit.convert_to('kg/km/m0.48',aircraft.environment.CO2_metric)"]]

file = "aircraft_explore_design.txt"

# res = process.eval_this(ac,var)                                  # This function allows to get the values of a list of addresses in the Aircraft
res = process.explore_design_space(ac, var, step, data, file)      # Build a set of experiments using above config data and store it in a file

field = 'MTOW'                                                                  # Optimization criteria, keys are from data
other = ['MLW']                                                                 # Additional useful data to show
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

process.draw_design_space(file, res, other, field, const, color, limit, bound) # Used stored result to build a graph of the design space


