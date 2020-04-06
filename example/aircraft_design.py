#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np

from aircraft.tool import unit
from aircraft.aircraft_root import Arrangement
from aircraft.aircraft_root import Aircraft
from aircraft.requirement import Requirement

import process

from aircraft.tool.dictionary import MarilibIO




agmt = Arrangement(body_type = "fuselage",          # "fuselage" or "blended"
                   wing_type = "classic",           # "classic" or "blended"
                   wing_attachment = "low",         # "low" or "high"
                   stab_architecture = "classic",   # "classic", "t_tail" or "h_tail"
                   tank_architecture = "wing_box",  # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",       # "twin" or "quadri"
                   nacelle_attachment = "wing",     # "wing", "rear" or "pods"
                   power_architecture = "tf",       # "tf", "pf", "pte1", "ef1", "ep1",
                   energy_source = "kerosene")      # "kerosene", "methane", "liquid_h2", "700bar_h2" or "battery"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.76,
                   cruise_altp = unit.m_ft(35000.),
                   arrangement = agmt)





ac = Aircraft("This_plane")

ac.factory(agmt, reqs)  # WARNING : arrangement must not be changed after this line


# ac.airframe.wing.area = 110.
# ac.airframe.nacelle.reference_thrust = unit.N_kN(110.)


process.mda(ac)


var = ["aircraft.airframe.nacelle.reference_thrust",
       "aircraft.airframe.wing.area"]

var_bnd = [[unit.N_kN(80.), unit.N_kN(200.)],
           [100., 200.]]

cst = ["aircraft.performance.take_off.tofl_req - aircraft.performance.take_off.tofl_eff",
       "aircraft.performance.approach.app_speed_req - aircraft.performance.approach.app_speed_eff",
       "aircraft.performance.mcl_ceiling.vz_eff - aircraft.performance.mcl_ceiling.vz_req",
       "aircraft.performance.mcr_ceiling.vz_eff - aircraft.performance.mcr_ceiling.vz_req",
       "aircraft.performance.oei_ceiling.path_eff - aircraft.performance.oei_ceiling.path_req",
       "aircraft.performance.time_to_climb.ttc_req - aircraft.performance.time_to_climb.ttc_eff"]

cst_mag = ["aircraft.performance.take_off.tofl_req",
           "aircraft.performance.approach.app_speed_req",
           "unit.mps_ftpmin(100.)",
           "unit.mps_ftpmin(100.)",
           "aircraft.performance.oei_ceiling.path_req",
           "aircraft.performance.time_to_climb.ttc_req"]

#crt = "aircraft.weight_cg.mtow"
crt = "aircraft.performance.mission.cost.fuel_block"

#process.mdf(ac, var,var_bnd, cst,cst_mag, crt)



res = [ac.airframe.nacelle.reference_thrust,
       ac.airframe.wing.area]

step = [0.05,
        0.05]    # Relative grid step

data = [["SLST", "daN", "%8.1f", "aircraft.airframe.nacelle.reference_thrust/10."],
        ["Wing_area", "m2", "%8.1f", "aircraft.airframe.wing.area"],
        ["Wing_span", "m", "%8.1f", "aircraft.airframe.wing.span"],
        ["MTOW", "kg", "%8.1f", "aircraft.weight_cg.mtow"],
        ["MLW", "kg", "%8.1f", "aircraft.weight_cg.mlw"],
        ["OWE", "kg", "%8.1f", "aircraft.weight_cg.owe"],
        ["MWE", "kg", "%8.1f", "aircraft.weight_cg.mwe"],
        ["Cruise_LoD", "no_dim", "%8.1f", "aircraft.performance.mission.crz_lod"],
        ["Cruise_SFC", "kg/daN/h", "%8.1f", "aircraft.performance.mission.crz_sfc"],
        ["TOFL", "m", "%8.1f", "aircraft.performance.take_off.tofl_eff"],
        ["App_speed", "kt", "%8.1f", "unit.kt_mps(aircraft.performance.approach.app_speed_eff)"],
        ["OEI_path", "%", "%8.1f", "aircraft.performance.oei_ceiling.path_eff*100"],
        ["Vz_MCL", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.performance.mcl_ceiling.vz_eff)"],
        ["Vz_MCR", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.performance.mcr_ceiling.vz_eff)"],
        ["TTC", "min", "%8.1f", "unit.min_s(aircraft.performance.time_to_climb.ttc_eff)"],
        ["Block_fuel", "kg", "%8.1f", "aircraft.performance.mission.cost.fuel_block"],
        ["Std_op_cost", "$/trip", "%8.1f", "aircraft.economics.std_op_cost"],
        ["Cash_op_cost", "$/trip", "%8.1f", "aircraft.economics.cash_op_cost"],
        ["Direct_op_cost", "$/trip", "%8.1f", "aircraft.economics.direct_op_cost"],
        ["CO2_metric", "kg/km/m0.48", "%8.4f", "unit.convert_to('kg/km/m0.48',aircraft.environment.CO2_metric)"]]

file = "explore_design.txt"

#process.explore_design_space(ac, res, step, data, file)

field = 'MTOW'
const = ['TOFL', 'App_speed', 'OEI_path', 'Vz_MCL', 'Vz_MCR', 'TTC']
color = ['red', 'blue', 'violet', 'orange', 'brown', 'yellow']
limit = [ac.performance.take_off.tofl_req,
         unit.kt_mps(ac.performance.approach.app_speed_req),
         unit.pc_no_dim(ac.performance.oei_ceiling.path_req),
         unit.ftpmin_mps(ac.performance.mcl_ceiling.vz_req),
         unit.ftpmin_mps(ac.performance.mcr_ceiling.vz_req),
         unit.min_s(ac.performance.time_to_climb.ttc_req)]       # Limit values
bound = np.array(["ub", "ub", "lb", "lb", "lb", "ub"])                 # ub: upper bound, lb: lower bound

#process.draw_design_space(file, res, field, const, color, limit, bound)

#ac.draw.payload_range("This_plot")
#ac.draw.view_3d("This_plot")

io = MarilibIO()
json = io.to_json_file(ac,'aircraft_test')
#dico = io.from_string(json)

io.to_binary_file(ac,'test')
#ac2 = io.from_binary_file('test.pkl')

