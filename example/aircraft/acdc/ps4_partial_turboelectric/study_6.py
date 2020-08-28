#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019
@author: DRUOT Thierry

This study illustrates the integration of a partial turbo-electric airplane
Execution requires some minutes
"""

import numpy

from marilib.tools import units as unit

from marilib.aircraft_data.aircraft_description import Aircraft

from marilib.processes import assembly as run, initialization as init

from marilib.aircraft_model.airplane.viewer import draw_design_space

from marilib.processes.assembly import explore_design_space

# Initialize aircraft data structure
#------------------------------------------------------------------------------------------------------
aircraft = Aircraft()

design_range = unit.m_NM(3000)
cruise_mach = 0.78
n_pax_ref = 150

propulsion_architecture = "TF"
n_engine = 2

# Initialize all input data
#------------------------------------------------------------------------------------------------------
run.aircraft_initialize(aircraft, n_pax_ref, design_range, cruise_mach, propulsion_architecture, n_engine)


# Possibility to modify initial values
#------------------------------------------------------------------------------------------------------
study_name = "This airplane"

e_power = 0.05e6       # Watts, electric motor power

aircraft.rear_electric_engine.mto_r_shaft_power = e_power
aircraft.rear_electric_engine.mcn_r_shaft_power = e_power
aircraft.rear_electric_engine.mcl_r_shaft_power = e_power
aircraft.rear_electric_engine.mcr_r_shaft_power = e_power

aircraft.pte1_battery.strategy = 0     # Battery sizing strategy, 0= no battery, 1= power_feed & energy_cruise driven, 2= battery mass driven
aircraft.pte1_battery.energy_cruise = unit.J_kWh(140)     # J, energy stored in the battery dedicated to the cruise
aircraft.pte1_battery.energy_density = unit.J_kWh(0.2)    # J/kg, # Battery energy density

aircraft.propulsion.bli_effect = 1                      #init.boundary_layer_effect()
aircraft.pte1_power_elec_chain.overall_efficiency = 0.9    # 0.90 from init.e_chain_efficiency()

aircraft.pte1_power_elec_chain.generator_pw_density = 5.e3      # W/kg
aircraft.pte1_power_elec_chain.rectifier_pw_density = 20.e3     # W/kg
aircraft.pte1_power_elec_chain.wiring_pw_density = 20.e3        # W/kg
aircraft.pte1_power_elec_chain.cooling_pw_density = 10.e3       # W/kg
aircraft.rear_electric_nacelle.controller_pw_density = 20.e3    # W/kg
aircraft.rear_electric_nacelle.motor_pw_density = 5.e3          # W/kg
aircraft.rear_electric_nacelle.nacelle_pw_density = 5.e3        # W/kg

# Automatic optimization of the airplane
#------------------------------------------------------------------------------------------------------
thrust_bnd = (110000,150000)
area_bnd = (100,200)

search_domain = (thrust_bnd,area_bnd)

# Criterion to be chosen among  "MTOW", "cost_fuel", "CO2_metric", "COC", "DOC"
criterion = "MTOW"
mda_type = "MDA2"

run.mdf_process(aircraft,search_domain,criterion,mda_type)


# Print relevant output
#------------------------------------------------------------------------------------------------------
print("")
print("Fuel price = ","%.3f"%(init.fuel_price()*unit.liter_usgal(1))," $/USgal")
print("Elec price = ","%.3f"%(init.elec_price()*unit.J_kWh(1))," $/kW")

print("")
print("Engine thrust = ","%.1f"%(aircraft.propulsion.reference_thrust_effective/10)," daN")
print("Wing area = ","%.1f"%aircraft.wing.area," m2")
print("MTOW = ","%.0f"%aircraft.weights.mtow," kg")
print("OWE = ","%.0f"%aircraft.weights.owe," kg")

print("")
print("Turbofan nacelle mass = ","%.1f"%aircraft.turbofan_nacelle.mass," kg")
if (aircraft.propulsion.architecture=="PTE1"):
    print("Electric nacelle mass = ","%.1f"%aircraft.rear_electric_nacelle.mass," kg")
    print("Power electric mass = ","%.1f"%aircraft.pte1_power_elec_chain.mass," kg")
    if (aircraft.pte1_battery.strategy>0):
        print("Battery mass = ","%.1f"%aircraft.pte1_battery.mass," kg")

print("")
print("LoD cruise = ","%.2f"%(aircraft.aerodynamics.cruise_lod_max)," no_dim")
print("SFC cruise = ","%.3f"%(aircraft.propulsion.sfc_cruise_ref*36000)," kg/daN/h")
print("Nominal mission fuel = ","%.1f"%(aircraft.nominal_mission.block_fuel)," kg")

print("")
print("Rayon d'action demandé = ","%.0f"%unit.NM_m(aircraft.design_driver.design_range)," NM")
print(" . . . . . .  effectif = ","%.0f"%unit.NM_m(aircraft.nominal_mission.range)," NM")
print("")
print("Longueur de piste au décollage demandée = "+"%.0f"%aircraft.low_speed.req_tofl+" m")
print(" . . . . . . . . . . . . . .  effective = "+"%.0f"%aircraft.low_speed.eff_tofl+" m")
print("")
print("Vitesse d'approche demandée = "+"%.1f"%unit.kt_mps(aircraft.low_speed.req_app_speed)+" kt")
print(" . . . . . . . .  effective = "+"%.1f"%unit.kt_mps(aircraft.low_speed.eff_app_speed)+" kt")
print("")
print("Vitesse de monté demandé (MCL) = "+"%.1f"%unit.ftpmin_mps(aircraft.high_speed.req_vz_climb)+" ft/min")
print(" . . . . . . . effective (MCL) = "+"%.1f"%unit.ftpmin_mps(aircraft.high_speed.eff_vz_climb)+" ft/min")
print("")
print("Temps de monté demandé = "+"%.1f"%unit.min_s(aircraft.high_speed.req_ttc)+" min")
print(" . . . . . .  effectif = "+"%.1f"%unit.min_s(aircraft.high_speed.eff_ttc)+" min")

print("")
print("Cash Operating Cost = ","%.1f"%aircraft.economics.cash_operating_cost," $/trip")
print("Carbon dioxid emission = ","%.1f"%(aircraft.cost_mission.block_CO2)," kg/trip")
print("Fuel efficiency metric = ","%.4f"%(aircraft.environmental_impact.CO2_metric*1e7)," 10-7kg/km/m0.48")

# airplane 3D view
#------------------------------------------------------------------------------------------------------
#show.draw_3d_view(aircraft,"study_n5",study_name)

# Compute grid points
#------------------------------------------------------------------------------------------------------
res = [aircraft.propulsion.reference_thrust,
       aircraft.wing.area]
step = [0.05, 0.05]    # Relative grid step
file = "explore_design.txt"

explore_design_space(aircraft, res, step, mda_type, file)

# Draw graphic
#------------------------------------------------------------------------------------------------------
field = criterion
const = ['TOFL', 'App_speed', 'OEI_path', 'Vz_MCL', 'Vz_MCR', 'TTC']
color = ['red', 'blue', 'violet', 'orange', 'brown', 'yellow']
limit = [aircraft.low_speed.req_tofl,
         unit.kt_mps(aircraft.low_speed.req_app_speed),
         unit.pc_no_dim(aircraft.low_speed.req_oei_path),
         unit.ftpmin_mps(aircraft.high_speed.req_vz_climb),
         unit.ftpmin_mps(aircraft.high_speed.req_vz_cruise),
         unit.min_s(aircraft.high_speed.req_ttc)]       # Limit values
bound = numpy.array(["ub", "ub", "lb", "lb", "lb", "ub"])                 # ub: upper bound, lb: lower bound

draw_design_space(file, res, field, const, color, limit, bound)

