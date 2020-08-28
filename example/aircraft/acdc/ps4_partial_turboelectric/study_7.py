#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019
@author: DRUOT Thierry
"""

from marilib import numpy

from marilib.tools import units as unit

from marilib.aircraft_model.airplane import viewer as show

from marilib.aircraft_data.aircraft_description import Aircraft

from marilib.processes import assembly as run, initialization as init

#======================================================================================================
# Initialization
#======================================================================================================
aircraft = Aircraft()

design_range = unit.m_NM(3000)
cruise_mach = 0.78
n_pax_ref = 150

propulsive_architecture = "PTE1" # TF:turbofan, PTE1:partial turboelectric 1
number_of_engine = 2

#------------------------------------------------------------------------------------------------------
run.aircraft_initialize(aircraft, n_pax_ref, design_range, cruise_mach, propulsive_architecture, number_of_engine)

#======================================================================================================
# Modify initial values here
#======================================================================================================

aircraft.propulsion.reference_thrust = 120000.
aircraft.wing.area = 149

e_power = 0.05e6       # Watts, electric motor power

result = numpy.array([["e-fan power               (kW)"],
                      ["TF ref thrust            (daN)"],
                      ["Effective ref thrust     (daN)"],
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
                      ["Cruise SFC          (kg/daN/h)"],
                      ["Cruise L/D            (no_dim)"],
                      ["Take Off Field Length      (m)"],
                      ["Approach speed            (kt)"],
                      ["One engine inop path       (%)"],
                      ["Vz TOC MCL rating     (ft/min)"],
                      ["Vz TOC MCR rating     (ft/min)"],
                      ["Time to climb            (min)"],
                      ["Design mission block fuel (kg)"],
                      ["Cost mission block fuel   (kg)"],
                      ["Cost mission block CO2    (kg)"],
                      ["Cash Op Cost          ($/trip)"],
                      ["CO2 metric (10e-4kg/km/m0.48)"]])

for e_power in (0.15e6, 0.20e6, 0.25e6, 0.30e6, 0.35e6, 0.40e6, 0.45e6, 0.50e6, 0.75e6, 1.0e6):

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

    #======================================================================================================
    # Design process
    #======================================================================================================

    #------------------------------------------------------------------------------------------------------
    thrust_bnd = (100000.,150000.)
    area_bnd = (100.,200.)
    search_domain = (thrust_bnd,area_bnd)

    # Perform MDF optimization
    #------------------------------------------------------------------------------------------------------
    criterion = "MTOW"
    mda_type = "MDA2"

    run.mdf_process(aircraft,search_domain,criterion,mda_type)

    print("-------------------------------------------")
    print("Optimization : done")

    # Some characteristics
    #------------------------------------------------------------------------------------------------------
    res = numpy.array([["%8.0f"%(e_power/1000.)],
                       ["%8.0f"%(aircraft.propulsion.reference_thrust/10)],
                       ["%8.0f"%(aircraft.propulsion.reference_thrust_effective/10)],
                       ["%8.1f"%aircraft.wing.area],
                       ["%8.1f"%aircraft.wing.span],
                       ["%8.1f"%aircraft.fuselage.length],
                       ["%8.1f"%aircraft.fuselage.width],
                       ["%8.0f"%aircraft.weights.mtow],
                       ["%8.0f"%aircraft.weights.mlw],
                       ["%8.0f"%aircraft.weights.owe],
                       ["%8.0f"%aircraft.weights.mwe],
                       ["%8.0f"%aircraft.turbofan_nacelle.mass],
                       ["%8.0f"%aircraft.rear_electric_nacelle.mass],
                       ["%8.0f"%aircraft.pte1_power_elec_chain.mass],
                       ["%8.4f"%(aircraft.propulsion.sfc_cruise_ref*36000)],
                       ["%8.4f"%(aircraft.aerodynamics.cruise_lod_max)],
                       ["%8.0f"%aircraft.low_speed.eff_tofl],
                       ["%8.1f"%unit.kt_mps(aircraft.low_speed.eff_app_speed)],
                       ["%8.1f"%(aircraft.low_speed.eff_oei_path*100)],
                       ["%8.0f"%unit.ftpmin_mps(aircraft.high_speed.eff_vz_climb)],
                       ["%8.0f"%unit.ftpmin_mps(aircraft.high_speed.eff_vz_cruise)],
                       ["%8.1f"%unit.min_s(aircraft.high_speed.eff_ttc)],
                       ["%8.0f"%aircraft.nominal_mission.block_fuel],
                       ["%8.0f"%aircraft.cost_mission.block_fuel],
                       ["%8.0f"%aircraft.cost_mission.block_CO2],
                       ["%8.0f"%aircraft.economics.cash_operating_cost],
                       ["%8.0f"%(aircraft.environmental_impact.CO2_metric*10000000)]])

    result = numpy.hstack([result,res])

numpy.savetxt("scan_result.txt",result,delimiter=" ;",fmt='%s')

#======================================================================================================
# Print some results
#======================================================================================================

# airplane 3D view
#------------------------------------------------------------------------------------------------------
