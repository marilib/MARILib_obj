#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
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

ac.airframe.system.chain_power = unit.W_MW(1.)      # Shaft power of the electric motor

if (ac.arrangement.power_architecture=="pte"):
    ac.airframe.tail_nacelle.bli_effect = "no"         # Include BLI effect in thrust computation

    ac.airframe.tail_nacelle.controller_efficiency = 0.99
    ac.airframe.tail_nacelle.motor_efficiency = 0.95

ac.airframe.system.generator_efficiency = 0.95
ac.airframe.system.rectifier_efficiency = 0.99
ac.airframe.system.wiring_efficiency = 0.995
ac.airframe.system.cooling_efficiency = 0.995

#     ac.airframe.tail_nacelle.controller_efficiency = 1.
#     ac.airframe.tail_nacelle.motor_efficiency = 1.
#
# ac.airframe.system.generator_efficiency = 1.
# ac.airframe.system.rectifier_efficiency = 1.
# ac.airframe.system.wiring_efficiency = 1.
# ac.airframe.system.cooling_efficiency = 1.

# Run MDA analysis
#------------------------------------------------------------------------------------------------------
process.mda(ac)                 # Run an MDA on the object (All internal constraints will be solved)


# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()
json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format

# Print some relevant output
#------------------------------------------------------------------------------------------------------
altp = ac.performance.mission.altp
disa = ac.performance.mission.disa
pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
tas = ac.performance.mission.mach * earth.sound_speed(tamb)

print("")
print("True air speed in cruise","%.1f"%tas," m/s")
print("Total thrust in cruise","%.0f"%ac.performance.mission.crz_thrust," N")

print("")
print("Engine reference thrust = ","%.1f"%(ac.power_system.reference_thrust/10)," daN")
print("Wing area = ","%.1f"%ac.airframe.wing.area," m2")
print("MTOW = ","%.0f"%ac.weight_cg.mtow," kg")
print("OWE = ","%.0f"%ac.weight_cg.owe," kg")

print("")
print("Turbofan nacelle mass = ","%.1f"%ac.airframe.nacelle.mass," kg")
if (ac.arrangement.power_architecture=="pte"):
    print("Electric nacelle mass = ","%.1f"%ac.airframe.tail_nacelle.mass," kg")
    print("Power electric mass = ","%.1f"%ac.airframe.system.power_chain_mass," kg")
    print("Power chain efficiency = ","%.3f"%ac.airframe.system.power_chain_efficiency)
    if (ac.airframe.system.battery=="yes"):
        print("Battery mass = ","%.1f"%ac.airframe.system.battery_mass," kg")

print("")
print("LoD cruise = ","%.2f"%ac.performance.mission.crz_lod," no_dim")
print("TSFC cruise = ","%.3f"%(ac.performance.mission.crz_tsfc*36000)," kg/daN/h")
if (ac.arrangement.power_architecture=="pte"):
    print("SEC cruise = ","%.3f"%(ac.performance.mission.crz_sec/100)," kW/daN (tail engine only)")
print("Design mission block fuel = ","%.1f"%(ac.performance.mission.nominal.fuel_block)," kg")

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

print("")
print("Cash Operating Cost = ","%.1f"%ac.economics.cash_op_cost," $/trip")
print("Cost mission block fuel = ","%.1f"%(ac.performance.mission.cost.fuel_block)," kg/trip")
print("Carbon dioxide emission = ","%.1f"%(ac.performance.mission.cost.fuel_block*3.14)," kg/trip")
print("Fuel efficiency metric = ","%.4f"%(ac.environment.CO2_metric*1e7)," 10-7kg/km/m0.48")

ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
ac.draw.payload_range("This_plot")                      # Draw a payload range diagram

