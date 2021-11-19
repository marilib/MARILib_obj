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
ac.power_system.reference_thrust = unit.N_kN(110.)
ac.airframe.wing.area = 110.

# Run MDA analysis
#------------------------------------------------------------------------------------------------------
process.mda(ac, mass_mission_matching=True)    # Run an MDA on the object, with mass - mission adaptation


# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()
json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format

# ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
# ac.draw.payload_range("This_plot")                      # Draw a payload range diagram

# Print some relevant data
#------------------------------------------------------------------------------------------------------
print("")
print("MTOW = ","%.1f"%ac.weight_cg.mtow," kg")

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
