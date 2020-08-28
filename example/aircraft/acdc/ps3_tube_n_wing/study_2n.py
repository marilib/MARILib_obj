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
mtow_i = 72000.

ac.weight_cg.mtow = mtow_i

# Run MDA analysis
#------------------------------------------------------------------------------------------------------
process.mda(ac, mass_mission_matching=False)    # Run an MDA on the object, without mass - mission adaptation

# Evaluate required MTOW to satisfy nominal mission range
#------------------------------------------------------------------------------------------------------
owe_mission = mtow_i - ac.performance.mission.nominal.payload - ac.performance.mission.nominal.fuel_total

# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()
json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format

# ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
# ac.draw.payload_range("This_plot")                      # Draw a payload range diagram

# Print some relevant data
#------------------------------------------------------------------------------------------------------
print("")
print("MTOW input = ","%.0f"%mtow_i," kg")
print("OWE structure = ","%.0f"%ac.weight_cg.owe," kg")

print("")
print("Total mission fuel = ","%.2f"%ac.performance.mission.nominal.fuel_total," kg")
print("Payload = ","%.3f"%ac.performance.mission.nominal.payload," kg")
print("OWE mission = ","%.2f"%owe_mission," kg")
