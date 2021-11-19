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


# Configure airplane arrangement
# ---------------------------------------------------------------------------------------------------------------------
agmt = Arrangement(body_type = "fuselage",            # "fuselage" or "blended"
                   wing_type = "classic",             # "classic" or "blended"
                   wing_attachment = "low",           # "low" or "high"
                   stab_architecture = "h_tail",    # "classic", "t_tail" or "h_tail"
                   tank_architecture = "piggy_back",# "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",         # "twin", "quadri" or "hexa"
                   nacelle_attachment = "wing",       # "wing", "rear", "pods", "body_cones"
                   power_architecture = "pte_piggy",         # "tf", "tp", "ef", "ep", "pte", "pte", "extf", "exef"
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

ac.airframe.system.chain_power_body = unit.W_MW(1)
ac.airframe.system.chain_power_piggyback = unit.W_MW(1)

process.mda(ac)                 # Run an MDA on the object (All internal constraints will be solved)


# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()

json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format
# dico = io.from_string(json)

io.to_binary_file(ac,'aircraft_binary_object')          # Write the complete Aircraft object into a binary file
# ac2 = io.from_binary_file('test.pkl')                 # Read the complete Aircraft object from a file

curves = ac.draw.get_3d_curves()
io.to_binary_file(curves,'aircraft_curves')

ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram

