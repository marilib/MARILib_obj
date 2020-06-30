#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

import numpy as np

from marilib.utils import unit
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


# Select airplane main requirements
# ---------------------------------------------------------------------------------------------------------------------
reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.))


# Create an instance of Aircraft object
# ---------------------------------------------------------------------------------------------------------------------
ac = Aircraft("This_plane")     # Instantiate an Aircraft object


# Configure the object according to Arrangement & Requirements
# ---------------------------------------------------------------------------------------------------------------------
ac.factory(agmt, reqs)          # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line


# overwrite default values for design space graph centering (see below)
# ac.power_system.reference_thrust = unit.N_kN(160.)
# ac.airframe.wing.area = 128.


# Run Multidisciplinary Analysis
# ---------------------------------------------------------------------------------------------------------------------
process.mda(ac)                 # Run an MDA on the object (All internal constraints will be solved)


# Main output
# ---------------------------------------------------------------------------------------------------------------------
ac.draw.payload_range("This_plot")                      # Draw a payload range diagram
ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram

io = MarilibIO()
json = io.to_json_file(ac,'aircraft_outpout_data')      # Write all output data into a json readable format
io.to_binary_file(ac,'aircraft_binary_object')          # Write the complete Aircraft object into a binary file

