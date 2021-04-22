#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
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
                   tank_architecture = "wing_box",   # "wing_box", "rear", "piggy_back" or "pods"
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


# Get the values that have been set by initialization
#------------------------------------------------------------------------------------------------------
wing_morphing_i = "aspect_ratio_driven" # wing morphing : "aspect_ratio_driven" or "span_driven"
wing_span_i = ac.airframe.wing.span
wing_aspect_ratio_i = ac.airframe.wing.aspect_ratio
wing_area_i = ac.airframe.wing.area
mtow_i = ac.weight_cg.mtow

# Eventual update of some values
#------------------------------------------------------------------------------------------------------
ac.airframe.wing.wing_morphing = wing_morphing_i

ac.airframe.wing.area = 150.
ac.airframe.wing.span = wing_span_i
ac.airframe.wing.aspect_ratio = wing_aspect_ratio_i

ac.weight_cg.mtow = mtow_i

# Run MDA analysis
#------------------------------------------------------------------------------------------------------
process.mda(ac, mass_mission_matching=False)    # Run an MDA on the object, without mass - mission adaptation

# Evaluate required MTOW to satisfy nominal mission range
#------------------------------------------------------------------------------------------------------
mtow_req = ac.weight_cg.owe + ac.performance.mission.nominal.payload + ac.performance.mission.nominal.fuel_total

# Main output
# ---------------------------------------------------------------------------------------------------------------------
io = MarilibIO()
json = io.to_json_file(ac,'aircraft_output_data')      # Write all output data into a json readable format

ac.draw.view_3d("This_plane")                           # Draw a 3D view diagram
ac.draw.payload_range("This_plot")                      # Draw a payload range diagram

# Print some relevant data
#------------------------------------------------------------------------------------------------------
disa = 0
altp = ac.requirement.cruise_altp
mach = ac.requirement.cruise_mach
vtas = earth.vtas_from_mach(altp,disa,mach)

print("")
print("Morphing = ",ac.airframe.wing.wing_morphing," ('aspect_ratio_driven' or 'span_driven')")
print("wing span = ","%.2f"%ac.airframe.wing.span," m")
print("wing aspect ratio = ","%.2f"%ac.airframe.wing.aspect_ratio," no_dim")
print("Wing area = ","%.2f"%ac.airframe.wing.area," m2")
print("Wing mass = ","%.2f"%ac.airframe.wing.mass," kg")

print("")
print("Initial MTOW = ","%.0f"%mtow_i," kg")
print("required MTOW = ","%.0f"%mtow_req," kg")

print("")
print("True air speed = ","%.2f"%unit.kt_mps(vtas)," kt")
print("Fuel mission = ","%.2f"%ac.performance.mission.nominal.fuel_block," kg")
print("LoD max = ","%.2f"%ac.aerodynamics.cruise_lodmax," no_dim")
print("LoD cruise = ","%.2f"%ac.performance.mission.crz_lod," no_dim")
print("TSFC cruise = ","%.3f"%(ac.performance.mission.crz_tsfc*36000)," kg/daN/h")
