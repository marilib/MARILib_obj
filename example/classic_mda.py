#!/usr/bin/env python3
"""
Classic MDA example
date : 06/2020

A basic usage example of MARILib, introduction to some usefull functions.
Let's say we want to design a transport airplane that fulfills the following Top Level Aircraft Requirements (TLARS):

* capacity : 150 passengers
* design range : 3000 NM
* cruise Mach : 0.78
* cruise altitude :  35 000 ft

"""
from marilib.aircraft import Arrangement, Requirement, Aircraft, mda
from marilib.utils import unit
from marilib.utils.read_write import MarilibIO

# User inputs : define the arrangement and TLARS
agmt = Arrangement(body_type = "fuselage",          # "fuselage" or "blended"
                   wing_type = "classic",           # "classic" or "blended"
                   wing_attachment = "low",         # "low" or "high"
                   stab_architecture = "classic",   # "classic", "t_tail" or "h_tail"
                   tank_architecture = "wing_box",  # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",       # "twin" or "quadri"
                   nacelle_attachment = "wing",     # "wing", "rear" or "pods"
                   power_architecture = "tf0",       # "tf", "extf", "ef", "exef", "tp", "ep"
                   power_source = "fuel",           # "fuel", "battery", "fuel_cell"
                   fuel_type = "kerosene")          # "kerosene", "methane", "liquid_h2", "700bar_h2", "battery"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.))

# Initialise the aircraft
ac = Aircraft("best_airplane_ever")
ac.factory(agmt, reqs)  # WARNING : arrangement must not be changed after this line

# Change some value
ac.requirement.take_off.tofl_req = 2500.  # change the Take Off Field Length to 2500 meters.
ac.airframe.nacelle.reference_thrust = unit.N_kN(120.)  # change the reference thrust to 120 000 N
ac.airframe.wing.area = 122.  # Set the wing area to 122 m2

# Run a Multidisciplinary analysis
mda(ac)

# Display Payload range
ac.draw.payload_range("My Payload-Range diagram")

# Display a representation of the aircraft.
ac.draw.view_3d("View of " + ac.name)

print(ac.airframe.cabin.m_furnishing)
# Save a written representation of all aircraft data
io = MarilibIO()  # instance of the input/output module
io.to_json_file(ac,"best_airplane_ever")  # write human readable output to the file 'best_airplane_ever.json'
# Load from existing file
#fakecopy = io.from_json_file('best_airplane_ever')  # WARNING, from_json_file IS NOT an exact copy :

# Save an exact copy of the current airplane
io.to_binary_file(ac,'best_airplane_ever')  # write binary output to the file 'best_airplane_ever.pkl'
exact_copy = io.from_binary_file('best_airplane_ever.pkl')  # load an exact copy of the original
