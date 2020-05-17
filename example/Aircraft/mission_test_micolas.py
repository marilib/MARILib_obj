#!/usr/bin/env python3
"""
Created on Wed Apr 29 11:11:11 2020

..author: DRUOT Thierry, Nicolas Monrolin
"""

from marilib.context import unit
from marilib.aircraft.aircraft_root import Arrangement
from marilib.aircraft.aircraft_root import Aircraft
from marilib.aircraft.requirement import Requirement

from marilib.aircraft.mission import MissionDef

from marilib.aircraft.design import process

agmt = Arrangement(body_type = "fuselage",          # "fuselage" or "blended"
                   wing_type = "classic",           # "classic" or "blended"
                   wing_attachment = "low",         # "low" or "high"
                   stab_architecture = "classic",   # "classic", "t_tail" or "h_tail"
                   tank_architecture = "wing_box",  # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",       # "twin" or "quadri"
                   nacelle_attachment = "wing",     # "wing", "rear" or "pods"
                   power_architecture = "tf",       # "tf", "extf", "efb", "exefb",
                   energy_source = "kerosene")      # "kerosene", "methane", "liquid_h2", "700bar_h2", "battery" or "fuel_cell"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.),
                   arrangement = agmt)


ac = Aircraft("This_plane")
ac.factory(agmt, reqs)  # WARNING : arrangement must not be changed after this line


process.mda(ac)

mymission = MissionDef(ac)

mymission.set_parameters() # set default requirement parameters

result = mymission.eval(inputs={'range':3000.*1.854,'tow':64000.}) # TODO Check result with Thierry

print(result)



