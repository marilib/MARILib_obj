#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import unit
import earth

from aircraft.requirement import Requirement
from aircraft.arrangement import Arrangement
import aircraft.airframe.process as process

agmt = Arrangement(body_type = "fuselage",          # "fuselage" or "blended"
                   wing_type = "classic",           # "classic" or "blended"
                   wing_attachment = "low",         # "low" or "high"
                   stab_architecture = "classic",   # "classic", "t_tail" or "h_tail"
                   tank_architecture = "wing_box",  # "wing_box", "piggy_back" or "pods"
                   number_of_engine = "twin",       # "twin" or "quadri"
                   nacelle_attachment = "wing",     # "wing", "rear" or "pods"
                   power_architecture = "tf",       # "tf", "pf", "pte1", "ef1", "ep1",
                   energy_source = "kerosene")      # "kerosene", "methane", "liquid_h2", "700bar_h2" or "battery"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.),
                   arrangement = agmt)


ac = process.factory(name = "my_plane", reqs = reqs, agmt = agmt)

# ac.airframe.geometry_analysis(ac)
ac.airframe.statistical_pre_design()

# ac.weight_cg.mass_analysis()
ac.weight_cg.mass_pre_design()

ac.power_system.thrust_analysis()
