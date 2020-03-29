#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from aircraft.airframe.airframe_root import Airframe
from aircraft.tool.drawing import Drawing

class Arrangement(object):
    """
    Top level aircraft requirements
    """
    def __init__(self,body_type = "fuselage",          # "fuselage" or "blended"
                      wing_type = "classic",           # "classic" or "blended"
                      wing_attachment = "low",         # "low" or "high"
                      stab_architecture = "classic",   # "classic", "t_tail" or "h_tail"
                      tank_architecture = "wing_box",  # "wing_box", "piggy_back" or "pods"
                      number_of_engine = "twin",       # "twin" or "quadri"
                      nacelle_attachment = "wing",     # "wing", "pod" or "rear"
                      power_architecture = "tf",       # "tf", "tp", "pte1", "ef1", "ep1",
                      energy_source = "kerosene"       # "kerosene", "methane", "liquid_h2", "700bar_h2" or "battery"
                 ):
        """
        Data structure, only one sub-level allowed
        """
        self.body_type = body_type
        self.wing_type = wing_type
        self.wing_attachment = wing_attachment
        self.stab_architecture = stab_architecture
        self.tank_architecture = tank_architecture
        self.number_of_engine = number_of_engine
        self.nacelle_attachment = nacelle_attachment
        self.power_architecture = power_architecture
        self.energy_source = energy_source


class Aircraft(object):
    """
    Logical aircraft description
    """
    def __init__(self, name, requirement, arrangement):
        """
        Data structure, only one sub-level allowed
        """
        self.name = name
        self.requirement = requirement
        self.arrangement = arrangement

        self.airframe = Airframe(self)

        self.power_system = None
        self.aerodynamics = None
        self.weight_cg = None
        self.performance = None
        self.economics = None
        self.environment = None
        self.draw = Drawing(self)




