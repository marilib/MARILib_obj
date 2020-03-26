#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from aircraft.airframe.airframe_root import Airframe

#--------------------------------------------------------------------------------------------------------------------------------
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
        self.economics = None
        self.environment = None




