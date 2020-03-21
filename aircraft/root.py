#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from aircraft.requirement import Requirement
from aircraft.arrangement import Arrangement



#--------------------------------------------------------------------------------------------------------------------------------
class Airframe(object):
    """
    Logical aircraft description
    """
    pass


class Weight_cg(object):

    def __init__(self, requirement):

        n_pax_ref = requirement.n_pax_ref
        design_range = requirement.design_range

        self.mtow = 20500. + 67.e-6*n_pax_ref*design_range
        self.mzfw = 25000. + 41.e-6*n_pax_ref*design_range
        self.mlw = 1.07*self.mzfw
        self.owe = None
        self.mwe = None


class Aerodynamics(object):

    def __init__(self, requirement):

        self.hld_conf_to = 0.30
        self.hld_conf_ld = 1.00


#--------------------------------------------------------------------------------------------------------------------------------
class Aircraft(object):
    """
    Logical aircraft description
    """
    def __init__(self, requirement, arrangement):
        """
        Data structure, only one sub-level allowed
        """
        self.name = None
        self.requirement = requirement
        self.arrangement = arrangement

        self.airframe = Airframe()

        self.payload = None
        self.power_system = None
        self.aerodynamics = Aerodynamics(requirement)
        self.weight_cg = Weight_cg(requirement)
        self.economics = None
        self.environment = None

        # attributs variables
        # self.cabin = None
        # self.main_body = None
        # self.wing = None
        # self.stabilizer = None
        # self.tank = None
        # self.landing_gear = None
        #
        # self.system = None



#       geom, mass, aero,
#       reglementation dans  requirements
