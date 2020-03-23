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

    def __init__(self, aircraft):
        self.aircraft = aircraft

    def geometry_analysis(self):
        stab_architecture = self.aircraft.arrangement.stab_architecture

        self.cabin.eval_geometry()
        self.body.eval_geometry()
        self.wing.eval_geometry()
        self.cargo.eval_geometry()
        self.nacelle.eval_geometry()

        if (stab_architecture in ["classic","t_tail"]):
            self.vertical_stab.eval_geometry()
            self.horizontal_stab.eval_geometry()
        elif (stab_architecture=="h_tail"):
            self.horizontal_stab.eval_geometry()
            self.vertical_stab.eval_geometry()

        self.vertical_stab.eval_area()
        self.horizontal_stab.eval_area()

        self.tank.eval_geometry()
        self.landing_gear.eval_geometry()
        self.system.eval_geometry()

    def geometry_pre_design(self):








    def mass_analysis(self):
        self.cabin.eval_mass()
        self.body.eval_mass()
        self.wing.eval_mass()
        self.cargo.eval_mass()
        self.nacelle.eval_mass()
        self.vertical_stab.eval_mass()
        self.horizontal_stab.eval_mass()
        self.tank.eval_mass()
        self.landing_gear.eval_mass()
        self.system.eval_mass()


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
    def __init__(self, name, requirement, arrangement):
        """
        Data structure, only one sub-level allowed
        """
        self.name = name
        self.requirement = requirement
        self.arrangement = arrangement

        self.airframe = Airframe(self)

        self.payload = None
        self.power_system = None
        self.aerodynamics = Aerodynamics(requirement)
        self.weight_cg = Weight_cg(requirement)
        self.economics = None
        self.environment = None




