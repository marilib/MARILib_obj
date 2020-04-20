#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from aircraft.airframe.airframe_root import Airframe
from aircraft.airframe import component, power_and_system
from aircraft.airframe import model

from engine import interface

from aircraft.performance import Performance
from aircraft.mission import Mission, E_mission
from aircraft.environment import Economics
from aircraft.environment import Environment

from aircraft.tool.drawing import Drawing


class Arrangement(object):
    """Architectural choices
    """
    def __init__(self,body_type = "fuselage",          # "fuselage" or "blended"
                      wing_type = "classic",           # "classic" or "blended"
                      wing_attachment = "low",         # "low" or "high"
                      stab_architecture = "classic",   # "classic", "t_tail" or "h_tail"
                      tank_architecture = "wing_box",  # "wing_box", "piggy_back" or "pods"
                      number_of_engine = "twin",       # "twin" or "quadri"
                      nacelle_attachment = "wing",     # "wing", "pod" or "rear"
                      power_architecture = "tf",       # "tf", "tp", "efb", "pte1", "ef1", "ep1",
                      energy_source = "kerosene"       # "kerosene", "methane", "liquid_h2", "700bar_h2", "battery", fuel_cell
                 ):

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
    """Logical aircraft description
    """
    def __init__(self, name):

        self.name = name
        self.arrangement = None
        self.requirement = None

        self.airframe = Airframe(self)

        self.power_system = None
        self.aerodynamics = None
        self.weight_cg = None
        self.performance = None
        self.economics = None
        self.environment = None
        self.draw = Drawing(self)

    def factory(self, arrangement, requirement):
        """Build an aircraft according to architectural choices
        """
        self.requirement = requirement
        self.arrangement = arrangement

        if (self.arrangement.power_architecture=="efb"):
            if(self.arrangement.energy_source!="battery"):

                raise Exception("Power architecture electro_fan (ef) requires energy source battery or fuel_cell")

        # plug a cabin component
        self.airframe.cabin = component.Cabin(self)

        if (self.arrangement.body_type=="fuselage"):
            self.airframe.body = component.Fuselage(self)
        else:
            raise Exception("Type of body is unknown")

        if (self.arrangement.wing_type=="classic"):
            self.airframe.wing = component.Wing(self)
        else:
            raise Exception("Type of wing is unknown")

        self.airframe.cargo = component.Cargo(self)

        if (self.arrangement.stab_architecture=="classic"):
            self.airframe.vertical_stab = component.VTP_classic(self)
            self.airframe.horizontal_stab = component.HTP_classic(self)
        elif (self.arrangement.stab_architecture=="t_tail"):
            self.airframe.vertical_stab = component.VTP_T(self)
            self.airframe.horizontal_stab = component.HTP_T(self)
        elif (self.arrangement.stab_architecture=="h_tail"):
            self.airframe.horizontal_stab = component.HTP_H(self)
            self.airframe.vertical_stab = component.VTP_H(self)
        else:
            raise Exception("stab_architecture is unknown")

        if (self.arrangement.tank_architecture=="wing_box"):
            self.airframe.tank = component.Tank_wing_box(self)
        elif (self.arrangement.tank_architecture=="piggy_back"):
            self.airframe.tank = component.Tank_piggy_back(self)
        elif (self.arrangement.tank_architecture=="pods"):
            self.airframe.tank = component.Tank_wing_pod(self)
        else:
            raise Exception("Type of tank is unknown")

        self.airframe.landing_gear = component.Landing_gear(self)

        if (self.arrangement.power_architecture=="efb"):
            self.airframe.system = power_and_system.System_efb(self)
        else:
            self.airframe.system = power_and_system.System(self)

        if (self.arrangement.power_architecture=="tf"):
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = power_and_system.Inboard_wing_mounted_tf_nacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = power_and_system.Outboard_wing_mounted_tf_nacelle(self)
                    self.airframe.internal_nacelle = power_and_system.Inboard_wing_mounted_tf_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = power_and_system.Rear_fuselage_mounted_tf_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")
            self.power_system = model.Turbofan(self)

        elif (self.arrangement.power_architecture=="extf"):
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Inboard_wing_mounted_extf_nacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = interface.Outboard_wing_mounted_extf_nacelle(self)
                    self.airframe.internal_nacelle = interface.Inboard_wing_mounted_extf_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Rear_fuselage_mounted_extf_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")
            self.power_system = model.Turbofan(self)

        elif (self.arrangement.power_architecture=="efb"):
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = power_and_system.Inboard_wing_mounted_ef_nacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = power_and_system.Outboard_wing_mounted_ef_nacelle(self)
                    self.airframe.internal_nacelle = power_and_system.Inboard_wing_mounted_ef_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = power_and_system.Rear_fuselage_mounted_ef_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")
            self.power_system = model.Electrofan(self)

        else:
            raise Exception("Type of power architecture is unknown")

        self.airframe.mass_analysis_order = ["cabin",
                                           "body",
                                           "wing",
                                           "landing_gear",
                                           "cargo",
                                           "nacelle",
                                           "vertical_stab",
                                           "horizontal_stab",
                                           "tank",
                                           "system"]

        self.aerodynamics = model.Aerodynamics(self)

        self.weight_cg = model.Weight_cg(self)

        self.performance = Performance(self)

        if (self.arrangement.power_architecture=="tf"):
            self.performance.mission = Mission(self)
        elif (self.arrangement.power_architecture=="efb"):
            self.performance.mission = E_mission(self)
        else:
            raise Exception("Type of power architecture is unknown")

        self.economics = Economics(self)

        self.environment = Environment(self)


        return







