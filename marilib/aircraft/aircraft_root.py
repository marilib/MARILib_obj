#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from marilib.aircraft.airframe.airframe_root import Airframe
from marilib.aircraft.airframe import propulsion, component, model

from marilib.engine import interface

from marilib.aircraft.performance import Performance
from marilib.aircraft.mission import Mission, MissionIsoMass
from marilib.aircraft.environment import Economics
from marilib.aircraft.environment import Environment

from marilib.aircraft.tool.drawing import Drawing


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

        if (self.arrangement.power_architecture in ["efb","exefb"]):
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
            self.airframe.vertical_stab = component.VtpClassic(self)
            self.airframe.horizontal_stab = component.HtpClassic(self)
        elif (self.arrangement.stab_architecture=="t_tail"):
            self.airframe.vertical_stab = component.VtpTtail(self)
            self.airframe.horizontal_stab = component.HtpTtail(self)
        elif (self.arrangement.stab_architecture=="h_tail"):
            self.airframe.horizontal_stab = component.HtpHtail(self)
            self.airframe.vertical_stab = component.VtpHtail(self)
        else:
            raise Exception("stab_architecture is unknown")

        if (self.arrangement.tank_architecture=="wing_box"):
            self.airframe.tank = component.TankWingBox(self)
        elif (self.arrangement.tank_architecture=="piggy_back"):
            self.airframe.tank = component.TankPiggyBack(self)
        elif (self.arrangement.tank_architecture=="pods"):
            self.airframe.tank = component.TankWingPod(self)
        else:
            raise Exception("Type of tank is unknown")

        self.airframe.landing_gear = component.LandingGear(self)

        if (self.arrangement.power_architecture in ["efb","exefb"]):
            self.airframe.system = propulsion.SystemElectrofanBattery(self)
        else:
            self.airframe.system = propulsion.System(self)

        if (self.arrangement.power_architecture=="tf"):
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTfNacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTfNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")
            self.power_system = model.Turbofan(self)

        elif (self.arrangement.power_architecture=="tp"):
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTpNacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTpNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTpNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is not allowed")
            self.power_system = model.Turboprop(self)

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
                    self.airframe.nacelle = propulsion.InboardWingMountedEfNacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedEfNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedEfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedEfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")
            self.power_system = model.Electrofan(self)

        elif (self.arrangement.power_architecture=="exefb"):
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Inboard_wing_mounted_exef_nacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = interface.Outboard_wing_mounted_exef_nacelle(self)
                    self.airframe.internal_nacelle = interface.Inboard_wing_mounted_exef_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Rear_fuselage_mounted_exef_nacelle(self)
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

        self.weight_cg = model.WeightCg(self)

        self.performance = Performance(self)

        if (self.arrangement.power_architecture in ["tf","extf","tp"]):
            self.performance.mission = Mission(self)
        elif (self.arrangement.power_architecture in ["efb","exefb"]):
            self.performance.mission = MissionIsoMass(self)
        else:
            raise Exception("Type of power architecture is unknown")

        self.economics = Economics(self)

        self.environment = Environment(self)


        return







