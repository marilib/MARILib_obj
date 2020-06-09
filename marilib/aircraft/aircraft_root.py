#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

from marilib.aircraft.airframe.airframe_root import Airframe
from marilib.aircraft.airframe import propulsion, component, system, model

from marilib.engine import interface

from marilib.aircraft.performance import Performance
from marilib.aircraft.mission import AllMissionVarMass, AllMissionIsoMass
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
                      power_architecture = "tf",       # "tf", "tp", "ef", "pte1", "ef1", "ep1",
                      power_source = "fuel",           # "fuel", "battery", "fuel_cell"
                      fuel_type = "kerosene"           # "kerosene", "liquid_h2", "Compressed_h2" or "battery"
                 ):

        self.body_type = body_type
        self.wing_type = wing_type
        self.wing_attachment = wing_attachment
        self.stab_architecture = stab_architecture
        self.tank_architecture = tank_architecture
        self.number_of_engine = number_of_engine
        self.nacelle_attachment = nacelle_attachment
        self.power_architecture = power_architecture
        self.power_source = power_source
        self.fuel_type = fuel_type


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

        self.requirement.init_all_requirements(arrangement)  # finalize the initialisation of all requirements.

        if (self.arrangement.power_architecture in ["ef","ep","exef"]):
            if(self.arrangement.power_source not in ["battery","fuel_cell"]):
                raise Exception("Power architecture electro_fan (ef) requires energy source battery or fuel_cell")

# ----------------------------------------------------------------------------------------------------------------------
        self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                             "nacelle","other_nacelle",
                                             "vertical_stab","horizontal_stab","tank","system"]

# ----------------------------------------------------------------------------------------------------------------------
        self.airframe.cabin = component.Cabin(self)

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.body_type=="fuselage"):
            self.airframe.body = component.Fuselage(self)
        else:
            raise Exception("Type of body is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.wing_type=="classic"):
            self.airframe.wing = component.Wing(self)
        else:
            raise Exception("Type of wing is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        self.airframe.cargo = component.Cargo(self)

# ----------------------------------------------------------------------------------------------------------------------
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

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.tank_architecture=="wing_box"):
            self.airframe.tank = component.TankWingBox(self)
        elif (self.arrangement.tank_architecture=="piggy_back"):
            self.airframe.tank = component.TankPiggyBack(self)
        elif (self.arrangement.tank_architecture=="pods"):
            self.airframe.tank = component.TankWingPod(self)
        else:
            raise Exception("Type of tank is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        self.airframe.landing_gear = component.LandingGear(self)

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.power_source == "battery"):
            self.airframe.system = system.SystemWithBattery(self)
        elif (self.arrangement.power_source == "fuel_cell"):
            self.airframe.system = system.SystemWithFuelCell(self)
        else:
            if (self.arrangement.power_architecture=="pte"):
                self.airframe.system = system.SystemPartialTurboElectric(self)
            else:
                self.airframe.system = system.System(self)

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.power_architecture=="tf0"):
            self.power_system = model.Turbofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTf0Nacelle(self)
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTf0Nacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTf0Nacelle(self)
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTf0Nacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTf0Nacelle(self)
                    self.airframe.left_internal_nacelle = propulsion.InboardWingMountedTf0Nacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTf0Nacelle(self)
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTf0Nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="tf"):
            self.power_system = model.Turbofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTfNacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTfNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTfNacelle(self)
                    self.airframe.left_internal_nacelle = propulsion.InboardWingMountedTfNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="pods"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.PodTailConeMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.PodTailConeMountedTfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="body_cones"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.BodyTailConeMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.PiggyBackTailConeMountedTfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="tp"):
            self.power_system = model.Turboprop(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTpNacelle(self)
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTpNacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTpNacelle(self)
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTpNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTpNacelle(self)
                    self.airframe.left_internal_nacelle = propulsion.InboardWingMountedTpNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is not allowed")

        elif (self.arrangement.power_architecture=="ep"):
            self.power_system = model.Electroprop(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedEpNacelle(self)
                    self.airframe.other_nacelle = propulsion.InboardWingMountedEpNacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedEpNacelle(self)
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedEpNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedEpNacelle(self)
                    self.airframe.left_internal_nacelle = propulsion.InboardWingMountedEpNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                elif (self.arrangement.number_of_engine=="hexa"):
                    self.airframe.nacelle = propulsion.ExternalWingMountedEpNacelle(self)
                    self.airframe.other_nacelle = propulsion.ExternalWingMountedEpNacelle(self)
                    self.airframe.median_nacelle = propulsion.OutboardWingMountedEpNacelle(self)
                    self.airframe.left_median_nacelle = propulsion.OutboardWingMountedEpNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedEpNacelle(self)
                    self.airframe.left_internal_nacelle = propulsion.InboardWingMountedEpNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","median_nacelle","left_median_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is not allowed")

        elif (self.arrangement.power_architecture=="ef"):
            self.power_system = model.Electrofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedEfNacelle(self)
                    self.airframe.other_nacelle = propulsion.InboardWingMountedEfNacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedEfNacelle(self)
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedEfNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedEfNacelle(self)
                    self.airframe.left_internal_nacelle = propulsion.InboardWingMountedEfNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedEfNacelle(self)
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedEfNacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="pte"):
            self.power_system = model.PartialTurboElectric(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTfNacelle(self)
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","tail_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTfNacelle(self)
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTfNacelle(self)
                    self.airframe.left_internal_nacelle = propulsion.InboardWingMountedTfNacelle(self)
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle","tail_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTfNacelle(self)
                    self.airframe.tail_nacelle = propulsion.FuselageTailConeMountedEfNacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","tail_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="extf"):
            self.power_system = model.Turbofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Inboard_wing_mounted_extf_nacelle(self)
                    self.airframe.other_nacelle = interface.Inboard_wing_mounted_extf_nacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = interface.Outboard_wing_mounted_extf_nacelle(self)
                    self.airframe.other_nacelle = interface.Outboard_wing_mounted_extf_nacelle(self)
                    self.airframe.internal_nacelle = interface.Inboard_wing_mounted_extf_nacelle(self)
                    self.airframe.left_internal_nacelle = interface.Inboard_wing_mounted_extf_nacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Rear_fuselage_mounted_extf_nacelle(self)
                    self.airframe.other_nacelle = interface.Rear_fuselage_mounted_extf_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="exef"):
            self.power_system = model.Electrofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Inboard_wing_mounted_exef_nacelle(self)
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = interface.Outboard_wing_mounted_exef_nacelle(self)
                    self.airframe.other_nacelle = interface.Outboard_wing_mounted_exef_nacelle(self)
                    self.airframe.internal_nacelle = interface.Inboard_wing_mounted_exef_nacelle(self)
                    self.airframe.left_internal_nacelle = interface.Inboard_wing_mounted_exef_nacelle(self)
                    self.airframe.mass_analysis_order = ["cabin","body","wing","landing_gear","cargo",
                                                         "nacelle","other_nacelle","internal_nacelle","left_internal_nacelle",
                                                         "vertical_stab","horizontal_stab","tank","system"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Rear_fuselage_mounted_exef_nacelle(self)
                    self.airframe.other_nacelle = interface.Rear_fuselage_mounted_exef_nacelle(self)
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

# ----------------------------------------------------------------------------------------------------------------------

        else:
            raise Exception("Type of power architecture is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        self.aerodynamics = model.Aerodynamics(self)

# ----------------------------------------------------------------------------------------------------------------------
        self.weight_cg = model.WeightCg(self)

# ----------------------------------------------------------------------------------------------------------------------
        self.performance = Performance(self)

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.power_source == "battery"):
            self.performance.mission = AllMissionIsoMass(self)
        else:
            self.performance.mission = AllMissionVarMass(self)

# ----------------------------------------------------------------------------------------------------------------------
        self.economics = Economics(self)

# ----------------------------------------------------------------------------------------------------------------------
        self.environment = Environment(self)


        return







