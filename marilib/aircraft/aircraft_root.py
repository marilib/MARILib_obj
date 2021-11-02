#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""
from marilib.utils import earth, unit

from marilib.aircraft.airframe.airframe_root import Airframe
from marilib.aircraft.airframe import propulsion, component, system, model

from marilib.engine import interface

from marilib.aircraft.handling_quality import HandlingQuality
from marilib.aircraft.performance import Performance
from marilib.aircraft.mission import AllMissionVarMass, AllMissionIsoMass
from marilib.aircraft.environment import Environment, Economics

from marilib.aircraft.tool.drawing import Drawing




class Arrangement(object):
    """Architectural choices
    """
    def __init__(self,body_type = "fuselage",               # "fuselage" or "blended"
                      wing_type = "classic",                # "classic" or "blended"
                      wing_attachment = "low",              # "low" or "high"
                      stab_architecture = "classic",        # "classic", "t_tail" or "h_tail"
                      tank_architecture = "wing_box",       # "wing_box", "piggy_back" or "pods"
                      gear_architecture = "retractable",    # "retractable", "bare_fixed"
                      number_of_engine = "twin",            # "twin" or "quadri"
                      nacelle_attachment = "wing",          # "wing", "pod" "rear" or "body_cones"
                      power_architecture = "tf",            # "tf", "tp", "ef", "pte1", "ef1", "ep1",
                      power_source = "fuel",                # "fuel", "battery", "fuel_cell", "fuel_cell_plus"
                      fuel_type = "kerosene"                # "kerosene", "liquid_h2", "Compressed_h2" or "battery"
                 ):

        self.body_type = body_type
        self.wing_type = wing_type
        self.wing_attachment = wing_attachment
        self.stab_architecture = stab_architecture
        self.tank_architecture = tank_architecture
        self.gear_architecture = gear_architecture
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
        self.handling_quality = None
        self.performance = None
        self.economics = None
        self.environment = None
        self.draw = Drawing(self)

    def get_init(self, obj,key,val=None):
        return self.requirement.model_config.get__init(obj,key,val=val)

    def factory(self, arrangement, requirement):
        """Build an aircraft according to architectural choices
        """
        self.requirement = requirement
        self.arrangement = arrangement

        self.requirement.init_all_requirements(arrangement)  # finalize the initialisation of all requirements.

        if (self.arrangement.power_architecture in ["ef","ep","exef"]):
            if(self.arrangement.power_source != "battery" and self.arrangement.power_source[0:9] != "fuel_cell"):
                raise Exception("Power architecture electro_fan (ef) requires energy source battery or fuel_cell")

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
            self.airframe.vertical_stab = component.VtpHtail(self,"right")
            self.airframe.other_vertical_stab = component.VtpHtail(self,"left")
        else:
            raise Exception("stab_architecture is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.tank_architecture=="wing_box"):
            self.airframe.tank = component.TankWingBox(self)
            self.airframe.tank_analysis_order = ["tank"]
        elif (self.arrangement.tank_architecture=="floor"):
            self.airframe.tank = component.TankFuselageFloor(self)
            self.airframe.tank_analysis_order = ["tank"]
        elif (self.arrangement.tank_architecture=="rear"):
            self.airframe.tank = component.TankRearFuselage(self)
            self.airframe.tank_analysis_order = ["tank"]
        elif (self.arrangement.tank_architecture=="piggy_back"):
            self.airframe.tank = component.TankPiggyBack(self)
            self.airframe.tank_analysis_order = ["tank"]
        elif (self.arrangement.tank_architecture=="pods"):
            self.airframe.tank = component.TankWingPod(self,"right")
            self.airframe.other_tank = component.TankWingPod(self,"left")
            self.airframe.tank_analysis_order = ["tank","other_tank"]
        else:
            raise Exception("Type of tank is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.gear_architecture=="retractable"):
            self.airframe.landing_gear = component.RetractableLandingGear(self)
        elif (self.arrangement.gear_architecture=="bare_fixed"):
            self.airframe.landing_gear = component.BareFixedLandingGear(self)
        else:
            raise Exception("Type of landing gear is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.power_source == "battery"):
            self.airframe.system = system.SystemWithBattery(self)
        elif (self.arrangement.power_source == "fuel_cell"):
            self.airframe.system = system.SystemWithFuelCell(self)
        elif (self.arrangement.power_source[0:13] == "fuel_cell_PEM"):
            self.airframe.system = system.SystemWithLaplaceFuelCell(self)
        else:
            if (self.arrangement.power_architecture=="pte"):
                self.airframe.system = system.SystemPartialTurboElectric(self)
            elif (self.arrangement.power_architecture=="pte_pod"):
                self.airframe.system = system.SystemPartialTurboElectricPods(self)
            elif (self.arrangement.power_architecture=="pte_piggy"):
                self.airframe.system = system.SystemPartialTurboElectricPiggyBack(self)
            else:
                self.airframe.system = system.System(self)

# ----------------------------------------------------------------------------------------------------------------------
        if (self.arrangement.power_architecture=="tf0"):
            self.power_system = model.Turbofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTf0Nacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTf0Nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTf0Nacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTf0Nacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTf0Nacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedTf0Nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTf0Nacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTf0Nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="tf"):
            self.power_system = model.Turbofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTfNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTfNacelle(self,"left")
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="pods"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.PodTailConeMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.PodTailConeMountedTfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="body_cones"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.BodyTailConeMountedTfNacelle(self)
                    self.airframe.other_nacelle = propulsion.PiggyBackTailConeMountedTfNacelle(self)
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="tp"):
            self.power_system = model.Turboprop(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTpNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTpNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTpNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTpNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTpNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedTpNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is not allowed")

        elif (self.arrangement.power_architecture=="ep"):
            self.power_system = model.Electroprop(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedEpNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedEpNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedEpNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedEpNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedEpNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedEpNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle"]
                elif (self.arrangement.number_of_engine=="hexa"):
                    self.airframe.nacelle = propulsion.ExternalWingMountedEpNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.ExternalWingMountedEpNacelle(self,"left")
                    self.airframe.median_nacelle = propulsion.OutboardWingMountedEpNacelle(self,"right")
                    self.airframe.other_median_nacelle = propulsion.OutboardWingMountedEpNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedEpNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedEpNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","median_nacelle","other_median_nacelle","internal_nacelle","other_internal_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is not allowed")

        elif (self.arrangement.power_architecture=="ef"):
            self.power_system = model.Electrofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedEfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedEfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedEfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedEfNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedEfNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedEfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedEfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedEfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="pte"):
            self.power_system = model.PartialTurboElectric(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","tail_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTfNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle","tail_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTfNacelle(self,"left")
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","tail_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="pte_pod"):
            self.power_system = model.PartialTurboElectricPods(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.pod_tail_nacelle = propulsion.PodTailConeMountedEfNacelle(self,"right")
                    self.airframe.other_pod_tail_nacelle = propulsion.PodTailConeMountedEfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","tail_nacelle","pod_tail_nacelle","other_pod_tail_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTfNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.pod_tail_nacelle = propulsion.PodTailConeMountedEfNacelle(self,"right")
                    self.airframe.other_pod_tail_nacelle = propulsion.PodTailConeMountedEfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle","pod_tail_nacelle","other_pod_tail_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTfNacelle(self,"left")
                    self.airframe.pod_tail_nacelle = propulsion.PodTailConeMountedEfNacelle(self,"right")
                    self.airframe.other_pod_tail_nacelle = propulsion.PodTailConeMountedEfNacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","tail_nacelle","pod_tail_nacelle","other_pod_tail_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment not allowed")

        elif (self.arrangement.power_architecture=="pte_piggy"):
            self.power_system = model.PartialTurboElectricPiggyBack(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.other_tail_nacelle = propulsion.PiggyBackTailConeMountedEfNacelle(self)
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","tail_nacelle","other_tail_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = propulsion.OutboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.OutboardWingMountedTfNacelle(self,"left")
                    self.airframe.internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"right")
                    self.airframe.other_internal_nacelle = propulsion.InboardWingMountedTfNacelle(self,"left")
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.other_tail_nacelle = propulsion.PiggyBackTailConeMountedEfNacelle(self)
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle","tail_nacelle","other_tail_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = propulsion.RearFuselageMountedTfNacelle(self,"right")
                    self.airframe.other_nacelle = propulsion.RearFuselageMountedTfNacelle(self,"left")
                    self.airframe.tail_nacelle = propulsion.BodyTailConeMountedEfNacelle(self)
                    self.airframe.other_tail_nacelle = propulsion.PiggyBackTailConeMountedEfNacelle(self)
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","tail_nacelle","other_tail_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment not allowed")

        elif (self.arrangement.power_architecture=="extf"):
            self.power_system = model.Turbofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Inboard_wing_mounted_extf_nacelle(self,"right")
                    self.airframe.other_nacelle = interface.Inboard_wing_mounted_extf_nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = interface.Outboard_wing_mounted_extf_nacelle(self,"right")
                    self.airframe.other_nacelle = interface.Outboard_wing_mounted_extf_nacelle(self,"left")
                    self.airframe.internal_nacelle = interface.Inboard_wing_mounted_extf_nacelle(self,"right")
                    self.airframe.other_internal_nacelle = interface.Inboard_wing_mounted_extf_nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Rear_fuselage_mounted_extf_nacelle(self,"right")
                    self.airframe.other_nacelle = interface.Rear_fuselage_mounted_extf_nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

        elif (self.arrangement.power_architecture=="exef"):
            self.power_system = model.Electrofan(self)
            if (self.arrangement.nacelle_attachment=="wing"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Inboard_wing_mounted_exef_nacelle(self,"right")
                    self.airframe.other_nacelle = interface.Inboard_wing_mounted_exef_nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                elif (self.arrangement.number_of_engine=="quadri"):
                    self.airframe.nacelle = interface.Outboard_wing_mounted_exef_nacelle(self,"right")
                    self.airframe.other_nacelle = interface.Outboard_wing_mounted_exef_nacelle(self,"left")
                    self.airframe.internal_nacelle = interface.Inboard_wing_mounted_exef_nacelle(self,"right")
                    self.airframe.other_internal_nacelle = interface.Inboard_wing_mounted_exef_nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle","internal_nacelle","other_internal_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            elif (self.arrangement.nacelle_attachment=="rear"):
                if (self.arrangement.number_of_engine=="twin"):
                    self.airframe.nacelle = interface.Rear_fuselage_mounted_exef_nacelle(self,"right")
                    self.airframe.other_nacelle = interface.Rear_fuselage_mounted_exef_nacelle(self,"left")
                    self.airframe.engine_analysis_order = ["nacelle","other_nacelle"]
                else:
                    raise Exception("Number of engines not allowed")
            else:
                raise Exception("Type of nacelle attachment is unknown")

# ----------------------------------------------------------------------------------------------------------------------

        else:
            raise Exception("Type of power architecture is unknown")

# ----------------------------------------------------------------------------------------------------------------------
        self.airframe.mass_analysis_order =   ["cabin","body","wing","landing_gear","cargo"] \
                                            + self.airframe.engine_analysis_order \
                                            + ["vertical_stab","horizontal_stab"] \
                                            + self.airframe.tank_analysis_order \
                                            + ["system"]

# ----------------------------------------------------------------------------------------------------------------------
        self.aerodynamics = model.Aerodynamics(self)

# ----------------------------------------------------------------------------------------------------------------------
        self.weight_cg = model.WeightCg(self)

# ----------------------------------------------------------------------------------------------------------------------
        self.handling_quality = HandlingQuality(self)

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







