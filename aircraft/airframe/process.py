#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve

import aircraft.tool.unit as unit
import earth

from aircraft.aircraft_root import Aircraft
from aircraft.performance import Performance

from aircraft.airframe import component
from aircraft.airframe import model




def factory(name="my_plane", reqs=None, agmt=None):
    """
    Build an aircraft
    :param name: the name of your aircraft
    :param reqs: requirement object
    :param agmt: arrangement object
    :return: an aircraft
    """

    ac = Aircraft(name, reqs, agmt)

    # plug a cabin component
    ac.airframe.cabin = component.Cabin(ac)

    if (ac.arrangement.body_type=="fuselage"):
        ac.airframe.body = component.Fuselage(ac)
    else:
        raise Exception("Type of body is unknown")

    if (ac.arrangement.wing_type=="classic"):
        ac.airframe.wing = component.Wing(ac)
    else:
        raise Exception("Type of wing is unknown")

    ac.airframe.cargo = component.Cargo(ac)

    if (ac.arrangement.stab_architecture=="classic"):
        ac.airframe.vertical_stab = component.VTP_classic(ac)
        ac.airframe.horizontal_stab = component.HTP_classic(ac)
    elif (ac.arrangement.stab_architecture=="t_tail"):
        ac.airframe.vertical_stab = component.VTP_T(ac)
        ac.airframe.horizontal_stab = component.HTP_T(ac)
    elif (ac.arrangement.stab_architecture=="h_tail"):
        ac.airframe.horizontal_stab = component.HTP_H(ac)
        ac.airframe.vertical_stab = component.VTP_H(ac)
    else:
        raise Exception("stab_architecture is unknown")

    if (ac.arrangement.tank_architecture=="wing_box"):
        ac.airframe.tank = component.Tank_wing_box(ac)
    elif (ac.arrangement.tank_architecture=="piggy_back"):
        ac.airframe.tank = component.Tank_piggy_back(ac)
    elif (ac.arrangement.tank_architecture=="pods"):
        ac.airframe.tank = component.Tank_wing_pod(ac)
    else:
        raise Exception("Type of tank is unknown")

    ac.airframe.landing_gear = component.Landing_gear(ac)

    ac.airframe.system = component.System(ac)

    if (ac.arrangement.power_architecture=="tf"):
        if (ac.arrangement.nacelle_attachment=="wing"):
            if (ac.arrangement.number_of_engine=="twin"):
                ac.airframe.nacelle = component.Inboard_wing_mounted_turbofan_nacelle(ac)
            elif (ac.arrangement.number_of_engine=="quadri"):
                ac.airframe.nacelle = component.Outboard_wing_mounted_turbofan_nacelle(ac)
                ac.airframe.internal_nacelle = component.Inboard_wing_mounted_turbofan_nacelle(ac)
            else:
                raise Exception("Number of engines not allowed")
        elif (ac.arrangement.nacelle_attachment=="rear"):
            if (ac.arrangement.number_of_engine=="twin"):
                ac.airframe.nacelle = component.Rear_mounted_turbofan_nacelle(ac)
            else:
                raise Exception("Number of engines not allowed")
        else:
            raise Exception("Type of nacelle attachment is unknown")
        ac.power_system = model.Turbofan(ac)
    elif (ac.arrangement.power_architecture=="tp"):
        pass
    elif (ac.arrangement.power_architecture=="pte1"):
        pass
    elif (ac.arrangement.power_architecture=="ef1"):
        pass
    elif (ac.arrangement.power_architecture=="ep1"):
        pass
    else:
        raise Exception("Type of power architecture is unknown")

    ac.airframe.mass_analysis_order = ["cabin",
                                       "body",
                                       "wing",
                                       "landing_gear",
                                       "cargo",
                                       "nacelle",
                                       "vertical_stab",
                                       "horizontal_stab",
                                       "tank",
                                       "system"]

    ac.aerodynamics = model.Aerodynamics(ac)

    ac.weight_cg = model.Weight_cg(ac)

    ac.performance = Performance(ac)

    return ac




