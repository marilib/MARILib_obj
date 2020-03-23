#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import unit
from aircraft.requirement import Requirement
from aircraft.arrangement import Arrangement
from aircraft.root import Aircraft

from aircraft.airframe import component

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



def factory(name = "my_plane", reqs=None, agmt=None):
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

    ac.airframe.cargo = component.Cargo_hold(ac)

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
        ac.airframe.nacelle = component.Turbofan(ac)
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

    return ac


def pre_design(ac):

    ac.airframe.cabin.eval_geometry()
    ac.airframe.body.eval_geometry()
    ac.airframe.wing.eval_geometry()
    ac.airframe.cargo.eval_geometry()
    ac.airframe.nacelle.eval_geometry()

    if (ac.arrangement.stab_architecture in ["classic","t_tail"]):
        ac.airframe.vertical_stab.eval_geometry()
        ac.airframe.horizontal_stab.eval_geometry()
    elif (ac.arrangement.stab_architecture=="h_tail"):
        ac.airframe.horizontal_stab.eval_geometry()
        ac.airframe.vertical_stab.eval_geometry()
    else:
        raise Exception("stab_architecture is unknown")

    ac.airframe.vertical_stab.eval_area()
    ac.airframe.horizontal_stab.eval_area()

    ac.airframe.tank.eval_geometry()
    ac.airframe.landing_gear.eval_geometry()
    ac.airframe.system.eval_geometry()


def mass_analysis(ac):

    ac.airframe.cabin.eval_mass()
    ac.airframe.body.eval_mass()
    ac.airframe.wing.eval_mass()
    ac.airframe.cargo.eval_mass()
    ac.airframe.nacelle.eval_mass()
    ac.airframe.vertical_stab.eval_mass()
    ac.airframe.horizontal_stab.eval_mass()
    ac.airframe.tank.eval_mass()
    ac.airframe.landing_gear.eval_mass()
    ac.airframe.system.eval_mass()


ac = factory(name = "my_plane", reqs = reqs, agmt = agmt)

pre_design(ac)

mass_analysis(ac)
