#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve, least_squares

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

import unit
from physical_data import PhysicalData
from models import DDM



#-----------------------------------------------------------------------------------------------------------------------
#
#  Power system analysis
#
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    phd = PhysicalData()
    ddm = DDM(phd)

    # # Airplane design analysis
    # #-------------------------------------------------------------------------------------------------------------------
    # npax = 6
    # distance = unit.convert_from("km", 500)
    # cruise_speed = unit.convert_from("km/h", 180)
    #
    # airplane_type = "general"
    # initial_power_system = {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.kerosene}
    #
    # altitude_data = ddm.cruise_altp(airplane_type)
    # reserve_data = ddm.reserve_data(airplane_type)
    #
    # tpws = [{"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.kerosene},
    #         {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.gh2},
    #         {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.lh2},
    #         {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery},
    #         {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.gh2},
    #         {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.lh2}]
    #
    # for target_power_system in tpws:
    #     ac_dict = ddm.design(npax, distance, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict)
    #
    #     ac_dict = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")
    #
    #     ac_dict = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")
    #


    # # Airplane design analysis
    # #-------------------------------------------------------------------------------------------------------------------
    # npax = 48
    # distance = unit.convert_from("km", 1300)
    # cruise_speed = unit.convert_from("km/h", 500)
    #
    # airplane_type = "commuter"
    # initial_power_system = {"thruster":ddm.propeller, "engine_type":ddm.turboprop, "energy_source":ddm.kerosene}
    #
    # altitude_data = ddm.cruise_altp(airplane_type)
    # reserve_data = ddm.reserve_data(airplane_type)
    #
    # tpws = [{"thruster":ddm.propeller, "engine_type":ddm.turboprop, "energy_source":ddm.kerosene},
    #         {"thruster":ddm.propeller, "engine_type":ddm.turboprop, "energy_source":ddm.gh2},
    #         {"thruster":ddm.propeller, "engine_type":ddm.turboprop, "energy_source":ddm.lh2},
    #         {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery},
    #         {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.gh2},
    #         {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.lh2}]
    #
    # for target_power_system in tpws:
    #     ac_dict = ddm.design(npax, distance, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict)
    #
    #     ac_dict = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")
    #
    #     ac_dict = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")
    #

    # Airplane design analysis
    #-------------------------------------------------------------------------------------------------------------------
    npax = 150
    distance = unit.convert_from("NM", 3000)
    cruise_speed = 0.78

    airplane_type = "narrow_body"
    initial_power_system = {"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.kerosene}

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)

    tpws = [{"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.kerosene},
            {"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.gh2},
            {"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.lh2}]

    for target_power_system in tpws:
        ac_dict = ddm.design(npax, distance, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
        ddm.print_design(ac_dict)

        ac_dict = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
        ddm.print_design(ac_dict, content="criteria")

        ac_dict = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
        ddm.print_design(ac_dict, content="criteria")


