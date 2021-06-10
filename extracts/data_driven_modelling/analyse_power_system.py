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

import matplotlib.colors as colors


#-----------------------------------------------------------------------------------------------------------------------
#
#  Power system analysis
#
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    phd = PhysicalData()
    ddm = DDM(phd)

    # Airplane design analysis
    #-------------------------------------------------------------------------------------------------------------------
    npax = 2                                        # To vary from 2 to 9, pax_list = [2, 3, 4,  5, 6, 7, 8, 9]
    n_engine = 1
    distance = unit.convert_from("km", 130)         # To vary between 100 km and 1000km by step of 50, dist_list = np.linspace(100, 1000, 19)
    cruise_speed = unit.convert_from("km/h", 130)   # Fixed

    ddm.battery_enrg_density = unit.J_Wh(200)  # Wh/kg  # To varry : 200, 400 and 600

    airplane_type = "general"   # Fixed
    initial_power_system = {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.kerosene}

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)

    tpws = [{"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.kerosene},
            {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.gh2},
            {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.lh2},
            {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery},
            {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.gh2},
            {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.lh2}]

    target_power_system = {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery}
    dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system)
    ddm.print_design(dict)


    # for target_power_system in tpws:
    #     dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system)
    #     ddm.print_design(dict)


    # npax_max = 19
    # dist_max = 1500
    # dist_step = 20
    # table = np.zeros((npax_max-1, int(dist_max/dist_step)))
    # # for target_power_system,num in zip(tpws, [1, 2, 3, 4, 5, 6]):
    # target_power_system = {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery}
    # for pw_density in [200, 300, 400, 500, 600]:
    #     ddm.battery_enrg_density = unit.J_Wh(pw_density)  # Wh/kg  # To varry : 200, 400 and 600
    #     prop = []
    #     for npax in range(2,npax_max+1):
    #         ind = []
    #         for dist in np.linspace(dist_step, dist_max, np.ceil(dist_max/dist_step)):
    #             dict = ddm.design_airplane(npax, unit.m_km(dist), cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system)
    #             # ddm.print_design(ac_dict)
    #             # if 0.0146 / (dict["npax"]/unit.km_m(dict["design_range"])) > 1 and \
    #             #    (dict["pk_o_mass"]/dict["pk_o_mass_mini"]) > 1 and \
    #             #     dict["mtow"] < 5700 :
    #
    #             if (dict["pk_o_mass"]/dict["pk_o_mass_mini"]) > 1 and \
    #                 dict["mtow"] < 5700 :
    #                 ind.append(1)
    #             else:
    #                 ind.append(0)
    #         prop.append(ind)
    #     table += np.array(prop)
    #
    # data_matrix = {"matrix":table, "range_step":dist_step, "npax_step":1}
    #
    # range_interval = data_matrix["range_step"]
    # capacity_interval = data_matrix["npax_step"]
    #
    # nc,nr = data_matrix["matrix"].shape
    # range_list = [int(dist_step+range_interval*j) for j in range(nr+1)]
    # capa_list = [int(1+capacity_interval*j) for j in range(nc+1)]
    #
    # fig, ax = plt.subplots(figsize=(14, 7))
    #
    # im = ax.pcolormesh(data_matrix["matrix"],
    #                    edgecolors='b',
    #                    linewidth=0.01,
    #                    cmap="rainbow",
    #                    norm=colors.LogNorm(vmin=data_matrix["matrix"].min()+0.1, vmax=data_matrix["matrix"].max()))
    # ax = plt.gca()
    # ax.set_aspect('equal')
    # ax.set_xlabel('Ranges (km)',
    #               fontsize=16)
    # ax.set_ylabel('Seat capacity',
    #               fontsize=16)
    # ax.xaxis.set_ticks(range(len(range_list)))
    # ax.xaxis.set_ticklabels(range_list,
    #                         fontsize=8,
    #                         rotation = 'vertical')
    # ax.yaxis.set_ticks(range(len(capa_list)))
    # ax.yaxis.set_ticklabels(capa_list,
    #                         fontsize=8)
    # plt.title('Number of flights per seat capacity and range',
    #           fontsize=16)
    # cbar = fig.colorbar(im, ax=ax,
    #                     orientation='horizontal',
    #                     aspect=40.)
    # plt.savefig("heat_map", dpi=500, bbox_inches='tight')
    # # plt.show()








        # ac_dict = ddm.get_best_range(distance, unit.m_km(20), "pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system)
        # ddm.print_design(ac_dict, content="criteria")
        #
        # ac_dict = ddm.get_best_range(distance, unit.m_km(20), "pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system)
        # ddm.print_design(ac_dict, content="criteria")



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
    #     ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict)
    #
    #     ac_dict = ddm.get_best_range(distance, unit.m_km(50), "pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")
    #
    #     ac_dict = ddm.get_best_range(distance, unit.m_km(50), "pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")


    # # Airplane design analysis
    # #-------------------------------------------------------------------------------------------------------------------
    # npax = 150
    # distance = unit.convert_from("NM", 3000)
    # cruise_speed = 0.78
    #
    # airplane_type = "narrow_body"
    # initial_power_system = {"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.kerosene}
    #
    # altitude_data = ddm.cruise_altp(airplane_type)
    # reserve_data = ddm.reserve_data(airplane_type)
    #
    # tpws = [{"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.kerosene},
    #         {"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.gh2},
    #         {"thruster":ddm.fan, "engine_type":ddm.turbofan, "energy_source":ddm.lh2}]
    #
    # for target_power_system in tpws:
    #     ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict)
    #
    #     ac_dict = ddm.get_best_range(distance, unit.m_km(50), "pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")
    #
    #     ac_dict = ddm.get_best_range(distance, unit.m_km(50), "pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict, content="criteria")


