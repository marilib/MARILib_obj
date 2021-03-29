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

    # Reference airplane
    #-------------------------------------------------------------------------------------------------------------------
    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    airplane_type = "general"
    power_system = {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.kerosene}

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict1 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system)

    print("")
    print(" airplane_type = ", airplane_type)
    print("------------------------------------------------")
    print(" Initial engine_type = ", power_system["engine_type"])
    print(" Initial energy source = ", power_system["energy_source"])
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(dict1["shaft_power"]), " kW")
    print(" owe = ", "%.0f"%dict1["owe"], " kg")
    print(" mtow = ", "%.0f"%dict1["mtow"], " kg")
    print(" payload = ", "%.0f"%dict1["payload"], " kg")
    print(" fuel_total = ", "%.0f"%dict1["total_fuel"], " kg")
    print(" energy_total = ", "%.0f"%unit.kWh_J(dict1["total_energy"]), " kWh")
    print("")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict1["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict1["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_1 = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_1["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_1["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_1["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_1 = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_1["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_1["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_1["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")


    #-------------------------------------------------------------------------------------------------------------------
    # npax = 6
    # distance = unit.convert_from("km", 500)
    # cruise_speed = unit.convert_from("km/h", 180)

    target_power_system = {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.gh2}

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict2 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=target_power_system)

    print("")
    print("------------------------------------------------")
    print(" Target engine_type = ", target_power_system["engine_type"])
    print(" Target energy source = ", target_power_system["energy_source"])
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(dict2["shaft_power"]), " kW")
    print(" owe = ", "%.0f"%dict2["owe"], " kg")
    print(" mtow = ", "%.0f"%dict2["mtow"], " kg")
    print(" payload = ", "%.0f"%dict2["payload"], " kg")
    print(" fuel_total = ", "%.0f"%dict2["total_fuel"], " kg")
    print(" energy_total = ", "%.0f"%unit.kWh_J(dict2["total_energy"]), " kWh")
    print("")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict2["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict2["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_2 = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_2["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_2["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_2["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_2 = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_2["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_2["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_2["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    #-------------------------------------------------------------------------------------------------------------------
    # npax = 6
    # distance = unit.convert_from("km", 500)
    # cruise_speed = unit.convert_from("km/h", 180)

    target_power_system = {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.lh2}

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict3 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=target_power_system)

    print("")
    print("------------------------------------------------")
    print(" Target engine_type = ", target_power_system["engine_type"])
    print(" Target energy source = ", target_power_system["energy_source"])
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(dict3["shaft_power"]), " kW")
    print(" owe = ", "%.0f"%dict3["owe"], " kg")
    print(" mtow = ", "%.0f"%dict3["mtow"], " kg")
    print(" payload = ", "%.0f"%dict3["payload"], " kg")
    print(" fuel_total = ", "%.0f"%dict3["total_fuel"], " kg")
    print(" energy_total = ", "%.0f"%unit.kWh_J(dict3["total_energy"]), " kWh")
    print("")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict3["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict3["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_3 = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_3["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_3["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_3["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_3 = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_3["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_3["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_3["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    #-------------------------------------------------------------------------------------------------------------------
    # npax = 6
    # distance = unit.convert_from("km", 500)
    # cruise_speed = unit.convert_from("km/h", 180)

    target_power_system = {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery}

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict4 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=target_power_system)

    print("")
    print("------------------------------------------------")
    print(" Target engine_type = ", target_power_system["engine_type"])
    print(" Target energy source = ", target_power_system["energy_source"])
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(dict4["shaft_power"]), " kW")
    print(" owe = ", "%.0f"%dict4["owe"], " kg")
    print(" mtow = ", "%.0f"%dict4["mtow"], " kg")
    print(" payload = ", "%.0f"%dict4["payload"], " kg")
    print(" energy_total = ", "%.0f"%unit.kWh_J(dict4["total_energy"]), " kWh")
    print(" Engine delta mass = ", "%.0f"%dict4["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict4["energy_management_mass"], " kg")
    print(" Battery energy density : ", unit.convert_to("Wh/kg", ddm.battery_enrg_density), " Wh/kg")
    print("")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict4["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict4["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_4 = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_4["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_4["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_4["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_4 = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_4["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_4["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_4["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    #-------------------------------------------------------------------------------------------------------------------
    # npax = 6
    # distance = unit.convert_from("km", 500)
    # cruise_speed = unit.convert_from("km/h", 180)

    target_power_system["energy_source"] = ddm.gh2

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict5 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=target_power_system)

    print("------------------------------------------------")
    print(" Target engine_type = ", target_power_system["engine_type"])
    print(" Target energy source = ", target_power_system["energy_source"])
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(dict5["shaft_power"]), " kW")
    print(" owe = ", "%.0f"%dict5["owe"], " kg")
    print(" mtow = ", "%.0f"%dict5["mtow"], " kg")
    print(" payload = ", "%.0f"%dict5["payload"], " kg")
    print(" energy_total = ", "%.0f"%unit.kWh_J(dict5["total_energy"]), " kWh")
    print(" Engine delta mass = ", "%.0f"%dict5["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict5["energy_management_mass"], " kg")
    print(" Hydrogen mass = ", "%.1f"%dict5["total_fuel"], " kg")
    print(" Overall energy density : ", "%.0f"%unit.convert_to("Wh/kg", (dict5["total_fuel"]*ddm.hydrogen_heat)/(dict5["total_fuel"]+dict5["energy_management_mass"])), " Wh/kg")
    print("")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict5["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict5["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_5 = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_5["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_5["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_5["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_5 = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_5["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_5["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_5["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    #-------------------------------------------------------------------------------------------------------------------
    # npax = 6
    # distance = unit.convert_from("km", 500)
    # cruise_speed = unit.convert_from("km/h", 180)

    target_power_system["energy_source"] = ddm.lh2

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict6 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=target_power_system)

    print("------------------------------------------------")
    print(" Target engine_type = ", target_power_system["engine_type"])
    print(" Target energy source = ", target_power_system["energy_source"])
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(dict6["shaft_power"]), " kW")
    print(" owe = ", "%.0f"%dict6["owe"], " kg")
    print(" mtow = ", "%.0f"%dict6["mtow"], " kg")
    print(" payload = ", "%.0f"%dict6["payload"], " kg")
    print(" energy_total = ", "%.0f"%unit.kWh_J(dict6["total_energy"]), " kWh")
    print(" Engine delta mass = ", "%.0f"%dict6["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict6["energy_management_mass"], " kg")
    print(" Hydrogen mass = ", "%.1f"%dict6["total_fuel"], " kg")
    print(" Overall energy density : ", "%.0f"%unit.convert_to("Wh/kg", (dict6["total_fuel"]*ddm.hydrogen_heat)/(dict6["total_fuel"]+dict6["energy_management_mass"])), " Wh/kg")
    print("")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict6["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict6["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_6 = ddm.get_best_range("pk_o_mass", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_6["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_6["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_6["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

    dict_6 = ddm.get_best_range("pk_o_enrg", npax, cruise_speed, altitude_data, reserve_data, power_system, target_power_system)
    print("")
    print(" distance = ", "%.2f"%unit.convert_to("km", dict_6["range"]), " km")
    print(" Efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict_6["pk_o_mass"]), " pax.km/kg")
    print(" Efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict_6["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")




