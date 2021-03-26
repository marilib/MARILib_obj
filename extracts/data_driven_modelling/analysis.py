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
#  Analysis functions
#
#-----------------------------------------------------------------------------------------------------------------------

def read_db(file):
    """Read data base and convert to standard units
    WARNING: special treatment for cruise_speed and max_speed which can be Mach number
    """
    raw_data = pd.read_excel(file)     # Load data base as a Pandas data frame
    un = raw_data.iloc[0:2,0:]                          # Take unit structure only
    df = raw_data.iloc[2:,0:].reset_index(drop=True)    # Remove unit rows and reset index

    for name in df.columns:
        if un.loc[0,name] not in ["string","int"] and name not in ["cruise_speed","max_speed"]:
            df[name] = unit.convert_from(un.loc[0,name], list(df[name]))
    for name in ["cruise_speed","max_speed"]:
        for j in df.index:
            if df.loc[j,name]>1.:
                df.loc[j,name] = float(unit.convert_from(un.loc[0,name], df.loc[j,name]))
    return df,un


def lin_lst_reg(df, abs, ord, order):
    """Linear least square regression of "ord" versus "abs" with given order
    order is the list of exponents to apply
    """
    def make_mat(param,order):
        mat_list = []
        for j in order:
            mat_list.append(param**j)
        mat = np.vstack(mat_list)
        return mat.T      # Need to transpose the stacked matrix

    param = np.array(list(df[abs]))
    A = make_mat(param, order)
    B = np.array(list(df[ord]))
    (C, res, rnk, s) = np.linalg.lstsq(A, B, rcond=None)

    AC = np.dot(A,C)
    res = np.sqrt(np.sum((AC-B)**2))

    x_reg = np.array(np.linspace(0, max(df[abs]), 400))
    F = make_mat(x_reg, order)
    y_reg = np.dot(F,C)

    return {"coef":C, "res":res, "reg":[x_reg,y_reg]}


def draw_reg(df, un, abs, ord, reg, coloration):
    """Draw the cloud of point in perspective of the regression given into "reg" as [abs_list, ord_list]
    Coloration of each airplane type is given into "coloration"
    """
    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title("Regression")

    title = ord + " - " + abs
    fig.suptitle(title, fontsize=12)

    cloud = []
    for typ in coloration.keys():
        abs_list = unit.convert_to(un.loc[0,abs],list(df.loc[df['airplane_type']==typ][abs]))
        ord_list = unit.convert_to(un.loc[0,ord],list(df.loc[df['airplane_type']==typ][ord]))
        cloud.append(plt.scatter(abs_list, ord_list, marker="o", c=coloration[typ], s=10, label=typ))
        axes.add_artist(cloud[-1])

    plt.plot(unit.convert_to(un.loc[0,abs],reg[0]), unit.convert_to(un.loc[0,ord],reg[1]), linewidth=1, color="grey")

    axes.legend(handles=cloud, loc="lower right")

    plt.ylabel(ord+' ('+un.loc[0,ord]+')')
    plt.xlabel(abs+' ('+un.loc[0,abs]+')')
    plt.grid(True)
    plt.show()


def draw_hist(rer,title):
    """Draw the histogram of relative errors given into "reg" as [abs_list, ord_list]
    """
    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title("Relative error distribution")
    fig.suptitle(title, fontsize=12)

    plt.hist(rer, bins=10, range=(-1,1))

    plt.ylabel('Count')
    plt.xlabel('Relative Error')
    plt.show()


def do_regression(df, un, abs, ord, coloration, order):
    """Perform regression and draw the corresponding graph
    """
    dict = lin_lst_reg(df, abs, ord, order)
    print("Coef = ", dict["coef"])
    print("Res = ", dict["res"])
    draw_reg(df, un, abs, ord, dict["reg"], coloration)
    return dict


def compare_owe_base_and_model(coloration):
    """Compare OWE from data base with OWE computed through nominal mission simulation
    Results are drawn on graphs
    """
    owe = []
    rer = []
    power_system = ddm.default_power_system
    for i in df.index:
        npax = float(df['n_pax'][i])
        mtow = float(df['MTOW'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        max_power = float(df['max_power'][i])
        airplane_type = df['airplane_type'][i]
        power_system["engine_type"] = df['engine_type'][i]
        owe_ref = float(df['OWE'][i])

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.owe_performance(npax, mtow, distance, cruise_speed, max_power, power_system, altitude_data, reserve_data)

        rer.append((dict["owe"]-owe_ref)/owe_ref)   # Store relative error
        owe.append(dict["owe"])

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    draw_reg(df, un, 'OWE', 'OWE_mod', [[0,max(df['OWE'])], [0,max(df['OWE'])]], coloration)

    draw_hist(rer, 'OWE model - OWE base')


def compare_adaptation(coloration, reg):
    """Compare OWE1 and MTOW1 from data base with OWE2 & MTOW2 computed through the mass-mission adaptation process
    Mass-Mission adaptation process is computed with input coming from the data base that are consistant with OWE1 & MTOW1
    Results are drawn on graphs
    """
    owe = []
    mtow = []
    rer_owe = []
    rer_mtow = []
    power_system = ddm.default_power_system
    for i in df.index:
        npax = float(df['n_pax'][i])
        mtow_ref = float(df['MTOW'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        airplane_type = df['airplane_type'][i]
        power_system["engine_type"] = df['engine_type'][i]
        owe_ref = float(df['OWE'][i])

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system)

        mtow.append(dict["mtow"])
        owe.append(dict["owe"])
        rer_owe.append((dict["owe"]-owe_ref)/owe_ref)       # Store relative error on OWE
        rer_mtow.append((dict["mtow"]-mtow_ref)/mtow_ref)   # Store relative error on MTOW

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    df['MTOW_mod'] = mtow
    un['MTOW_mod'] = un['MTOW']

    draw_reg(df, un, 'MTOW_mod', 'OWE_mod', reg, coloration)

    draw_hist(rer_owe, 'OWE model - OWE base')

    draw_hist(rer_mtow, 'MTOW model - MTOW base')




#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis
#
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "All_Data.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

    # perform regressions
    #-------------------------------------------------------------------------------------------------------------------
    coloration = {"general":"yellow", "commuter":"green", "business":"blue", "narrow_body":"orange", "wide_body":"red"}

    abs = "MTOW"
    ord = "OWE"

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1]
    dict_owe = do_regression(df, un, abs, ord, coloration, order)

    #----------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "total_power"                           # Name of the new column

    df[ord] = df['max_power']*df['n_engine']      # Add the new column to the dataframe
    un[ord] = un['max_power']                     # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1]
    dict = do_regression(df, un, abs, ord, coloration, order)

    #----------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "wing_loading"                           # Name of the new column

    df[ord] = df['MTOW']/df['wing_area']     # Add the new column to the dataframe
    un[ord] = ["kg/m2", "None"]                             # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [0.8, 0.20, 0.02]
    # dict = do_regression(df, un, abs, ord, coloration, order)

    #----------------------------------------------------------------------------------
    abs = "pax_distance"
    ord = "MTOW"                           # Name of the new column

    df[abs] = df['n_pax']*df['nominal_range']     # Add the new column to the dataframe
    un[abs] = un['nominal_range']                 # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1, 0]
    # dict = do_regression(df, un, abs, ord, coloration, order)


    phd = PhysicalData()
    ddm = DDM(phd)

    # Analyse OWE model versus OWE data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_owe_base_and_model(coloration)


    # Analyse OWE & MTOW model versus OWE & MTOW data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_adaptation(coloration, dict_owe["reg"])


    # Small experiment with battery airplane
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
    print("")
    print(" Efficiency factor, K.P/M = ", "%.2f"%(npax*unit.convert_to("km", distance)/dict1["mtow"]))

    #-------------------------------------------------------------------------------------------------------------------
    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

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
    print(" Engine delta mass = ", "%.0f"%dict2["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict2["energy_management_mass"], " kg")
    print(" Battery energy density : ", unit.convert_to("Wh/kg", ddm.battery_enrg_density), " Wh/kg")
    print("")
    print(" Efficiency factor, K.P/M = ", "%.2f"%(npax*unit.convert_to("km", distance)/dict2["mtow"]))

    #-------------------------------------------------------------------------------------------------------------------
    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    target_power_system = {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.lh2}

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
    print(" Engine delta mass = ", "%.0f"%dict2["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict2["energy_management_mass"], " kg")
    print(" Battery energy density : ", unit.convert_to("Wh/kg", ddm.battery_enrg_density), " Wh/kg")
    print("")
    print(" Efficiency factor, K.P/M = ", "%.2f"%(npax*unit.convert_to("km", distance)/dict2["mtow"]))

    #-------------------------------------------------------------------------------------------------------------------
    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    target_power_system = {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery}

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
    print(" Engine delta mass = ", "%.0f"%dict2["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict2["energy_management_mass"], " kg")
    print(" Battery energy density : ", unit.convert_to("Wh/kg", ddm.battery_enrg_density), " Wh/kg")
    print("")
    print(" Efficiency factor, K.P/M = ", "%.2f"%(npax*unit.convert_to("km", distance)/dict2["mtow"]))

    #-------------------------------------------------------------------------------------------------------------------
    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    target_power_system["energy_source"] = ddm.gh2

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict3 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=target_power_system)

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
    print(" Engine delta mass = ", "%.0f"%dict3["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict3["energy_management_mass"], " kg")
    print(" Hydrogen mass = ", "%.1f"%dict3["total_fuel"], " kg")
    print(" Overall energy density : ", "%.0f"%unit.convert_to("Wh/kg", (dict3["total_fuel"]*ddm.hydrogen_heat)/(dict3["total_fuel"]+dict3["energy_management_mass"])), " Wh/kg")
    print("")
    print(" Efficiency factor, K.P/M = ", "%.2f"%(npax*unit.convert_to("km", distance)/dict3["mtow"]))

    #-------------------------------------------------------------------------------------------------------------------
    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    target_power_system["energy_source"] = ddm.lh2

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict4 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=target_power_system)

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
    print(" Engine delta mass = ", "%.0f"%dict4["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict4["energy_management_mass"], " kg")
    print(" Hydrogen mass = ", "%.1f"%dict4["total_fuel"], " kg")
    print(" Overall energy density : ", "%.0f"%unit.convert_to("Wh/kg", (dict4["total_fuel"]*ddm.hydrogen_heat)/(dict4["total_fuel"]+dict4["energy_management_mass"])), " Wh/kg")
    print("")
    print(" Efficiency factor, K.P/M = ", "%.2f"%(npax*unit.convert_to("km", distance)/dict4["mtow"]))




