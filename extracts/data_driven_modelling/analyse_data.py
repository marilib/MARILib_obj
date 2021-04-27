#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve, least_squares

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

import unit
import utils
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
    xmax = 0
    ymax = 0
    for typ in coloration.keys():
        abs_list = unit.convert_to(un.loc[0,abs],list(df.loc[df['airplane_type']==typ][abs]))
        ord_list = unit.convert_to(un.loc[0,ord],list(df.loc[df['airplane_type']==typ][ord]))
        if len(abs_list)>0:
            xmax = max(xmax, max(abs_list))
            ymax = max(ymax, max(ord_list))
        cloud.append(plt.scatter(abs_list, ord_list, marker="o", c=coloration[typ], s=10, label=typ))
        axes.add_artist(cloud[-1])

    plt.plot(unit.convert_to(un.loc[0,abs],reg[0]), unit.convert_to(un.loc[0,ord],reg[1]), linewidth=1, color="grey")

    axes.legend(handles=cloud, loc="lower right")

    plt.ylabel(ord+' ('+un.loc[0,ord]+')')
    plt.xlabel(abs+' ('+un.loc[0,abs]+')')
    plt.xlim([0, xmax*1.05])
    plt.ylim([0, ymax*1.05])
    plt.grid(True)
    plt.show()


def get_error(df, un, abs, ord, reg, abs_interval):

    # Remove A380-800 row and reset index
    df1 = df[abs_interval[0]<=df[abs]].reset_index(drop=True).copy()
    df1 = df1[df1[abs]<=abs_interval[1]].reset_index(drop=True)

    fct = interp1d(reg[0], reg[1], kind="cubic", fill_value='extrapolate')

    df1['relative_error'] = (fct(df1[abs]) - df1[ord]) / df1[ord]

    print("Mean relative error = ", np.mean(list(df1['relative_error'])))
    print("Variance of relative error = ", np.var(list(df1['relative_error'])))

    draw_hist(list(df1['relative_error']), "error")


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
        dict = ddm.design(npax, distance, cruise_speed, altitude_data, reserve_data, power_system)

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
    path_to_data_base = "All_Data_v2.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

    # perform regressions
    #-------------------------------------------------------------------------------------------------------------------
    coloration = {"general":"gold", "commuter":"green", "business":"blue", "narrow_body":"darkorange", "wide_body":"red"}

    abs = "MTOW"
    ord = "OWE"

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t

    # order = [1]
    order = [2, 1]
    dict_owe = do_regression(df, un, abs, ord, coloration, order)

    # get_error(df, un, abs, ord, dict_owe["reg"], [0, 10000])

    #----------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "total_power"                           # Name of the new column

    df[ord] = df['max_power']*df['n_engine']      # Add the new column to the dataframe
    un[ord] = un['max_power']                     # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1]
    dict = do_regression(df, un, abs, ord, coloration, order)

    # #----------------------------------------------------------------------------------
    # abs = "total_power"
    # ord = "cruise_power"                           # Name of the new column
    #
    # lod_min, mtow_min = [15, 1000]
    # lod_max, mtow_max = [20, 200000]
    # mtow_list = [0.     , mtow_min, mtow_max, np.inf]
    # lod_list =  [lod_min, lod_min , lod_max , lod_max]
    # fct = interp1d(mtow_list, lod_list, kind="linear", fill_value="extrapolate")
    #
    # # df['cruise_speed'] = df['cruise_speed'].mul(296.5)
    #
    # df[ord] = 9.81*df['MTOW']*df['cruise_speed']/fct(df['MTOW'])      # Add the new column to the dataframe
    # un[ord] = un['max_power']                     # Add its unit
    #
    # # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    #
    # draw_reg(df, un, 'cruise_power', 'total_power', [[0,max(df['total_power'])], [0,max(df['total_power'])]], coloration)

    # #----------------------------------------------------------------------------------
    # abs = "MTOW"
    # ord = "wing_loading"                           # Name of the new column
    #
    # df[ord] = df['MTOW']/df['wing_area']     # Add the new column to the dataframe
    # un[ord] = ["kg/m2", "None"]                             # Add its unit
    #
    # # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    #
    # order = [0.8, 0.20, 0.02]
    # # dict = do_regression(df, un, abs, ord, coloration, order)
    #
    # #----------------------------------------------------------------------------------
    # abs = "n_pax"                           # Name of the new column
    # ord = "PoM"
    #
    # df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    # # df1 = df.copy()
    # un1 = un.copy()
    #
    # df1[ord] = df1['n_pax']/df1['MTOW']     # Add the new column to the dataframe
    # un1[ord] = "m/kg"                 # Add its unit
    #
    # # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    #
    # # order = [1.8, 0.8]
    # order = [2, 1, 0]
    # dict = do_regression(df1, un1, abs, ord, coloration, order)

    #----------------------------------------------------------------------------------
    abs = "nominal_range"                           # Name of the new column
    ord = "n_pax"

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    # df1 = df.copy()
    un1 = un.copy()

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = [2, 1, 0]
    dict = do_regression(df1, un1, abs, ord, coloration, order)

    #----------------------------------------------------------------------------------
    abs = "nominal_range"                           # Name of the new column
    ord = "PKoM"

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    # df1 = df.copy()
    un1 = un.copy()

    df1[ord] = df1['n_pax']*df1['nominal_range']/df1['MTOW']     # Add the new column to the dataframe
    un1[ord] = "m/kg"                 # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    # order = [1.8, 0.8]
    order = [2, 0.8]
    dict = do_regression(df1, un1, abs, ord, coloration, order)

    #----------------------------------------------------------------------------------

    npax_list = list(df1['n_pax'])
    dist_list = unit.convert_to("km", list(df1['nominal_range']))
    pkom_list = unit.convert_to("km", list(df1['PKoM']))

    fig = plt.figure("Test")
    c = plt.tricontourf(dist_list, npax_list, pkom_list)
    fig.colorbar(c)

    plt.scatter(dist_list, npax_list)

    plt.show()







    #----------------------------------------------------------------------------------
    abs = "nominal_range"                           # Name of the new column
    ord = "PKoM"

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    # df1 = df.copy()
    un1 = un.copy()

    df1[ord] = df1['n_pax']*df1['nominal_range']/df1['MTOW']     # Add the new column to the dataframe
    un1[ord] = "km/kg"                 # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    # order = [1.8, 0.8]
    order = [2, 0.8]
    dict = do_regression(df1, un1, abs, ord, coloration, order)


    phd = PhysicalData()
    ddm = DDM(phd)

    # Analyse OWE model versus OWE data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_owe_base_and_model(coloration)


    # Analyse OWE & MTOW model versus OWE & MTOW data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_adaptation(coloration, dict_owe["reg"])

