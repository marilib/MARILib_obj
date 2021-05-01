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

from analyse_data import coloration, read_db, draw_reg, draw_hist, do_regression


def compare_owe_base_and_model(df, ddm, coloration):
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

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.owe_performance(npax, mtow, distance, cruise_speed, max_power, power_system, altitude_data, reserve_data)

        owe_ref = float(df['OWE'][i])
        rer.append((dict["owe"]-owe_ref)/owe_ref)   # Store relative error
        owe.append(dict["owe"])

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    draw_reg(df, un, 'OWE', 'OWE_mod', [[0,max(df['OWE'])], [0,max(df['OWE'])]], coloration)
    draw_hist(rer, 'OWE model - OWE base')


def compare_mlw_base_and_model(df, ddm, coloration):
    """Compare OWE from data base with OWE computed through nominal mission simulation
    Results are drawn on graphs
    """
    mlw = []
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

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, power_system)

        mlw_ref = float(df['MLW'][i])
        rer.append((dict["mlw"]-mlw_ref)/mlw_ref)   # Store relative error
        mlw.append(dict["mlw"])

    df['MLW_mod'] = mlw
    un['MLW_mod'] = un['MLW']

    draw_reg(df, un, 'MLW', 'MLW_mod', [[0,max(df['MLW'])], [0,max(df['MLW'])]], coloration)
    draw_hist(rer, 'MLW model - MLW base')


def compare_adaptation(df, ddm, coloration):
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
        dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, power_system)

        mtow.append(dict["mtow"])
        owe.append(dict["owe"])
        rer_owe.append((dict["owe"]-owe_ref)/owe_ref)       # Store relative error on OWE
        rer_mtow.append((dict["mtow"]-mtow_ref)/mtow_ref)   # Store relative error on MTOW

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    df['MTOW_mod'] = mtow
    un['MTOW_mod'] = un['MTOW']

    df['MTOW_err'] = 1. + (df['MTOW_mod'] - df['MTOW'])/df['MTOW']
    un['MTOW_err'] = "no_dim"

    draw_reg(df, un, 'MTOW', 'MTOW_mod', [[0,max(df['MTOW'])], [0,max(df['MTOW'])]], coloration)
    draw_reg(df, un, 'MTOW', 'MTOW_err', [[0,max(df['MTOW'])], [0,max(df['MTOW'])]], coloration)
    draw_hist(rer_owe, 'OWE model - OWE base')
    draw_hist(rer_mtow, 'MTOW model - MTOW base')


def compare_vapp_base_and_model(df, ddm, coloration):
    """Compare OWE from data base with OWE computed through nominal mission simulation
    Results are drawn on graphs
    """
    app_speed = []
    rer = []
    power_system = ddm.default_power_system
    for i in df.index:
        npax = float(df['n_pax'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        airplane_type = df['airplane_type'][i]
        power_system["engine_type"] = df['engine_type'][i]

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, power_system)

        wing_area = float(df['wing_area'][i])
        vapp_ref = float(df['approach_speed'][i])
        disa = 0
        altp = unit.m_ft(0)
        mlw = float(df['MLW'][i])

        vapp = ddm.get_app_speed(dict, wing_area, disa, altp, mlw)

        rer.append((vapp-vapp_ref)/vapp_ref)   # Store relative error
        app_speed.append(vapp)

    df['app_speed_mod'] = app_speed
    un['app_speed_mod'] = un['approach_speed']

    draw_reg(df, un, 'approach_speed', 'app_speed_mod', [[0,max(df['approach_speed'])], [0,max(df['approach_speed'])]], coloration)
    draw_hist(rer, 'vapp model - vapp base')


def compare_tofl_base_and_model(df, ddm, coloration):
    """Compare OWE from data base with OWE computed through nominal mission simulation
    Results are drawn on graphs
    """
    field_length = []
    rer = []
    power_system = ddm.default_power_system
    for i in df.index:
        npax = float(df['n_pax'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        airplane_type = df['airplane_type'][i]
        power_system["engine_type"] = df['engine_type'][i]

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, power_system)

        wing_area = float(df['wing_area'][i])
        tofl_ref = float(df['tofl'][i])
        disa = 0
        altp = unit.m_ft(0)
        mtow = float(df['MTOW'][i])

        tofl = ddm.get_tofl(dict, wing_area, disa, altp, mtow)

        rer.append((tofl-tofl_ref)/tofl_ref)   # Store relative error
        field_length.append(tofl)

    df['tofl_mod'] = field_length
    un['tofl_mod'] = un['tofl']

    draw_reg(df, un, 'tofl', 'tofl_mod', [[0,max(df['tofl'])], [0,max(df['tofl'])]], coloration)
    draw_hist(rer, 'tofl model - tofl base')


#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis
#
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "All_Data_v4.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)


    # perform regressions
    #-------------------------------------------------------------------------------------------------------------------
    phd = PhysicalData()
    ddm = DDM(phd)

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    un1 = un.copy()


    # Analyse OWE model versus OWE data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_owe_base_and_model(df1, ddm, coloration)


    # Analyse OWE & MTOW model versus OWE & MTOW data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_adaptation(df1, ddm, coloration)


    # Analyse OWE model versus OWE data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_mlw_base_and_model(df1, ddm, coloration)


    # Analyse OWE model versus OWE data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_tofl_base_and_model(df1, ddm, coloration)


    # Analyse OWE model versus OWE data base
    #-------------------------------------------------------------------------------------------------------------------
    compare_vapp_base_and_model(df1, ddm, coloration)

