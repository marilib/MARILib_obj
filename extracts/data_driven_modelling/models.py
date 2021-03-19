#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import least_squares

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

import unit
from physical_data import PhysicalData



class DDM(object):                  # Data Driven Modelling

    def __init__(self, phd):
        self.phd = phd

        self.disa = 0.

        self.eta_prop = 0.82
        self.eta_fan = 0.82
        self.eta_motor = 0.95

        self.turbofan_pw_density = unit.W_kW(7)  # Wh/kg
        self.turboprop_pw_density = unit.W_kW(5) # Wh/kg
        self.piston_eng_pw_density = unit.W_kW(1) # Wh/kg
        self.elec_motor_pw_density = unit.W_kW(4.5) # Wh/kg   (MAGNIX)

        self.battery_enrg_density = unit.J_Wh(200)  # Wh/kg
        self.battery_vol_density = 2500.            # kg/m3

        self.piston = "piston"
        self.turbofan = "turbofan"
        self.turboprop = "turboprop"
        self.fan_battery = "fan_battery"
        self.prop_battery = "prop_battery"

        self.general = "general"
        self.commuter = "commuter"
        self.business = "business"
        self.narrow_body = "narrow_body"
        self.wide_body = "wide_body"

        self.pax_allowance = {"general":90, "commuter":100, "business":150, "narrow_body":130, "wide_body":150}  # kg

        self.ref_lod = {"general":17, "business":17, "commuter":17, "narrow_body":17, "wide_body":17}

        self.ref_tsfc = {"business":0.17e-5, "commuter":0.16e-5, "narrow_body":0.15e-5, "wide_body":0.15e-5} # kg/N/s

        self.ref_psfc = {"general":0.69e-7, "commuter":0.67e-7, "business":0.67e-7} # kg/w/s

    def get_tas(self,tamb,speed,speed_type):
        if speed_type=="mach":
            vsnd = self.phd.sound_speed(tamb)
            tas = speed * vsnd
            return tas
        elif speed_type=="tas":
            return speed

    def cruise_altp(self,airplane_type):
        """return [cruise altitude, diversion altitude]
        """
        if airplane_type==self.general:
            mz, dz, hz = unit.m_ft(5000), unit.m_ft(5000), unit.m_ft(1500)
        elif airplane_type==self.commuter:
            mz, dz, hz = unit.m_ft(20000), unit.m_ft(10000), unit.m_ft(1500)
        elif airplane_type in [self.business, self.narrow_body]:
            mz, dz, hz = unit.m_ft(35000), unit.m_ft(25000), unit.m_ft(1500)
        elif airplane_type==self.wide_body:
            mz, dz, hz = unit.m_ft(35000), unit.m_ft(25000), unit.m_ft(1500)
        return {"mission":mz, "diversion":dz, "holding":hz}

    def reserve_data(self,airplane_type):
        """return [mission fuel factor, diversion leg, holding time]
        """
        if airplane_type==self.general:
            ff,dl,ht = 0., 0., unit.s_min(30)
        elif airplane_type==self.commuter:
            ff,dl,ht = 0., 0., unit.s_min(30)
        elif airplane_type in [self.business, self.narrow_body]:
            ff,dl,ht = 0.05, unit.m_NM(200), unit.s_min(30)
        elif airplane_type==self.wide_body:
            ff,dl,ht = 0.03, unit.m_NM(200), unit.s_min(30)
        return {"fuel_factor":ff, "diversion_leg":dl, "holding_time":ht}

    def pax_allowance(self, airplane_type):
        return self.pax_allowance[airplane_type]

    def tsfc(self, airplane_type):
        return self.ref_tsfc[airplane_type]

    def psfc(self, airplane_type):
        return self.ref_psfc[airplane_type]

    def lod(self,engine_type,airplane_type):
        return self.ref_lod[airplane_type]

    def ref_power(self, mtow):
        return mtow * 250

    def ref_owe(self, mtow):
        return mtow * 0.5

    def leg_fuel(self,start_mass,distance,altp,speed,speed_type,engine_type,airplane_type):
        """Compute the fuel over a given distance
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.ref_lod[airplane_type]
        if engine_type==self.turbofan:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.ref_tsfc[airplane_type]
            return start_mass*(1-np.exp(-(sfc*g*distance)/(tas*lod)))              # turbofan
        elif engine_type==self.fan_battery:
            return start_mass*g*distance / (self.eta_fan*self.eta_motor*lod)       # fan_battery
        elif engine_type==self.piston:
            sfc = self.ref_psfc[airplane_type]
            return start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.turboprop:
            sfc = self.ref_psfc[airplane_type]
            return start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.prop_battery:
            return start_mass*g*distance / (self.eta_prop*self.eta_motor*lod)      # prop_battery
        else:
            raise Exception("engine_type is unknown : "+engine_type)

    def holding_fuel(self,start_mass,time,altp,speed,speed_type,engine_type,airplane_type):
        """Compute the fuel for a given holding time
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.ref_lod[airplane_type]
        if engine_type==self.turbofan:
            sfc = self.ref_tsfc[airplane_type]
            return start_mass*(1 - np.exp(-g*sfc*time/lod))             # turbofan
        elif engine_type==self.fan_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            return start_mass*g*tas*time / (self.eta_fan*self.eta_motor*lod)       # fan_battery
        elif engine_type==self.piston:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.ref_psfc[airplane_type]
            return start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.turboprop:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.ref_psfc[airplane_type]
            return start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.prop_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            return start_mass*g*tas*time / (self.eta_prop*self.eta_motor*lod)      # prop_battery

    def total_fuel(self,tow,range,cruise_speed,speed_type,engine_type,airplane_type):
        """Compute the total fuel required for a mission
        """
        altitude = self.cruise_altp(airplane_type)
        cruise_altp = altitude["mission"]
        mission_fuel = self.leg_fuel(tow,range,cruise_altp,cruise_speed,speed_type,engine_type,airplane_type)
        ldw = tow - mission_fuel
        data = self.reserve_data(airplane_type)
        reserve_fuel = 0.
        if data["fuel_factor"]>0:
            reserve_fuel += data["fuel_factor"]*mission_fuel
        if data["diversion_leg"]>0:
            leg = data["diversion_leg"]
            diversion_altp = altitude["diversion"]
            reserve_fuel += self.leg_fuel(ldw,leg,diversion_altp,cruise_speed,speed_type,engine_type,airplane_type)
        if data["holding_time"]>0:
            time = data["holding_time"]
            holding_altp = altitude["diversion"]
            speed = 0.5 * cruise_speed
            reserve_fuel += self.holding_fuel(ldw,time,holding_altp,speed,speed_type,engine_type,airplane_type)
        return mission_fuel+reserve_fuel

    def owe_performance(self, npax, mtow, range, cruise_speed, engine_type, airplane_type):
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"
        total_fuel = self.total_fuel(mtow, range, cruise_speed, speed_type, engine_type, airplane_type)
        payload = npax * self.pax_allowance[airplane_type]
        # print(mtow, payload, total_fuel)
        return mtow-payload-total_fuel


    def set_param(self, param):
        self.ref_lod["general"]     = 15*param[0]
        self.ref_lod["business"]    = 17*param[1]
        self.ref_lod["commuter"]    = 17*param[2]
        self.ref_lod["narrow_body"] = 17*param[3]
        self.ref_lod["wide_body"]   = 19*param[4]

        self.ref_tsfc["business"]    = 0.17e-5*param[5]
        self.ref_tsfc["commuter"]    = 0.17e-5*param[6]
        self.ref_tsfc["narrow_body"] = 0.16e-5*param[7]
        self.ref_tsfc["wide_body"]   = 0.15e-5*param[8]

        self.ref_psfc["general"]  = 0.69e-7*param[9]
        self.ref_psfc["business"] = 0.67e-7*param[11]
        self.ref_psfc["commuter"] = 0.67e-7*param[10]


def read_db(file):
    raw_data = pd.read_excel(file)     # Load data base as a Pandas data frame
    un = raw_data.iloc[0:2,0:]                          # Take unit structure only
    df = raw_data.iloc[2:,0:].reset_index(drop=True)    # Remove unit rows and reset index
    for name in df.columns:
        if un[name][0] not in ["string","int"] and name not in ["cruise_speed","max_speed"]:
            df[name] = unit.convert_from(un[name][0], list(df[name]))
    for name in ["cruise_speed","max_speed"]:
        for j in df.index:
            if df[name][j]>0.:
                df[name][j] = unit.convert_from(un[name][0], df[name][j])
    return df,un


def lin_lst_reg(abs, ord, order, df):

    def make_mat(param,order):
        mat = np.array(param**0)
        for j in range(order):
            mat = np.vstack([param**(1+j),mat])
        return mat.T      # Need to transpose the stacked matrix

    param = np.array(list(df[abs]))
    A = make_mat(param, order)
    B = np.array(list(df[ord]))
    (C, res, rnk, s) = np.linalg.lstsq(A, B, rcond=None)

    AC = np.dot(A,C)
    res = np.sqrt(np.sum((AC-B)**2))

    x_reg = np.array(np.linspace(0, max(df[abs]), 20))
    F = make_mat(x_reg, order)
    y_reg = np.dot(F,C)

    return {"coef":C, "res":res, "reg":[x_reg,y_reg]}


def draw_reg(df, un, abs, ord, reg, coloration):
    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title("Regression")

    title = ord + " - " + abs
    fig.suptitle(title, fontsize=12)

    for typ in coloration.keys():
        abs_list = list(df.loc[df['airplane_type']==typ][abs])
        ord_list = list(df.loc[df['airplane_type']==typ][ord])
        plt.scatter(abs_list, ord_list, marker="o", c=coloration[typ], s=10)

    plt.plot(reg[0], reg[1], linewidth=1, color="grey")

    plt.ylabel(ord+' ('+un[ord][0]+')')
    plt.xlabel(abs+' ('+un[abs][0]+')')
    plt.grid(True)
    plt.show()





if __name__ == '__main__':

    phd = PhysicalData()
    ddm = DDM(phd)

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "All_Data.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

    # View data
    #-------------------------------------------------------------------------------------------------------------------
    abs = "MTOW"
    ord = "OWE"

    print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    # ord = "total_power"
    #
    # df[ord] = df['max_power']*df['n_engine']
    # un[ord] = un['max_power']

    lin = False
    lst = True

    if lst:
        # Regression
        #-------------------------------------------------------------------------------------------------------------------
        def fct(param, res="sqr"):
            ddm.set_param(param)

            out = []
            for i in df.index:
                npax = df['n_pax'][i]
                mtow = df['MTOW'][i]
                distance = df['nominal_range'][i]
                cruise_speed = df['cruise_speed'][i]
                airplane_type = df['airplane_type'][i]
                engine_type = df['engine_type'][i]
                owe_ref = df['OWE'][i]
                owe_mod = ddm.owe_performance(npax, mtow, distance, cruise_speed, engine_type, airplane_type)
                if res=="sqr":
                    out.append((owe_ref-owe_mod)**2)
                else:
                    out.append(owe_mod)

            return out

        x0 = [1, 1, 1, 1, 1,    1, 1, 1, 1,    1, 1, 1]

        df['OWE_mod'] = fct(x0, res="owe")
        un['OWE_mod'] = un['OWE']

        coloration = {"general":"yellow", "commuter":"green", "business":"blue", "narrow_body":"orange", "wide_body":"red"}

        draw_reg(df, un, 'OWE', 'OWE_mod', [[0,max(df['OWE'])], [0,max(df['OWE'])]], coloration)


        # out = least_squares(residual, x0)
        #
        # ddm.set_param(out.x)
        #
        # print(out.x)



    if lin:
        # Regression
        #-------------------------------------------------------------------------------------------------------------------
        order = 2       # Select regression order

        dict = lin_lst_reg(abs, ord, order, df)

        # Prints & Draws
        #-------------------------------------------------------------------------------------------------------------------
        print("Coef = ", dict["coef"])
        print("Res = ", dict["res"])

        coloration = {"general":"yellow", "commuter":"green", "business":"blue", "narrow_body":"orange", "wide_body":"red"}

        draw_reg(df, un, abs, ord, dict["reg"], coloration)


