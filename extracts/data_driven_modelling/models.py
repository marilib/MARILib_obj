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

        self.battery_enrg_density = unit.J_Wh(400)  # Wh/kg
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

        self.mpax_allowance_low = [90, unit.m_km(400)]
        self.mpax_allowance_high = [150, unit.m_km(8000)]

        self.lod_low = [15, 1000]
        self.lod_high = [20, 200000]

        self.psfc_low = [unit.convert_from("lb/shp/h",0.6), unit.convert_from("kW",50)]
        self.psfc_high = [unit.convert_from("lb/shp/h",0.4), unit.convert_from("kW",1000)]

        self.tsfc_low = [unit.convert_from("kg/daN/h",0.60), unit.convert_from("MW",1)]
        self.tsfc_high = [unit.convert_from("kg/daN/h",0.54), unit.convert_from("MW",10)]


    def get_pax_allowance(self,distance):
        mpax_min, dist_min = self.mpax_allowance_low
        mpax_max, dist_max = self.mpax_allowance_high
        if distance<dist_min:
            return mpax_min
        elif distance<dist_max:
            return mpax_min + (mpax_max-mpax_min)*(distance-dist_min)/(dist_max-dist_min)
        else:
            return mpax_max

    def get_lod(self,mtow):
        lod_min, mtow_min = self.lod_low
        lod_max, mtow_max = self.lod_high
        if mtow<mtow_min:
            return lod_min
        elif mtow<mtow_max:
            return lod_min + (lod_max-lod_min)*(mtow-mtow_min)/(mtow_max-mtow_min)
        else:
            return lod_max

    def get_psfc(self,max_power):
        psfc_max, pw_min = self.psfc_low
        psfc_min, pw_max = self.psfc_high
        if max_power<pw_min:
            return psfc_max
        elif max_power<pw_max:
            return psfc_max - (psfc_max-psfc_min)*(max_power-pw_min)/(pw_max-pw_min)
        else:
            return psfc_min

    def get_tsfc(self,max_power):
        tsfc_max, pw_min = self.tsfc_low
        tsfc_min, pw_max = self.tsfc_high
        if max_power<pw_min:
            return tsfc_max
        elif max_power<pw_max:
            return tsfc_max - (tsfc_max-tsfc_min)*(max_power-pw_min)/(pw_max-pw_min)
        else:
            return tsfc_min

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

    def ref_power(self, mtow):
        """Required total power for an airplane with a given MTOW
        """
        a, b, c = [7.56013195e-05, 2.03471207e+02, 0. ]
        power = (a*mtow + b)*mtow + c
        return power

    def ref_owe(self, mtow):
        """Averaged OWE for an airplane with a given MTOW
        """
        a, b, c = [-2.52877960e-07, 5.72803778e-01, 0. ]
        owe = (a*mtow + b)*mtow + c
        return owe

    def leg_fuel(self,start_mass,distance,altp,speed,speed_type,mtow,max_power,engine_type):
        """Compute the fuel over a given distance
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.get_lod(mtow)
        if engine_type==self.turbofan:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_tsfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(tas*lod)))             # turbofan
        elif engine_type==self.fan_battery:
            fuel = start_mass*g*distance / (self.eta_fan*self.eta_motor*lod)       # fan_battery
        elif engine_type==self.piston:
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.turboprop:
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.prop_battery:
            fuel = start_mass*g*distance / (self.eta_prop*self.eta_motor*lod)      # prop_battery
        else:
            raise Exception("engine_type is unknown : "+engine_type)
        return fuel

    def holding_fuel(self,start_mass,time,altp,speed,speed_type,mtow,max_power,engine_type):
        """Compute the fuel for a given holding time
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.get_lod(mtow)
        if engine_type==self.turbofan:
            sfc = self.get_tsfc(max_power)
            fuel = start_mass*(1 - np.exp(-g*sfc*time/lod))             # turbofan
        elif engine_type==self.fan_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            fuel = start_mass*g*tas*time / (self.eta_fan*self.eta_motor*lod)       # fan_battery
        elif engine_type==self.piston:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.turboprop:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # turboprop
        elif engine_type==self.prop_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            fuel = start_mass*g*tas*time / (self.eta_prop*self.eta_motor*lod)      # prop_battery
        else:
            raise Exception("engine_type is unknown : "+engine_type)
        return fuel

    def total_fuel(self,tow,range,cruise_speed,speed_type,mtow,max_power,engine_type,airplane_type):
        """Compute the total fuel required for a mission
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        altitude = self.cruise_altp(airplane_type)
        cruise_altp = altitude["mission"]
        mission_fuel = self.leg_fuel(tow,range,cruise_altp,cruise_speed,speed_type,mtow,max_power,engine_type)
        if engine_type in [self.prop_battery, self.fan_battery]:
            ldw = tow
        else:
            ldw = tow - mission_fuel
        data = self.reserve_data(airplane_type)
        reserve_fuel = 0.
        if data["fuel_factor"]>0:
            reserve_fuel += data["fuel_factor"]*mission_fuel
        if data["diversion_leg"]>0:
            leg = data["diversion_leg"]
            diversion_altp = altitude["diversion"]
            reserve_fuel += self.leg_fuel(ldw,leg,diversion_altp,cruise_speed,speed_type,mtow,max_power,engine_type)
        if data["holding_time"]>0:
            time = data["holding_time"]
            holding_altp = altitude["diversion"]
            speed = 0.5 * cruise_speed
            reserve_fuel += self.holding_fuel(ldw,time,holding_altp,speed,speed_type,mtow,max_power,engine_type)
        return mission_fuel+reserve_fuel

    def owe_performance(self, npax, mtow, range, cruise_speed, max_power, engine_type, airplane_type):
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"
        total_fuel = self.total_fuel(mtow, range, cruise_speed, speed_type, mtow, max_power, engine_type, airplane_type)
        payload = npax * self.get_pax_allowance(range)
        if engine_type in [self.prop_battery, self.fan_battery]:
            owe = mtow - payload - total_fuel/self.battery_enrg_density
        else:
            owe = mtow - payload - total_fuel
        return owe

    def owe_structure(self, mtow, initial_engine_type=None, target_engine_type=None):
        power = self.ref_power(mtow)
        owe = self.ref_owe(mtow)
        dm = 0.
        if initial_engine_type is not None:
            if target_engine_type is not None:
                # remove initial engine mass
                if initial_engine_type==self.piston:
                    dm -= power / self.piston_eng_pw_density
                elif initial_engine_type==self.turboprop:
                    dm -= power / self.turboprop_pw_density
                elif initial_engine_type==self.turbofan:
                    dm -= power / self.turbofan_pw_density
                elif initial_engine_type==self.prop_battery:
                    dm -= power / self.elec_motor_pw_density
                elif initial_engine_type==self.fan_battery:
                    dm -= power / self.elec_motor_pw_density
                # Add new engine mass
                if target_engine_type==self.piston:
                    dm += power / self.piston_eng_pw_density
                elif target_engine_type==self.turboprop:
                    dm += power / self.turboprop_pw_density
                elif target_engine_type==self.turbofan:
                    dm += power / self.turbofan_pw_density
                elif target_engine_type==self.prop_battery:
                    dm += power / self.elec_motor_pw_density
                elif target_engine_type==self.fan_battery:
                    dm += power / self.elec_motor_pw_density
        return owe+dm

    def mass_mission_adapt(self, npax, distance, cruise_speed, airplane_type, engine_type, target_engine_type=None):

        if target_engine_type is None: target_engine_type = engine_type

        def fct(mtow):
            max_power = self.ref_power(mtow)
            owe_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, target_engine_type, airplane_type)
            owe_b = self.owe_structure(mtow, initial_engine_type=engine_type, target_engine_type=target_engine_type)
            return (owe_p-owe_b)/owe_b

        mtow_ini = (-8.57e-15*npax*distance + 1.09e-04)*npax*distance
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        mtow = output_dict[0][0]
        owe = self.owe_structure(mtow, initial_engine_type=engine_type, target_engine_type=target_engine_type)
        return owe,mtow

#-----------------------------------------------------------------------------------------------------------------------
#
#  Analysis functions
#
#-----------------------------------------------------------------------------------------------------------------------

def read_db(file):
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


def lin_lst_reg(df, abs, ord, order, through_zero):

    def make_mat(param,order):
        mat_list = []
        if through_zero: n = 1
        else: n = 0
        for j in range(n,order+1,1):
            mat_list.append(param**(j))
        mat = np.vstack(mat_list)
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
    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title("Relative error distribution")
    fig.suptitle(title, fontsize=12)

    plt.hist(rer, bins=10, range=(-1,1))

    plt.ylabel('Count')
    plt.xlabel('Relative Error')
    plt.show()



def do_regression(df, un, abs, ord, coloration, order, through_zero=False):
    # Regression
    #-------------------------------------------------------------------------------------------------------------------
    dict = lin_lst_reg(df, abs, ord, order, through_zero)

    # Prints & Draws
    #-------------------------------------------------------------------------------------------------------------------
    print("Coef = ", dict["coef"])
    print("Res = ", dict["res"])

    draw_reg(df, un, abs, ord, dict["reg"], coloration)

    return dict


def compare_owe_base_and_model(coloration):
    owe = []
    rer = []
    for i in df.index:
        npax = float(df['n_pax'][i])
        mtow = float(df['MTOW'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        max_power = float(df['max_power'][i])
        airplane_type = df['airplane_type'][i]
        engine_type = df['engine_type'][i]
        owe_ref = float(df['OWE'][i])
        owe_mod = ddm.owe_performance(npax, mtow, distance, cruise_speed, max_power, engine_type, airplane_type)
        rer.append((owe_mod-owe_ref)/owe_ref)
        owe.append(owe_mod)

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    draw_reg(df, un, 'OWE', 'OWE_mod', [[0,max(df['OWE'])], [0,max(df['OWE'])]], coloration)

    draw_hist(rer, 'OWE model - OWE base')


def compare_adaptation(coloration, reg):
    owe = []
    mtow = []
    rer_owe = []
    rer_mtow = []
    for i in df.index:
        npax = float(df['n_pax'][i])
        mtow_ref = float(df['MTOW'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        max_power = float(df['max_power'][i])
        airplane_type = df['airplane_type'][i]
        engine_type = df['engine_type'][i]
        owe_ref = float(df['OWE'][i])

        owe_mod,mtow_mod = ddm.mass_mission_adapt(npax, distance, cruise_speed, airplane_type, engine_type)
        mtow.append(mtow_mod)
        owe.append(owe_mod)
        rer_owe.append((owe_mod-owe_ref)/owe_ref)
        rer_mtow.append((mtow_mod-mtow_ref)/mtow_ref)

    df['OWE_mod'] = owe
    un['OWE_mod'] = un['OWE']

    df['MTOW_mod'] = mtow
    un['MTOW_mod'] = un['MTOW']

    draw_reg(df, un, 'MTOW_mod', 'OWE_mod', reg, coloration)

    draw_hist(rer_owe, 'OWE model - OWE base')

    draw_hist(rer_mtow, 'MTOW model - MTOW base')




if __name__ == '__main__':

    phd = PhysicalData()
    ddm = DDM(phd)

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

    order = 2
    through_zero = True
    # dict = do_regression(df, un, abs, ord, coloration, order, through_zero)


    abs = "MTOW"
    ord = "total_power"                           # Name of the new column

    df[ord] = df['max_power']*df['n_engine']      # Add the new column to the dataframe
    un[ord] = un['max_power']                     # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = 2
    through_zero = True
    # dict = do_regression(df, un, abs, ord, coloration, order, through_zero)


    abs = "pax_distance"
    ord = "MTOW"                           # Name of the new column

    df[abs] = df['n_pax']*df['nominal_range']     # Add the new column to the dataframe
    un[abs] = un['nominal_range']                 # Add its unit

    # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))

    order = 2
    through_zero = False
    # dict = do_regression(df, un, abs, ord, coloration, order, through_zero)


    # Analyse owe_performance
    #-------------------------------------------------------------------------------------------------------------------
    # compare_owe_base_and_model(coloration)


    # compare_adaptation(coloration, dict["reg"])


    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)
    airplane_type = "general"
    engine_type = "piston"
    # target = "piston"
    target = "prop_battery"

    # altp = ddm.cruise_altp(airplane_type)["mission"]
    # max_power = ddm.ref_power(1400)
    # fuel1 = ddm.leg_fuel(1400,200000,altp,cruise_speed,'tas',1400,max_power,"prop_battery")
    # fuel2 = ddm.total_fuel(1400,200000,cruise_speed,'tas',1400,max_power,"prop_battery","general")
    # print(fuel2,fuel2/ddm.battery_enrg_density)

    owe,mtow = ddm.mass_mission_adapt(npax, distance, cruise_speed, airplane_type, engine_type, target_engine_type=target)

    print("------------------------")
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print(" airplane_type = ", airplane_type)
    print(" Initial engine_type = ", engine_type)
    print(" Target engine_type = ", target)
    print("")
    print(" owe = ", owe, " kg")
    print(" mtow = ", mtow, " kg")








