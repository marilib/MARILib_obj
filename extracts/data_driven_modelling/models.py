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

        self.kerosene_heat = unit.convert_from("MJ",43)  # MJ/kg
        self.hydrogen_heat = unit.convert_from("MJ",121) # MJ/kg

        self.eta_prop = 0.80
        self.eta_fan = 0.82
        self.eta_motor = 0.95       # MAGNIX
        self.eta_fuel_cell = 0.50   # (Horizon Fuel Cell)

        self.turbofan_pw_density = unit.W_kW(7)  # Wh/kg
        self.turboprop_pw_density = unit.W_kW(5) # Wh/kg
        self.piston_eng_pw_density = unit.W_kW(1) # Wh/kg
        self.elec_motor_pw_density = unit.W_kW(4.5) # Wh/kg   MAGNIX

        self.battery_enrg_density = unit.J_Wh(400)  # Wh/kg
        self.battery_vol_density = 2500.            # kg/m3

        self.fuel_cell_gravimetric_index = unit.convert_from("kW/kg", 5)    # kW/kg, Horizon Fuel Cell
        self.cooling_gravimetric_index = unit.convert_from("kW/kg", 5)      # kW/kg, Dissipated thermal power per cooloing system mass

        self.gh2_tank_gravimetric_index = 0.05              # kgH2 / (kg_H2 + kg_Tank)
        self.gh2_tank_volumetric_index = 25                 # kgH2 / (m3_H2 + m3_Tank)

        self.lh2_tank_gravimetric_index = 0.10              # kgH2 / (kg_H2 + kg_Tank)
        self.lh2_tank_volumetric_index = 45                 # kgH2 / (m3_H2 + m3_Tank)

        self.piston = "piston"
        self.turbofan = "turbofan"
        self.turboprop = "turboprop"

        self.fan_battery = "fan_battery"
        self.fan_fc_gh2 = "fan_fc_gh2"
        self.fan_fc_lh2 = "fan_fc_lh2"

        self.prop_battery = "prop_battery"
        self.prop_fc_gh2 = "prop_fc_gh2"
        self.prop_fc_lh2 = "prop_fc_lh2"

        self.general = "general"
        self.commuter = "commuter"
        self.business = "business"
        self.narrow_body = "narrow_body"
        self.wide_body = "wide_body"

        self.mpax_allowance_low = [90, unit.m_km(1000)]
        self.mpax_allowance_med = [120, unit.m_km(8000)]
        self.mpax_allowance_high = [150, unit.m_km(np.inf)]

        self.lod_low = [15, 1000]
        self.lod_high = [20, 200000]

        self.psfc_low = [unit.convert_from("lb/shp/h",0.6), unit.convert_from("kW",50)]
        self.psfc_high = [unit.convert_from("lb/shp/h",0.4), unit.convert_from("kW",1000)]

        self.tsfc_low = [unit.convert_from("kg/daN/h",0.60), unit.convert_from("MW",1)]
        self.tsfc_high = [unit.convert_from("kg/daN/h",0.54), unit.convert_from("MW",10)]


    def get_pax_allowance(self,distance):
        mpax_min, dist_min = self.mpax_allowance_low
        mpax_med, dist_med = self.mpax_allowance_med
        mpax_max, dist_max = self.mpax_allowance_high
        if distance<dist_min:
            return mpax_min
        elif distance<dist_med:
            return mpax_med
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
        # a, b, c = [7.56013195e-05, 2.03471207e+02, 0. ]
        a, b, c = [5.84451131e-05, 1.67987893e+02, 0.]
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
        if engine_type==self.piston:
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # piston engine

        elif engine_type==self.turbofan:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_tsfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(tas*lod)))             # turbofan
        elif engine_type in [self.fan_fc_gh2, self.fan_fc_lh2]:
            eff = self.eta_fan * self.eta_motor * self.eta_fuel_cell
            fhv = self.hydrogen_heat
            fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))             # electrofan + fuel cell
        elif engine_type==self.fan_battery:
            fuel = start_mass*g*distance / (self.eta_fan*self.eta_motor*lod)       # electrofan + battery

        elif engine_type==self.turboprop:
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # turboprop
        elif engine_type in [self.prop_fc_gh2, self.prop_fc_lh2]:
            eff = self.eta_prop * self.eta_motor * self.eta_fuel_cell
            fhv = self.hydrogen_heat
            fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))             # electroprop + fuel cell
        elif engine_type==self.prop_battery:
            fuel = start_mass*g*distance / (self.eta_prop*self.eta_motor*lod)      # electroprop + battery

        else:
            raise Exception("engine_type is unknown : "+engine_type)
        return fuel * 1.05  # WARNING: correction to take account of climb phases


    def holding_fuel(self,start_mass,time,altp,speed,speed_type,mtow,max_power,engine_type):
        """Compute the fuel for a given holding time
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.get_lod(mtow)
        if engine_type==self.piston:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # piston

        elif engine_type==self.turbofan:
            sfc = self.get_tsfc(max_power)
            fuel = start_mass*(1 - np.exp(-g*sfc*time/lod))                         # turbofan
        elif engine_type in [self.fan_fc_gh2, self.fan_fc_lh2]:
            tas = self.get_tas(tamb,speed,speed_type)
            eff = self.eta_fan * self.eta_motor * self.eta_fuel_cell
            fhv = self.hydrogen_heat
            fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))             # electrofan + fuel cell
        elif engine_type==self.fan_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            fuel = start_mass*g*tas*time / (self.eta_fan*self.eta_motor*lod)        # electrofan + battery

        elif engine_type==self.turboprop:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_psfc(max_power)
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # turboprop
        elif engine_type in [self.prop_fc_gh2, self.prop_fc_lh2]:
            tas = self.get_tas(tamb,speed,speed_type)
            eff = self.eta_prop * self.eta_motor * self.eta_fuel_cell
            fhv = self.hydrogen_heat
            fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))            # electroprop + fuel cell
        elif engine_type==self.prop_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            fuel = start_mass*g*tas*time / (self.eta_prop*self.eta_motor*lod)      # electroprop + battery

        else:
            raise Exception("engine_type is unknown : "+engine_type)
        return fuel


    def total_fuel(self,tow,range,cruise_speed,speed_type,mtow,max_power,engine_type,altitude_data,reserve_data):
        """Compute the total fuel required for a mission
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        cruise_altp = altitude_data["mission"]
        mission_fuel = self.leg_fuel(tow,range,cruise_altp,cruise_speed,speed_type,mtow,max_power,engine_type)
        if engine_type in [self.prop_battery, self.fan_battery]:
            ldw = tow
        else:
            ldw = tow - mission_fuel

        reserve_fuel = 0.
        if reserve_data["fuel_factor"]>0:
            reserve_fuel += reserve_data["fuel_factor"]*mission_fuel
        if reserve_data["diversion_leg"]>0:
            leg = reserve_data["diversion_leg"]
            diversion_altp = altitude_data["diversion"]
            reserve_fuel += self.leg_fuel(ldw,leg,diversion_altp,cruise_speed,speed_type,mtow,max_power,engine_type)
        if reserve_data["holding_time"]>0:
            time = reserve_data["holding_time"]
            holding_altp = altitude_data["holding"]
            speed = 0.5 * cruise_speed
            reserve_fuel += self.holding_fuel(ldw,time,holding_altp,speed,speed_type,mtow,max_power,engine_type)

        return mission_fuel+reserve_fuel


    def owe_performance(self, npax, mtow, range, cruise_speed, max_power, engine_type, altitude_data, reserve_data):
        """Compute OWE from the point of view of mission
        delta_system_mass contains the battery weight or tank weight for GH2 or LH2 storage
        """
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"

        payload = npax * self.get_pax_allowance(range)

        medium = self.total_fuel(mtow, range, cruise_speed, speed_type, mtow, max_power, engine_type, altitude_data, reserve_data)

        if engine_type in [self.prop_battery, self.fan_battery]:
            total_fuel = 0.
            energy_storage = medium/self.battery_enrg_density
        elif engine_type in [self.prop_fc_gh2, self.fan_fc_gh2]:
            total_fuel = medium
            energy_storage = total_fuel * (1./self.gh2_tank_gravimetric_index - 1.)
        elif engine_type in [self.prop_fc_lh2, self.fan_fc_lh2]:
            total_fuel = medium
            energy_storage = total_fuel * (1./self.lh2_tank_gravimetric_index - 1.)
        else:
            total_fuel = medium
            energy_storage = 0.

        owe = mtow - payload - total_fuel

        return {"owe":owe, "energy_storage":energy_storage, "total_fuel":total_fuel, "payload":payload}


    def owe_structure(self, mtow, energy_storage, initial_engine_type=None, target_engine_type=None):
        power = self.ref_power(mtow)
        owe = self.ref_owe(mtow)
        delta_engine_mass = 0.
        delta_system_mass = 0.
        if initial_engine_type is not None:
            if target_engine_type is not None:

                # remove initial engine mass
                if initial_engine_type==self.piston:
                    delta_engine_mass -= power / self.piston_eng_pw_density
                elif initial_engine_type==self.turboprop:
                    delta_engine_mass -= power / self.turboprop_pw_density
                elif initial_engine_type==self.turbofan:
                    delta_engine_mass -= power / self.turbofan_pw_density

                # Add new engine mass
                if target_engine_type==self.piston:
                    delta_engine_mass += power / self.piston_eng_pw_density
                elif target_engine_type==self.turboprop:
                    delta_engine_mass += power / self.turboprop_pw_density
                elif target_engine_type in [self.prop_fc_gh2, self.prop_fc_lh2]:
                    delta_engine_mass += power / self.elec_motor_pw_density
                    delta_system_mass += power / self.eta_motor / self.fuel_cell_gravimetric_index
                    eff = self.eta_motor*self.eta_fuel_cell
                    delta_system_mass += power * (1-eff)/eff / self.cooling_gravimetric_index  # All power which is not on the shaft have to be dissipated
                elif target_engine_type==self.prop_battery:
                    delta_engine_mass += power / self.elec_motor_pw_density

                elif target_engine_type==self.turbofan:
                    delta_engine_mass += power / self.turbofan_pw_density
                elif target_engine_type in [self.fan_fc_gh2, self.fan_fc_lh2]:
                    delta_engine_mass += power / self.elec_motor_pw_density
                    delta_system_mass += power / self.eta_motor / self.fuel_cell_gravimetric_index
                    eff = self.eta_motor*self.eta_fuel_cell
                    delta_system_mass += power * (1-eff)/eff / self.cooling_gravimetric_index  # All power which is not on the shaft have to be dissipated
                elif target_engine_type==self.fan_battery:
                    delta_engine_mass += power / self.elec_motor_pw_density

        return {"owe":owe+energy_storage+delta_engine_mass+delta_system_mass, "delta_engine_mass":delta_engine_mass, "delta_system_mass":delta_system_mass}


    def mass_mission_adapt(self, npax, distance, cruise_speed, altitude_data, reserve_data, engine_type, target_engine_type=None):

        if target_engine_type is None: target_engine_type = engine_type

        def fct(mtow):
            max_power = self.ref_power(mtow)
            dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, target_engine_type,altitude_data, reserve_data)
            dict_s = self.owe_structure(mtow, dict_p["energy_storage"], initial_engine_type=engine_type, target_engine_type=target_engine_type)
            return (dict_p["owe"]-dict_s["owe"])/dict_s["owe"]

        mtow_ini = (-8.57e-15*npax*distance + 1.09e-04)*npax*distance
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        mtow = output_dict[0][0]
        max_power = self.ref_power(mtow)
        dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, target_engine_type, altitude_data, reserve_data)
        dict_s = self.owe_structure(mtow, dict_p["energy_storage"], initial_engine_type=engine_type, target_engine_type=target_engine_type)

        return {"mtow":mtow, "owe":dict_s["owe"], "payload":dict_p["payload"], "delta_engine_mass":dict_s["delta_engine_mass"],
                "energy_management_mass":dict_p["energy_storage"]+dict_s["delta_system_mass"], "total_fuel":dict_p["total_fuel"]}

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
    for i in df.index:
        npax = float(df['n_pax'][i])
        mtow = float(df['MTOW'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        max_power = float(df['max_power'][i])
        airplane_type = df['airplane_type'][i]
        engine_type = df['engine_type'][i]
        owe_ref = float(df['OWE'][i])

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.owe_performance(npax, mtow, distance, cruise_speed, max_power, engine_type, altitude_data, reserve_data)

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
    for i in df.index:
        npax = float(df['n_pax'][i])
        mtow_ref = float(df['MTOW'][i])
        distance = float(df['nominal_range'][i])
        cruise_speed = float(df['cruise_speed'][i])
        airplane_type = df['airplane_type'][i]
        engine_type = df['engine_type'][i]
        owe_ref = float(df['OWE'][i])

        altitude_data = ddm.cruise_altp(airplane_type)
        reserve_data = ddm.reserve_data(airplane_type)
        dict = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, engine_type)

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
    # dict = do_regression(df, un, abs, ord, coloration, order)

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
    engine_type = "piston"

    target = "piston"

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict1 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, engine_type, target_engine_type=target)
    power1 = ddm.ref_power(dict1["mtow"])

    print("")
    print(" airplane_type = ", airplane_type)
    print("------------------------------------------------")
    print(" Initial engine_type = ", engine_type)
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(power1), " kW")
    print(" owe = ", "%.0f"%dict1["owe"], " kg")
    print(" mtow = ", "%.0f"%dict1["mtow"], " kg")
    print(" payload = ", "%.0f"%dict1["payload"], " kg")
    print(" fuel_total = ", "%.0f"%dict1["total_fuel"], " kg")
    print("")
    print(" K.P/M = ", npax*unit.convert_to("km", distance)/dict1["mtow"])


    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    target = "prop_battery"

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict2 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, engine_type, target_engine_type=target)
    power2 = ddm.ref_power(dict2["mtow"])

    print("")
    print("------------------------------------------------")
    print(" Target engine_type = ", target)
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(power2), " kW")
    print(" owe = ", "%.0f"%dict2["owe"], " kg")
    print(" mtow = ", "%.0f"%dict2["mtow"], " kg")
    print(" payload = ", "%.0f"%dict2["payload"], " kg")
    print(" Engine delta mass = ", "%.0f"%dict2["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict2["energy_management_mass"], " kg")
    print(" Battery energy density : ", unit.convert_to("Wh/kg", ddm.battery_enrg_density), " Wh/kg")
    print("")
    print(" K.P/M = ", npax*unit.convert_to("km", distance)/dict2["mtow"])


    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    target = "prop_fc_gh2"

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict3 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, engine_type, target_engine_type=target)
    power3 = ddm.ref_power(dict3["mtow"])

    print("------------------------------------------------")
    print(" Target engine_type = ", target)
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(power3), " kW")
    print(" owe = ", "%.0f"%dict3["owe"], " kg")
    print(" mtow = ", "%.0f"%dict3["mtow"], " kg")
    print(" payload = ", "%.0f"%dict3["payload"], " kg")
    print(" Engine delta mass = ", "%.0f"%dict3["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict3["energy_management_mass"], " kg")
    print(" Hydrogen mass = ", "%.1f"%dict3["total_fuel"], " kg")
    print(" Overall energy density : ", "%.0f"%unit.convert_to("Wh/kg", (dict3["total_fuel"]*ddm.hydrogen_heat)/(dict3["total_fuel"]+dict3["energy_management_mass"])), " Wh/kg")
    print("")
    print(" K.P/M = ", npax*unit.convert_to("km", distance)/dict3["mtow"])


    npax = 6
    distance = unit.convert_from("km", 500)
    cruise_speed = unit.convert_from("km/h", 180)

    target = "prop_fc_lh2"

    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)
    dict4 = ddm.mass_mission_adapt(npax, distance, cruise_speed, altitude_data, reserve_data, engine_type, target_engine_type=target)
    power4 = ddm.ref_power(dict4["mtow"])

    print("------------------------------------------------")
    print(" Target engine_type = ", target)
    print(" npax = ", npax)
    print(" distance = ", unit.convert_to("km", distance), " km")
    print(" cruise_speed = ", unit.convert_to("km/h", cruise_speed), " km/h")
    print("")
    print(" power = ", "%.0f"%unit.kW_W(power4), " kW")
    print(" owe = ", "%.0f"%dict4["owe"], " kg")
    print(" mtow = ", "%.0f"%dict4["mtow"], " kg")
    print(" payload = ", "%.0f"%dict4["payload"], " kg")
    print(" Engine delta mass = ", "%.0f"%dict4["delta_engine_mass"], " kg")
    print(" Energy system mass = ", "%.0f"%dict4["energy_management_mass"], " kg")
    print(" Hydrogen mass = ", "%.1f"%dict4["total_fuel"], " kg")
    print(" Overall energy density : ", "%.0f"%unit.convert_to("Wh/kg", (dict4["total_fuel"]*ddm.hydrogen_heat)/(dict4["total_fuel"]+dict4["energy_management_mass"])), " Wh/kg")
    print("")
    print(" K.P/M = ", npax*unit.convert_to("km", distance)/dict4["mtow"])







