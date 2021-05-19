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
import utils
from physical_data import PhysicalData



#-----------------------------------------------------------------------------------------------------------------------
#
#  Main class
#
#-----------------------------------------------------------------------------------------------------------------------


class DDM(object):                  # Data Driven Modelling

    def __init__(self, phd):
        self.phd = phd

        self.disa = 0.
        self.max_payload_factor = 1.20          # max_payload = nominal_paylod * max_payload_factor
        self.max_fuel_payload_factor = 0.60     # max_fuel_payload = nominal_payload * max_fuel_payload_factor
        self.mlw_factor = 1.07                  # MLW = MZFW * mlw_factor

        self.kerosene_heat = unit.convert_from("MJ",43)  # MJ/kg
        self.hydrogen_heat = unit.convert_from("MJ",121) # MJ/kg

        self.prop_eff = 0.80
        self.fan_eff = 0.82
        self.motor_eff = 0.95       # MAGNIX
        self.fuel_cell_eff = 0.50   # (Horizon Fuel Cell)

        self.turbofan_pw_density = unit.W_kW(7)  # W/kg
        self.turboprop_pw_density = unit.W_kW(5) # W/kg
        self.piston_eng_pw_density = unit.W_kW(1) # W/kg
        self.elec_motor_pw_density = unit.W_kW(4.5) # W/kg   MAGNIX
        self.power_elec_pw_density = unit.W_kW(10) # W/kg
        self.propeller_pw_density = unit.W_kW(9) # W/kg     ATR
        self.fan_nacelle_pw_density = unit.W_kW(10) # W/kg

        self.battery_enrg_density = unit.J_Wh(400)  # Wh/kg
        self.battery_vol_density = 2500.            # kg/m3

        self.fuel_cell_gravimetric_index = unit.convert_from("kW/kg", 5)    # kW/kg, Horizon Fuel Cell
        self.fc_system_gravimetric_index = unit.convert_from("kW/kg", 2)    # kW/kg
        self.cooling_gravimetric_index = unit.convert_from("kW/kg", 5)      # kW/kg, Dissipated thermal power per cooloing system mass

        self.gh2_tank_gravimetric_index = 0.05              # kgH2 / (kg_H2 + kg_Tank)
        self.gh2_tank_volumetric_index = 25                 # kgH2 / (m3_H2 + m3_Tank)

        self.lh2_tank_gravimetric_index = 0.10              # kgH2 / (kg_H2 + kg_Tank)
        self.lh2_tank_volumetric_index = 45                 # kgH2 / (m3_H2 + m3_Tank)

        self.propeller = "propeller"    # thruster item
        self.fan = "fan"                # thruster item

        self.piston = "piston"          # engine_type item
        self.turbofan = "turbofan"      # engine_type item
        self.turboprop = "turboprop"    # engine_type item
        self.emotor = "emotor"          # engine_type item

        self.kerosene = "kerosene"      # energy_source item
        self.gh2 = "gh2"                # energy_source item
        self.lh2 = "lh2"                # energy_source item
        self.battery = "battery"        # energy_source item

        self.default_power_system = {"thruster":None, "engine_type":None, "energy_source":self.kerosene}

        self.general = "general"
        self.commuter = "commuter"
        self.business = "business"
        self.narrow_body = "narrow_body"
        self.wide_body = "wide_body"

        self.mpax_allowance_low = [90, unit.m_km(1000)]
        self.mpax_allowance_med = [110, unit.m_km(8000)]
        self.mpax_allowance_high = [130, unit.m_km(np.inf)]

        self.structure_mass_factor_low = [0.40, unit.m_km(4000)]      # [fac, PK]
        self.structure_mass_factor_high = [1.00, unit.m_km(16000)]     # [fac, PK]

        self.engine_mass_factor_low = [1.00, unit.W_MW(10)]      # [fac, max_power]
        self.engine_mass_factor_high = [0.85, unit.W_MW(20)]     # [fac, max_power]

        self.cl_max_to = 2.0
        self.kvs1g_to = 1.13
        self.tuner_to = [12., 300]

        self.cl_max_ld = 2.6
        self.kvs1g_ld = 1.23
        self.tuner_app = [1.15, -6]

        self.lod_low = [14, 1000]       # [lod, mtow]
        self.lod_high = [20, 125000]    # [lod, mtow]

        self.psfc_low = [unit.convert_from("lb/shp/h",0.65), unit.convert_from("kW",50)]     # here power is shaft power
        self.psfc_high = [unit.convert_from("lb/shp/h",0.50), unit.convert_from("kW",500)]

        self.tsfc_low = [unit.convert_from("kg/daN/h",0.65), unit.convert_from("MW",5)]     # Here, power is equivalent shaft power developped during take off as defined in the data base
        self.tsfc_high = [unit.convert_from("kg/daN/h",0.52), unit.convert_from("MW",50)]

        self.fc_eff_low = [0.45, 1.0]     # Fuel cell efficiency at system level for design power (max power)
        self.fc_eff_high = [0.55, 0.1]    # Fuel cell efficiency at system level for low power (power close to zero)

        self.delta_ref_power = 0.       # Reference power tuning
        self.wing_area = None

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


    def get_structure_mass_factor(self,nominal_range):
        fac_min, dist_min = self.structure_mass_factor_low
        fac_max, dist_max = self.structure_mass_factor_high
        dist_list = [0.      , dist_min  , dist_max  , np.inf]
        fac_list = [fac_min, fac_min, fac_max, fac_max]
        fac = utils.lin_interp_1d(nominal_range, dist_list, fac_list)
        return fac


    def get_engine_mass_factor(self,max_power):
        fac_min, pw_min = self.engine_mass_factor_low
        fac_max, pw_max = self.engine_mass_factor_high
        mxpw_list = [0.      , pw_min  , pw_max  , np.inf]
        fac_list = [fac_min, fac_min, fac_max, fac_max]
        fac = utils.lin_interp_1d(max_power, mxpw_list, fac_list)
        return fac


    def get_lod(self,mtow):
        lod_min, mtow_min = self.lod_low
        lod_max, mtow_max = self.lod_high
        mtow_list = [0.     , mtow_min, mtow_max, np.inf]
        lod_list =  [lod_min, lod_min , lod_max , lod_max]
        lod = utils.lin_interp_1d(mtow, mtow_list, lod_list)
        return lod


    def get_cl_max_to(self):
        # clm_min, mtow_min = self.cl_max_to_low
        # clm_max, mtow_max = self.cl_max_to_high
        # mtow_list = [0.     , mtow_min, mtow_max, np.inf]
        # clm_list =  [clm_min, clm_min , clm_max , clm_max]
        # clm = utils.lin_interp_1d(mtow, mtow_list, clm_list)
        return self.cl_max_to


    def get_cl_max_ld(self):
        # clm_min, mtow_min = self.cl_max_ld_low
        # clm_max, mtow_max = self.cl_max_ld_high
        # mtow_list = [0.     , mtow_min, mtow_max, np.inf]
        # clm_list =  [clm_min, clm_min , clm_max , clm_max]
        # clm = utils.lin_interp_1d(mtow, mtow_list, clm_list)
        return self.cl_max_ld


    def get_psfc(self,max_power, fuel_type):
        psfc_max, pw_min = self.psfc_low
        psfc_min, pw_max = self.psfc_high

        if fuel_type==self.kerosene:
            fhv = self.kerosene_heat
        elif fuel_type==self.gh2:
            fhv = self.hydrogen_heat
            psfc_min *= self.kerosene_heat/self.hydrogen_heat
            psfc_max *= self.kerosene_heat/self.hydrogen_heat
        elif fuel_type==self.lh2:
            fhv = self.hydrogen_heat
            psfc_min *= self.kerosene_heat/self.hydrogen_heat
            psfc_max *= self.kerosene_heat/self.hydrogen_heat
        else:
            raise Exception("fuel type is unknown")

        mxpw_list = [0.      , pw_min  , pw_max  , np.inf]
        psfc_list = [psfc_max, psfc_max, psfc_min, psfc_min]
        psfc = utils.lin_interp_1d(max_power, mxpw_list, psfc_list)
        return psfc,fhv


    def get_tsfc(self,max_power, fuel_type):
        tsfc_max, pw_min = self.tsfc_low
        tsfc_min, pw_max = self.tsfc_high

        if fuel_type==self.kerosene:
            fhv = self.kerosene_heat
        elif fuel_type==self.gh2:
            fhv = self.hydrogen_heat
            tsfc_min *= self.kerosene_heat/self.hydrogen_heat
            tsfc_max *= self.kerosene_heat/self.hydrogen_heat
        elif fuel_type==self.lh2:
            fhv = self.hydrogen_heat
            tsfc_min *= self.kerosene_heat/self.hydrogen_heat
            tsfc_max *= self.kerosene_heat/self.hydrogen_heat
        else:
            raise Exception("fuel type is unknown")

        mxpw_list = [0.      , pw_min  , pw_max  , np.inf]
        tsfc_list = [tsfc_max, tsfc_max, tsfc_min, tsfc_min]
        tsfc = utils.lin_interp_1d(max_power, mxpw_list, tsfc_list)
        return tsfc,fhv


    def get_fuel_cell_eff(self, pw, pw_max):
        eff_min, kpw_max = self.fc_eff_low
        eff_max, kpw_min = self.fc_eff_high
        pw_list =  [0.     , kpw_min*pw_max, kpw_max*pw_max]
        eff_list = [eff_max, eff_max       , eff_min]
        eff = utils.lin_interp_1d(pw, pw_list, eff_list)
        return eff


    def ref_power(self, mtow):
        """Required total power for an airplane with a given MTOW
        """
        a, b, c = [7.56013195e-05, 2.03471207e+02, 0. ]
        power = (a*mtow + b)*mtow + c + self.delta_ref_power
        return power


    def ref_owe(self, mtow):
        """Averaged OWE for an airplane with a given MTOW
        """
        a, b, c = [-2.52877960e-07, 5.72803778e-01, 0. ]
        owe = (a*mtow + b)*mtow + c
        return owe


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
        else:
            raise Exception("airplane type is unknown")
        return {"airplane_type":airplane_type, "mission":mz, "diversion":dz, "holding":hz}


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
        else:
            raise Exception("airplane type is unknown")
        return {"airplane_type":airplane_type, "fuel_factor":ff, "diversion_leg":dl, "holding_time":ht}


    def get_tas(self,tamb,speed,speed_type):
        if speed_type=="mach":
            vsnd = self.phd.sound_speed(tamb)
            tas = speed * vsnd
            return tas
        elif speed_type=="tas":
            return speed


    def leg_fuel(self,start_mass,distance,altp,speed,speed_type,mtow,max_power,power_system):
        """Compute fuel and or energy over a given distance
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.get_lod(mtow)
        if power_system["engine_type"]==self.piston:
            sfc,fhv = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.prop_eff*lod)))   # piston engine
        elif power_system["engine_type"]==self.turboprop:
            sfc,fhv = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.prop_eff*lod)))   # turboprop
        elif power_system["engine_type"]==self.turbofan:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc,fhv = self.get_tsfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(tas*lod)))             # turbofan
        elif power_system["engine_type"]==self.emotor:
            if power_system["thruster"]==self.propeller:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    eff = self.prop_eff * self.motor_eff * self.fuel_cell_eff
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))             # electroprop + fuel cell
                elif power_system["energy_source"]==self.battery:
                    enrg = start_mass*g*distance / (self.prop_eff*self.motor_eff*lod)      # electroprop + battery
            elif power_system["thruster"]==self.fan:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    eff = self.fan_eff * self.motor_eff * self.fuel_cell_eff
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))             # electrofan + fuel cell
                elif power_system["energy_source"]==self.battery:
                    enrg = start_mass*g*distance / (self.fan_eff*self.motor_eff*lod)       # electrofan + battery
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

        if power_system["energy_source"]==self.battery:
            fuel = 0.
            enrg *= 1.05  # WARNING: correction to take account of climb phases
        else:
            fuel *= 1.05  # WARNING: correction to take account of climb phases
            enrg = fuel*fhv

        return fuel,enrg,lod


    def holding_fuel(self, start_mass, time, altp, speed, speed_type, mtow, max_power, power_system):
        """Compute the fuel for a given holding time
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.get_lod(mtow)
        if power_system["engine_type"]==self.piston:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc,fhv = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.prop_eff*lod)))   # piston
        elif power_system["engine_type"]==self.turboprop:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc,fhv = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.prop_eff*lod)))   # turboprop
        elif power_system["engine_type"]==self.turbofan:
            sfc,fhv = self.get_tsfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1 - np.exp(-g*sfc*time/lod))                         # turbofan
        elif power_system["engine_type"]==self.emotor:
            if power_system["thruster"]==self.propeller:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    tas = self.get_tas(tamb,speed,speed_type)
                    eff = self.prop_eff * self.motor_eff * self.fuel_cell_eff
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))            # electroprop + fuel cell
                elif power_system["energy_source"]==self.battery:
                    tas = self.get_tas(tamb,speed,speed_type)
                    enrg = start_mass*g*tas*time / (self.prop_eff*self.motor_eff*lod)      # electroprop + battery
            elif power_system["thruster"]==self.fan:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    tas = self.get_tas(tamb,speed,speed_type)
                    eff = self.fan_eff * self.motor_eff * self.fuel_cell_eff
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))             # electrofan + fuel cell
                elif power_system["energy_source"]==self.battery:
                    tas = self.get_tas(tamb,speed,speed_type)
                    enrg = start_mass*g*tas*time / (self.fan_eff*self.motor_eff*lod)        # electrofan + battery
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

        if power_system["energy_source"]==self.battery:
            fuel = 0.
        else:
            enrg = fuel*fhv

        return fuel,enrg


    def total_fuel(self,tow,distance,cruise_speed,speed_type,mtow,max_power,power_system,altitude_data,reserve_data):
        """Compute the total fuel required for a mission
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        cruise_altp = altitude_data["mission"]
        mission_fuel,mission_enrg,mission_lod = self.leg_fuel(tow,distance,cruise_altp,cruise_speed,speed_type,mtow,max_power,power_system)
        if power_system["energy_source"]==self.battery:
            ldw = tow
        else:
            ldw = tow - mission_fuel

        reserve_fuel = 0.
        reserve_enrg = 0.
        if reserve_data["fuel_factor"]>0:
            reserve_fuel += reserve_data["fuel_factor"]*mission_fuel
            reserve_enrg += reserve_data["fuel_factor"]*mission_enrg
        if reserve_data["diversion_leg"]>0:
            leg = reserve_data["diversion_leg"]
            diversion_altp = altitude_data["diversion"]
            lf,le,lod = self.leg_fuel(ldw,leg,diversion_altp,cruise_speed,speed_type,mtow,max_power,power_system)
            reserve_fuel += lf
            reserve_enrg += le
        if reserve_data["holding_time"]>0:
            time = reserve_data["holding_time"]
            holding_altp = altitude_data["holding"]
            speed = 0.5 * cruise_speed
            hf,he = self.holding_fuel(ldw,time,holding_altp,speed,speed_type,mtow,max_power,power_system)
            reserve_fuel += hf
            reserve_enrg += he

        return {"tow":tow,
                "distance":distance,
                "total_fuel":mission_fuel+reserve_fuel,
                "mission_fuel":mission_fuel,
                "reserve_fuel":reserve_fuel,
                "total_enrg":mission_enrg+reserve_enrg,
                "mission_enrg":mission_enrg,
                "reserve_enrg":reserve_enrg,
                "mission_lod":mission_lod}


    def owe_performance(self, npax, mtow, range, cruise_speed, max_power, power_system, altitude_data, reserve_data):
        """Compute OWE from the point of view of mission
        delta_system_mass contains the battery weight or tank weight for GH2 or LH2 storage
        """
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"

        payload = npax * self.get_pax_allowance(range)

        dict = self.total_fuel(mtow, range, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)

        if power_system["energy_source"]==self.kerosene:
            energy_storage = 0.
        elif power_system["energy_source"]==self.gh2:
            energy_storage = dict["total_fuel"] * (1./self.gh2_tank_gravimetric_index - 1.)
        elif power_system["energy_source"]==self.lh2:
            energy_storage = dict["total_fuel"] * (1./self.lh2_tank_gravimetric_index - 1.)
        elif power_system["energy_source"]==self.battery:
            energy_storage = dict["total_enrg"]/self.battery_enrg_density
        else:
            raise Exception("power system - energy source is unknown")

        owe = mtow - payload - dict["total_fuel"]

        return {"owe":owe,
                "energy_storage":energy_storage,
                "total_energy":dict["total_enrg"],
                "total_fuel":dict["total_fuel"],
                "payload":payload,
                "lod":dict["mission_lod"]}


    def owe_structure(self, mtow, total_power, energy_storage, initial_power_system=None, target_power_system=None):
        owe = self.ref_owe(mtow)
        delta_engine_mass = 0.
        delta_system_mass = 0.
        if initial_power_system is not None:

            # Compute initial engine mass
            if initial_power_system["engine_type"]==self.piston:
                initial_engine_mass = total_power / self.piston_eng_pw_density
                initial_engine_mass += total_power / self.propeller_pw_density
            elif initial_power_system["engine_type"]==self.turboprop:
                initial_engine_mass = total_power / self.turboprop_pw_density
                initial_engine_mass += total_power / self.propeller_pw_density
            elif initial_power_system["engine_type"]==self.turbofan:
                initial_engine_mass = total_power / self.turbofan_pw_density
            else:
                raise Exception("initial power system - engine type is not allowed")

            if target_power_system is not None:

                # Compute new engine mass
                if target_power_system["engine_type"]==self.piston:
                    target_engine_mass = total_power / self.piston_eng_pw_density
                    target_engine_mass += total_power / self.propeller_pw_density
                elif target_power_system["engine_type"]==self.turboprop:
                    target_engine_mass = total_power / self.turboprop_pw_density
                    target_engine_mass += total_power / self.propeller_pw_density
                elif target_power_system["engine_type"]==self.turbofan:
                    target_engine_mass = total_power / self.turbofan_pw_density
                elif target_power_system["engine_type"]==self.emotor:
                    target_engine_mass = total_power / self.elec_motor_pw_density
                    target_engine_mass += total_power / self.power_elec_pw_density
                    if target_power_system["thruster"]==self.fan:
                        target_engine_mass += total_power / self.fan_nacelle_pw_density
                    elif target_power_system["thruster"]==self.propeller:
                        target_engine_mass += total_power / self.propeller_pw_density
                    else:
                        raise Exception("target power system - thruster type is unknown")
                    if target_power_system["energy_source"] in [self.gh2, self.lh2]:
                        delta_system_mass += (total_power / self.motor_eff) / self.fuel_cell_gravimetric_index
                        delta_system_mass += (total_power / self.motor_eff) / self.fc_system_gravimetric_index
                        eff = self.motor_eff*self.fuel_cell_eff
                        delta_system_mass += total_power * (1-eff)/eff / self.cooling_gravimetric_index  # All power which is not on the shaft have to be dissipated
                    elif target_power_system["energy_source"]==self.battery:
                        pass
                    else:
                        raise Exception("target power system - energy_source is unknown")
                else:
                    raise Exception("target power system - engine type is unknown")

                delta_engine_mass = target_engine_mass - initial_engine_mass
            else:
                target_engine_mass = np.nan
        else:
            initial_engine_mass = np.nan

        delta_system_mass += energy_storage

        return {"owe":owe+delta_engine_mass+delta_system_mass,
                "initial_engine_mass":initial_engine_mass,
                "target_engine_mass":target_engine_mass,
                "delta_engine_mass":delta_engine_mass,
                "delta_system_mass":delta_system_mass,
                "total_power":total_power}


    def design_airplane(self, npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system=None):

        if target_power_system is None: target_power_system = initial_power_system

        def fct(mtow):
            max_power = self.ref_power(mtow)
            total_power = max_power * n_engine
            dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, target_power_system, altitude_data, reserve_data)
            dict_s = self.owe_structure(mtow, total_power, dict_p["energy_storage"], initial_power_system, target_power_system)
            return (dict_p["owe"]-dict_s["owe"])/dict_s["owe"]

        mtow_ini = (-8.57e-15*npax*distance + 1.09e-04)*npax*distance
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        mtow = output_dict[0][0]
        max_power = self.ref_power(mtow)
        total_power = max_power * n_engine
        dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, target_power_system, altitude_data, reserve_data)
        dict_s = self.owe_structure(mtow, total_power, dict_p["energy_storage"], initial_power_system, target_power_system)

        payload_max = dict_p["payload"] * self.max_payload_factor
        mzfw = dict_s["owe"] + payload_max
        mlw = mzfw * self.mlw_factor

        return {"airplane_type":altitude_data["airplane_type"],
                "npax":npax,
                "design_range":distance,
                "cruise_speed":cruise_speed,
                "altitude_data":altitude_data,
                "reserve_data":reserve_data,

                "initial_power_system":initial_power_system,
                "initial_engine_mass":dict_s["initial_engine_mass"],

                "target_power_system":target_power_system,
                "target_engine_mass":dict_s["target_engine_mass"],

                "n_engine":n_engine,
                "max_power":max_power,
                "delta_engine_mass":dict_s["delta_engine_mass"],
                "delta_system_mass":dict_s["delta_system_mass"],
                "system_energy_density":dict_p["total_energy"]/(dict_s["delta_system_mass"]+dict_p["total_fuel"]),

                "mtow":mtow,
                "mlw":mlw,
                "mzfw":mzfw,
                "owe":dict_s["owe"],
                "payload":dict_p["payload"],
                "total_fuel":dict_p["total_fuel"],
                "total_energy":dict_p["total_energy"],

                "pk_o_mass":npax*distance/mtow,
                "pk_o_enrg":npax*distance/dict_p["total_energy"]}


    def get_best_range(self, range_init, range_step, criterion, npax, cruise_speed, altitude_data, reserve_data, n_engine, power_system, target_power_system=None):
        """Look for the design range that maximizes the criterion
        """
        def fct(distance):
            dict = self.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system, target_power_system=target_power_system)
            return dict[criterion]

        dx = range_step
        distance, res, rc = utils.maximize_1d(range_init,dx,[fct])

        dict = self.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system, target_power_system=target_power_system)
        return dict


    def get_app_speed(self, wing_area, mass):
        disa = 0.
        altp = unit.m_ft(0.)
        pamb,tamb,g = self.phd.atmosphere(altp, disa)
        rho = self.phd.gas_density(pamb,tamb)
        cl = self.get_cl_max_ld() / self.kvs1g_ld**2
        vapp = np.sqrt((2*mass*g)/(rho*wing_area*cl))
        alpha, betha = self.tuner_app
        app_speed = alpha*vapp + betha  # Tuning versus data base
        return app_speed


    def get_tofl(self, total_power, wing_area, mass):
        disa = 0.
        altp = unit.m_ft(0.)
        pamb,tamb,g = self.phd.atmosphere(altp, disa)
        rho = self.phd.gas_density(pamb,tamb)
        sigma = rho/1.225
        cl = self.get_cl_max_to() / self.kvs1g_to**2
        vtas35ft = np.sqrt((2*mass*g)/(rho*wing_area*cl))
        fn_max = (total_power / (1.0*vtas35ft)) * self.prop_eff
        ml_factor = mass**2 / (cl*fn_max*wing_area*sigma**0.8 )  # Magic Line factor
        alpha, betha = self.tuner_to
        tofl = alpha*ml_factor + betha    # Magic line
        return tofl


    def design_airplane_v2(self, npax, distance, cruise_speed, altitude_data, reserve_data,
                           to_field, app_speed, n_engine, initial_power_system, target_power_system=None):

        def fct(x):
            total_power = x[0]
            wing_area = x[1]
            dict = self.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data,
                                        initial_power_system, target_power_system=target_power_system)
            tofl = self.get_tofl(total_power, wing_area, dict['mtow'])
            vapp = self.get_app_speed(wing_area, dict['mlw'])
            return [tofl-to_field, vapp-app_speed]

        mtow_ini = (-8.57e-15*npax*distance + 1.09e-04)*npax*distance
        total_power = self.ref_power(self, mtow_ini) * n_engine
        wing_area = mtow_ini*(2.94/app_speed)**2

        x_ini = [total_power, wing_area]
        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        max_power = output_dict[0][0] / n_engine
        wing_area = output_dict[0][1]

        return


    def print_design(self, dict, content="all"):

        if content=="all":
            print("")
            print("========================================================")
            print(" Airplane type = ", dict["airplane_type"])
            print(" Number of engine = ", dict["n_engine"])
            print(" Number of passenger = ", dict["npax"])
            print(" Design range = ", unit.convert_to("km", dict["design_range"]), " km")
            print(" Cruise speed = ", unit.convert_to("km/h", dict["cruise_speed"]), " km/h")
            print("")
            print("--------------------------------------------------------")
            print(" Max power = ", "%.0f"%unit.kW_W(dict["max_power"]), " kW")
            print("")
            print(" Initial engine type = ", dict["initial_power_system"]["engine_type"])
            print(" Initial thruster type = ", dict["initial_power_system"]["thruster"])
            print(" Initial energy source = ", dict["initial_power_system"]["energy_source"])
            print(" Initial engine mass = ", "%.0f"%dict["initial_engine_mass"])
            print("")
            print(" Target engine type = ", dict["target_power_system"]["engine_type"])
            print(" Target thruster type = ", dict["target_power_system"]["thruster"])
            print(" Target energy source = ", dict["target_power_system"]["energy_source"])
            print(" Target engine mass = ", "%.0f"%dict["target_engine_mass"])
            print("")
            print(" Delta engine mass = ", "%.0f"%dict["delta_engine_mass"], " kg")
            print(" Delta system mass = ", "%.0f"%dict["delta_system_mass"], " kg")
            print(" System energy density = ", "%.0f"%unit.Wh_J(dict["system_energy_density"]), " Wh/kg")
            print("")
            print(" mtow = ", "%.0f"%dict["mtow"], " kg")
            print(" owe = ", "%.0f"%dict["owe"], " kg")
            print(" payload = ", "%.0f"%dict["payload"], " kg")
            print(" Total fuel = ", "%.0f"%dict["total_fuel"], " kg")
            print(" Total energy = ", "%.0f"%unit.kWh_J(dict["total_energy"]), " kWh")
            print("")
            print(" Mass efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict["pk_o_mass"]), " pax.km/kg")
            print(" Energy efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")

        elif content=="criteria":
            print("")
            print(" Design range = ", "%.0f"%unit.convert_to("km", dict["design_range"]), " km")
            print(" Mass efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict["pk_o_mass"]), " pax.km/kg")
            print(" Energy efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")


    def fly_tow(self, ac_dict, tow, input, input_type="pax"):
        """Compute data from tow & payload
        """
        mpax = ac_dict["payload"]/ac_dict["npax"]
        if input_type=="pax":
            payload = input*mpax
        elif input_type=="mass":
            payload = input
        else:
            raise Exception("input_type is unknown")

        cruise_speed = ac_dict["cruise_speed"]
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"

        mtow = ac_dict["mtow"]
        owe = ac_dict["owe"]
        max_power = ac_dict["max_power"]
        power_system = ac_dict["target_power_system"]
        altitude_data = ac_dict["altitude_data"]
        reserve_data = ac_dict["reserve_data"]

        def fct(distance):
            dict = self.total_fuel(tow, distance, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)
            return tow - (owe + payload + dict["total_fuel"])

        dist_ini = 0.75*ac_dict["design_range"]
        output_dict = fsolve(fct, x0=dist_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        distance = output_dict[0][0]
        dict = self.total_fuel(tow, distance, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)
        dict["payload"] = payload
        return dict


    def fly_distance(self, ac_dict, distance, input, input_type="pax"):
        """Compute data from distance & payload
        """
        mpax = ac_dict["payload"]/ac_dict["npax"]
        if input_type=="pax":
            payload = input*mpax
        elif input_type=="mass":
            payload = input
        else:
            raise Exception("input_type is unknown")

        cruise_speed = ac_dict["cruise_speed"]
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"

        mtow = ac_dict["mtow"]
        owe = ac_dict["owe"]
        max_power = ac_dict["max_power"]
        power_system = ac_dict["target_power_system"]
        altitude_data = ac_dict["altitude_data"]
        reserve_data = ac_dict["reserve_data"]

        def fct(tow):
            dict = self.total_fuel(tow, distance, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)
            return tow - (owe + payload + dict["total_fuel"])

        tow_ini = 0.75*mtow
        output_dict = fsolve(fct, x0=tow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        tow = output_dict[0][0]
        dict = self.total_fuel(tow, distance, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)
        dict["payload"] = payload
        return dict


    def build_payload_range(self, ac_dict):
        """Compute payload - range characteristics and add them to ac_dict
        """
        mtow = ac_dict["mtow"]

        payload_max = ac_dict["payload"] * self.max_fuel_payload_factor
        dict = self.fly_tow(ac_dict, mtow, payload_max, input_type="mass")
        ac_dict["payload_max"] = payload_max        # Maximum payload
        ac_dict["range_pl_max"] = dict["distance"]  # Range for maximum payload mission

        payload_max_fuel =  ac_dict["payload"] * self.max_fuel_payload_factor
        dict = self.fly_tow(ac_dict, mtow, payload_max_fuel, input_type="mass")
        ac_dict["payload_fuel_max"] = payload_max_fuel  # Payload for max fuel mission
        ac_dict["range_fuel_max"] = dict["distance"]    # Range for max fuel mission

        tow_zero_payload = mtow - payload_max_fuel
        dict = self.fly_tow(ac_dict, tow_zero_payload, 0., input_type="mass")
        ac_dict["range_no_pl"] = dict["distance"]       # Range for zero payload mission



if __name__ == '__main__':

    phd = PhysicalData()
    ddm = DDM(phd)

    # Airplane design analysis
    #-------------------------------------------------------------------------------------------------------------------
    npax = 6
    n_engine = 1
    distance = unit.convert_from("km", 400)
    cruise_speed = unit.convert_from("km/h", 180)

    airplane_type = "general"
    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)

    initial_power_system = {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.kerosene}

    tpws = [{"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.kerosene},
            {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.gh2},
            {"thruster":ddm.propeller, "engine_type":ddm.piston, "energy_source":ddm.lh2},
            {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.battery},
            {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.gh2},
            {"thruster":ddm.propeller, "engine_type":ddm.emotor, "energy_source":ddm.lh2}]

    target_power_system = tpws[3]

    ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system)
    ddm.print_design(ac_dict)

