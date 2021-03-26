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
        self.fan_nacelle_pw_density = unit.W_kW(10) # Wh/kg

        self.battery_enrg_density = unit.J_Wh(400)  # Wh/kg
        self.battery_vol_density = 2500.            # kg/m3

        self.fuel_cell_gravimetric_index = unit.convert_from("kW/kg", 5)    # kW/kg, Horizon Fuel Cell
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
        self.mpax_allowance_med = [120, unit.m_km(8000)]
        self.mpax_allowance_high = [150, unit.m_km(np.inf)]

        self.lod_low = [15, 1000]
        self.lod_high = [20, 200000]

        self.psfc_low = [unit.convert_from("lb/shp/h",0.6), unit.convert_from("kW",50)]     # here power is
        self.psfc_high = [unit.convert_from("lb/shp/h",0.4), unit.convert_from("kW",1000)]

        self.tsfc_low = [unit.convert_from("kg/daN/h",0.60), unit.convert_from("MW",1)]     # Here, power is equivalent shaft power developped during take off as defined in the data base
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


    def get_psfc(self,max_power, fuel_type):
        psfc_max, pw_min = self.psfc_low
        psfc_min, pw_max = self.psfc_high

        if fuel_type==self.kerosene:
            pass
        elif fuel_type==self.gh2:
            psfc_min *= self.kerosene_heat/self.gh2_heat
            psfc_max *= self.kerosene_heat/self.gh2_heat
        elif fuel_type==self.lh2:
            psfc_min *= self.kerosene_heat/self.lh2_heat
            psfc_max *= self.kerosene_heat/self.lh2_heat
        else:
            raise Exception("fuel type is unknown")

        if max_power<pw_min:
            return psfc_max
        elif max_power<pw_max:
            return psfc_max - (psfc_max-psfc_min)*(max_power-pw_min)/(pw_max-pw_min)
        else:
            return psfc_min


    def get_tsfc(self,max_power, fuel_type):
        tsfc_max, pw_min = self.tsfc_low
        tsfc_min, pw_max = self.tsfc_high

        if fuel_type==self.kerosene:
            pass
        elif fuel_type==self.gh2:
            tsfc_min *= self.kerosene_heat/self.gh2_heat
            tsfc_max *= self.kerosene_heat/self.gh2_heat
        elif fuel_type==self.lh2:
            tsfc_min *= self.kerosene_heat/self.lh2_heat
            tsfc_max *= self.kerosene_heat/self.lh2_heat
        else:
            raise Exception("fuel type is unknown")

        if max_power<pw_min:
            return tsfc_max
        elif max_power<pw_max:
            return tsfc_max - (tsfc_max-tsfc_min)*(max_power-pw_min)/(pw_max-pw_min)
        else:
            return tsfc_min


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
        else:
            raise Exception("airplane type is unknown")
        return {"fuel_factor":ff, "diversion_leg":dl, "holding_time":ht}


    def get_tas(self,tamb,speed,speed_type):
        if speed_type=="mach":
            vsnd = self.phd.sound_speed(tamb)
            tas = speed * vsnd
            return tas
        elif speed_type=="tas":
            return speed


    def leg_fuel(self,start_mass,distance,altp,speed,speed_type,mtow,max_power,power_system):
        """Compute the fuel over a given distance
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.get_lod(mtow)
        if power_system["engine_type"]==self.piston:
            sfc = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # piston engine
        elif power_system["engine_type"]==self.turboprop:
            sfc = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # turboprop
        elif power_system["engine_type"]==self.turbofan:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_tsfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(tas*lod)))             # turbofan
        elif power_system["engine_type"]==self.emotor:
            if power_system["thruster"]==self.propeller:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    eff = self.eta_prop * self.eta_motor * self.eta_fuel_cell
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))             # electroprop + fuel cell
                elif power_system["energy_source"]==self.battery:
                    fuel = start_mass*g*distance / (self.eta_prop*self.eta_motor*lod)      # electroprop + battery
            elif power_system["thruster"]==self.fan:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    eff = self.eta_fan * self.eta_motor * self.eta_fuel_cell
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))             # electrofan + fuel cell
                elif power_system["energy_source"]==self.battery:
                    fuel = start_mass*g*distance / (self.eta_fan*self.eta_motor*lod)       # electrofan + battery
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

        return fuel * 1.05  # WARNING: correction to take account of climb phases


    def holding_fuel(self,start_mass,time,altp,speed,speed_type,mtow,max_power,power_system):
        """Compute the fuel for a given holding time
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.get_lod(mtow)
        if power_system["engine_type"]==self.piston:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # piston
        elif power_system["engine_type"]==self.turboprop:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.get_psfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # turboprop
        elif power_system["engine_type"]==self.turbofan:
            sfc = self.get_tsfc(max_power, power_system["energy_source"])
            fuel = start_mass*(1 - np.exp(-g*sfc*time/lod))                         # turbofan
        elif power_system["engine_type"]==self.emotor:
            if power_system["thruster"]==self.propeller:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    tas = self.get_tas(tamb,speed,speed_type)
                    eff = self.eta_prop * self.eta_motor * self.eta_fuel_cell
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))            # electroprop + fuel cell
                elif power_system["energy_source"]==self.battery:
                    tas = self.get_tas(tamb,speed,speed_type)
                    fuel = start_mass*g*tas*time / (self.eta_prop*self.eta_motor*lod)      # electroprop + battery
            elif power_system["thruster"]==self.fan:
                if power_system["energy_source"] in [self.gh2, self.lh2]:
                    tas = self.get_tas(tamb,speed,speed_type)
                    eff = self.eta_fan * self.eta_motor * self.eta_fuel_cell
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))             # electrofan + fuel cell
                elif power_system["energy_source"]==self.battery:
                    tas = self.get_tas(tamb,speed,speed_type)
                    fuel = start_mass*g*tas*time / (self.eta_fan*self.eta_motor*lod)        # electrofan + battery
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

        return fuel


    def total_fuel(self,tow,range,cruise_speed,speed_type,mtow,max_power,power_system,altitude_data,reserve_data):
        """Compute the total fuel required for a mission
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        cruise_altp = altitude_data["mission"]
        mission_fuel = self.leg_fuel(tow,range,cruise_altp,cruise_speed,speed_type,mtow,max_power,power_system)
        if power_system["energy_source"]==self.battery:
            ldw = tow
        else:
            ldw = tow - mission_fuel

        reserve_fuel = 0.
        if reserve_data["fuel_factor"]>0:
            reserve_fuel += reserve_data["fuel_factor"]*mission_fuel
        if reserve_data["diversion_leg"]>0:
            leg = reserve_data["diversion_leg"]
            diversion_altp = altitude_data["diversion"]
            reserve_fuel += self.leg_fuel(ldw,leg,diversion_altp,cruise_speed,speed_type,mtow,max_power,power_system)
        if reserve_data["holding_time"]>0:
            time = reserve_data["holding_time"]
            holding_altp = altitude_data["holding"]
            speed = 0.5 * cruise_speed
            reserve_fuel += self.holding_fuel(ldw,time,holding_altp,speed,speed_type,mtow,max_power,power_system)

        return mission_fuel+reserve_fuel


    def owe_performance(self, npax, mtow, range, cruise_speed, max_power, power_system, altitude_data, reserve_data):
        """Compute OWE from the point of view of mission
        delta_system_mass contains the battery weight or tank weight for GH2 or LH2 storage
        """
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"

        payload = npax * self.get_pax_allowance(range)

        medium = self.total_fuel(mtow, range, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)

        if power_system["energy_source"]==self.kerosene:
            total_fuel = medium
            energy_storage = 0.
        elif power_system["energy_source"]==self.gh2:
            total_fuel = medium
            energy_storage = total_fuel * (1./self.gh2_tank_gravimetric_index - 1.)
        elif power_system["energy_source"]==self.lh2:
            total_fuel = medium
            energy_storage = total_fuel * (1./self.lh2_tank_gravimetric_index - 1.)
        elif power_system["energy_source"]==self.battery:
            total_fuel = 0.
            energy_storage = medium/self.battery_enrg_density
        else:
            raise Exception("power system - energy source is unknown")

        owe = mtow - payload - total_fuel

        return {"owe":owe,
                "energy_storage":energy_storage,
                "total_fuel":total_fuel,
                "payload":payload}


    def owe_structure(self, mtow, energy_storage, initial_power_system=None, target_power_system=None):
        power = self.ref_power(mtow)
        owe = self.ref_owe(mtow)
        delta_engine_mass = 0.
        delta_system_mass = 0.
        if initial_power_system is not None:
            if target_power_system is not None:

                # remove initial engine mass
                if initial_power_system["engine_type"]==self.piston:
                    delta_engine_mass -= power / self.piston_eng_pw_density
                elif initial_power_system["engine_type"]==self.turboprop:
                    delta_engine_mass -= power / self.turboprop_pw_density
                elif initial_power_system["engine_type"]==self.turbofan:
                    delta_engine_mass -= power / self.turbofan_pw_density
                else:
                    raise Exception("power system - engine type is unknown for initial architecture")

                # Add new engine mass
                if target_power_system["engine_type"]==self.piston:
                    delta_engine_mass += power / self.piston_eng_pw_density
                elif target_power_system["engine_type"]==self.turboprop:
                    delta_engine_mass += power / self.turboprop_pw_density
                elif target_power_system["engine_type"]==self.turbofan:
                    delta_engine_mass += power / self.turbofan_pw_density
                elif target_power_system["engine_type"]==self.emotor:
                    delta_engine_mass += power / self.elec_motor_pw_density
                    if target_power_system["thruster"]==self.fan:
                        delta_engine_mass += power / self.fan_nacelle_pw_density
                    elif target_power_system["thruster"]==self.propeller:
                        pass
                    else:
                        raise Exception("power system - thruster type is unknown")
                    if target_power_system["energy_source"] in [self.gh2, self.lh2]:
                        delta_system_mass += power / self.eta_motor / self.fuel_cell_gravimetric_index
                        eff = self.eta_motor*self.eta_fuel_cell
                        delta_system_mass += power * (1-eff)/eff / self.cooling_gravimetric_index  # All power which is not on the shaft have to be dissipated
                    elif target_power_system["energy_source"]==self.battery:
                        pass
                    else:
                        raise Exception("power system - energy_source is unknown for target architecture")
                else:
                    raise Exception("power system - engine type is unknown")

        return {"owe":owe+energy_storage+delta_engine_mass+delta_system_mass,
                "delta_engine_mass":delta_engine_mass,
                "delta_system_mass":delta_system_mass,
                "shaft_power":power}


    def mass_mission_adapt(self, npax, distance, cruise_speed, altitude_data, reserve_data, power_system, target_power_system=None):

        if target_power_system is None: target_power_system = power_system

        def fct(mtow):
            max_power = self.ref_power(mtow)
            dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, target_power_system,altitude_data, reserve_data)
            dict_s = self.owe_structure(mtow, dict_p["energy_storage"], initial_power_system=power_system, target_power_system=target_power_system)
            return (dict_p["owe"]-dict_s["owe"])/dict_s["owe"]

        mtow_ini = (-8.57e-15*npax*distance + 1.09e-04)*npax*distance
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        mtow = output_dict[0][0]
        max_power = self.ref_power(mtow)
        dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, target_power_system, altitude_data, reserve_data)
        dict_s = self.owe_structure(mtow, dict_p["energy_storage"], initial_power_system=power_system, target_power_system=target_power_system)

        return {"mtow":mtow,
                "owe":dict_s["owe"],
                "payload":dict_p["payload"],
                "delta_engine_mass":dict_s["delta_engine_mass"],
                "energy_management_mass":dict_p["energy_storage"]+dict_s["delta_system_mass"],
                "total_fuel":dict_p["total_fuel"],
                "shaft_power":dict_s["shaft_power"]}



