#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
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

        self.commuter = "commuter"
        self.business = "business"
        self.narrow_body = "narrow_body"
        self.wide_body = "wide_body"

        self.pax_allowance = {"commuter":100, "business":150, "narrow_body":130, "wide_body":150}  # kg
        self.ref_tsfc = {"commuter":0.17e-5, "business":0.17e-5, "narrow_body":0.15e-5, "wide_body":0.15e-5} # kg/N/s
        self.ref_psfc = {"commuter":0.67e-7, "business":0.67e-7} # kg/w/s

    def get_tas(self,tamb,speed,speed_type):
        if speed_type=="mach": return speed*self.phd.sound_speed(tamb)
        elif speed_type=="tas": return speed

    def cruise_altp(self,airplane_type):
        """return [cruise altitude, diversion altitude]
        """
        if airplane_type==self.commuter:
            mz, dz, hz = unit.m_ft(20000), unit.m_ft(10000), unit.m_ft(1500)
        elif airplane_type in [self.business, self.narrow_body]:
            mz, dz, hz = unit.m_ft(35000), unit.m_ft(25000), unit.m_ft(1500)
        elif airplane_type==self.wide_body:
            mz, dz, hz = unit.m_ft(35000), unit.m_ft(25000), unit.m_ft(1500)
        return {"mission":mz, "diversion":dz, "holding":hz}

    def reserve_data(self,airplane_type):
        """return [mission fuel factor, diversion leg, holding time]
        """
        if airplane_type==self.commuter:
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

    def lod(self,propulsion_type,airplane_type):
        return 17.

    def ref_power(self, mtow):
        return mtow * 250

    def ref_owe(self, mtow):
        return mtow * 0.5

    def leg_fuel(self,start_mass,distance,altp,speed,speed_type,propulsion_type,airplane_type):
        """Compute the fuel over a given distance
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.lod(propulsion_type,airplane_type)
        if propulsion_type==self.turbofan:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.tsfc(airplane_type)
            return start_mass*(1-np.exp(-(sfc*g*distance)/(tas*lod)))              # turbofan
        elif propulsion_type==self.fan_battery:
            return start_mass*g*distance / (self.eta_fan*self.eta_motor*lod)       # fan_battery
        elif propulsion_type==self.turboprop:
            sfc = self.psfc(airplane_type)
            return start_mass*(1.-np.exp(-(sfc*g*distance)/(self.eta_prop*lod)))   # turboprop
        elif propulsion_type==self.prop_battery:
            return start_mass*g*distance / (self.eta_prop*self.eta_motor*lod)      # prop_battery

    def holding_fuel(self,start_mass,time,altp,speed,speed_type,propulsion_type,airplane_type):
        """Compute the fuel for a given holding time
        """
        pamb,tamb,g = self.phd.atmosphere(altp, self.disa)
        lod = self.lod(propulsion_type,airplane_type)
        if propulsion_type==self.turbofan:
            sfc = self.tsfc(airplane_type)
            return start_mass*(1 - np.exp(-g*sfc*time/lod))             # turbofan
        elif propulsion_type==self.fan_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            return start_mass*g*tas*time / (self.eta_fan*self.eta_motor*lod)       # fan_battery
        elif propulsion_type==self.turboprop:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc = self.psfc(airplane_type)
            return start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.eta_prop*lod)))   # turboprop
        elif propulsion_type==self.prop_battery:
            tas = self.get_tas(tamb,speed,speed_type)
            return start_mass*g*tas*time / (self.eta_prop*self.eta_motor*lod)      # prop_battery

    def total_fuel(self,tow,range,cruise_speed,speed_type,propulsion_type,airplane_type):
        """Compute the total fuel required for a mission
        """
        altitude = self.cruise_altp(airplane_type)
        cruise_altp = altitude["mission"]
        mission_fuel = self.leg_fuel(tow,range,cruise_altp,cruise_speed,speed_type,propulsion_type,airplane_type)
        ldw = tow - mission_fuel
        data = self.reserve_data(airplane_type)
        reserve_fuel = 0.
        if data["fuel_factor"]>0:
            reserve_fuel += data["fuel_factor"]*mission_fuel
        if data["diversion_leg"]>0:
            leg = data["diversion_leg"]
            diversion_altp = altitude["diversion"]
            reserve_fuel += self.leg_fuel(ldw,leg,diversion_altp,cruise_speed,speed_type,propulsion_type,airplane_type)
        if data["holding_time"]>0:
            time = data["holding_time"]
            holding_altp = altitude["diversion"]
            speed = 0.5 * cruise_speed
            reserve_fuel += self.holding_fuel(ldw,time,holding_altp,speed,speed_type,propulsion_type,airplane_type)
        return mission_fuel+reserve_fuel

    def owe_performance(self, npax, mtow, range, cruise_speed, propulsion_type, airplane_type):
        if cruise_speed>1:
            speed_type = "tas"
        else:
            speed_type = "mach"
        total_fuel = self.total_fuel(mtow,range,cruise_speed,speed_type,propulsion_type,airplane_type)
        payload = npax * self.pax_allowance[airplane_type]
        return mtow-payload-total_fuel












