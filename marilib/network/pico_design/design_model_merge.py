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

from marilib.utils.math import lin_interp_1d

from physical_data import PhysicalData

from analyse_data import coloration, read_db, lin_lst_reg, draw_reg, subplots_by_varname,\
    draw_colored_cloud_on_axis, get_error, do_regression


#-----------------------------------------------------------------------------------------------------------------------
#
#  Main class
#
#-----------------------------------------------------------------------------------------------------------------------


class DDM(object):                  # Data Driven Modelling

    def __init__(self, phd):
        self.phd = phd

        self.disa = 0.
        self.max_payload_factor = 1.15          # max_payload = nominal_paylod * max_payload_factor
        self.max_fuel_payload_factor = 0.60     # max_fuel_payload = nominal_payload * max_fuel_payload_factor
        self.mlw_factor = 1.07                  # MLW = MZFW * mlw_factor

        self.petrol_heat = unit.convert_from("MJ",43)  # MJ/kg
        self.hydrogen_heat = unit.convert_from("MJ",121) # MJ/kg

        # Efficiencies
        self.prop_eff = 0.80
        self.fan_eff = 0.82
        self.motor_eff = 0.95       # MAGNIX
        self.fuel_cell_eff = 0.50   # (Horizon Fuel Cell)

        # Engine power densities
        self.turbofan_pw_density = unit.W_kW(7)  # W/kg
        self.turboprop_pw_density = unit.W_kW(5) # W/kg
        self.piston_eng_pw_density = unit.W_kW(2) # W/kg
        self.elec_motor_pw_density = unit.W_kW(4.5) # W/kg
        self.power_elec_pw_density = unit.W_kW(40) # W/kg
        self.propeller_pw_density = unit.W_kW(15) # W/kg
        self.fan_nacelle_pw_density = unit.W_kW(10) # W/kg

        # Other densities
        self.battery_enrg_density = unit.J_Wh(250)  # Wh/kg
        self.battery_vol_density = 2500.            # kg/m3

        self.fuel_cell_gravimetric_index = unit.convert_from("kW/kg", 5)    # kW/kg, Horizon Fuel Cell
        self.fc_system_gravimetric_index = unit.convert_from("kW/kg", 2)    # kW/kg
        self.cooling_gravimetric_index = unit.convert_from("kW/kg", 5)      # kW/kg, Dissipated thermal power per cooloing system mass

        self.initial_gh2_pressure = unit.convert_from("bar", 700)
        self.tank_efficiency_factor = unit.convert_from("bar", 930e-3)  # bar.L/kg, state of the art efficiency factor for pressured vessel

        h2_density = phd.fuel_density("compressed_h2", press=self.initial_gh2_pressure)

        # kg_H2 / (kg_H2 + kg_Tank)
        self.gh2_tank_gravimetric_index = 1/(1+self.initial_gh2_pressure/(self.tank_efficiency_factor*h2_density))
        self.gh2_tank_volumetric_index = 25         # kgH2 / (m3_Tank_+_H2)

        self.lh2_tank_gravimetric_index = 0.30      # kg_H2 / (kg_H2 + kg_Tank)
        self.lh2_tank_volumetric_index = 0.800      # m3_H2 / (m3_H2 + m3_Tank)     No used in this version

        # Key definitions
        self.propeller = "propeller"    # thruster key
        self.fan = "fan"                # thruster key

        self.piston = "piston"          # engine_type key
        self.turbofan = "turbofan"      # engine_type key
        self.turboprop = "turboprop"    # engine_type key
        self.emotor = "emotor"          # engine_type key

        self.petrol = "petrol"          # energy_type key
        self.gh2 = "gh2"                # energy_type key
        self.lh2 = "lh2"                # energy_type key
        self.battery = "battery"        # energy_type key

        self.default_power_system = {"thruster_type":None, "engine_type":None, "energy_type":self.petrol}

        self.general = "general"
        self.commuter = "commuter"
        self.business = "business"
        self.narrow_body = "narrow_body"
        self.wide_body = "wide_body"

        self.cl_max_to = 2.0            # Not used in this version
        self.kvs1g_to = 1.13            # Not used in this version
        self.tuner_to = [12., 300]      # Not used in this version

        self.cl_max_ld = 2.6            # Not used in this version
        self.kvs1g_ld = 1.23            # Not used in this version
        self.tuner_app = [1.15, -6]     # Not used in this version

        self.delta_power = 0.           # Reference power tuning
        self.delta_mwe = 0.             # Reference owe & standard mwe tuning

        self.wing_area = None


    def get_lod(self, mtow):
        mtow_list = [200., 20000., 200000., 500000.]
        lod_list = [13., 15.,  19.,  20.]
        lod = lin_interp_1d(mtow, mtow_list, lod_list)
        return lod


    def get_fuel_heat(self, fuel_type):
        if fuel_type==self.petrol:
            fhv = self.petrol_heat
        elif fuel_type in [self.gh2, self.lh2]:
            fhv = self.hydrogen_heat
        else:
            raise Exception("fuel type is unknown")
        return fhv


    def get_sfc(self,  sfc_type, max_power, fuel_type):
        if fuel_type==self.petrol:
            fhv = self.petrol_heat
            ksfc = 1
        elif fuel_type in [self.gh2, self.lh2]:
            fhv = self.hydrogen_heat
            ksfc = self.petrol_heat/self.hydrogen_heat
        else:
            raise Exception("fuel type is unknown")

        if sfc_type=="psfc":
            power_list = [10.e3, 50.e3, 500.e3, 80000.e3]
            sfc_list = unit.convert_from("lb/shp/h", [0.66, 0.65,  0.51,  0.50])
        elif sfc_type=="tsfc":
            power_list = [500.e3, 5.e6, 50.e6, 100.e6]
            sfc_list = unit.convert_from("kg/daN/h", [0.7, 0.66,  0.56,  0.55])
        else:
            raise Exception("SFC type is unknown")

        sfc = ksfc * lin_interp_1d(max_power, power_list, sfc_list)
        return sfc, fhv


    def ref_power(self, mtow):
        """Required total power for an airplane with a given MTOW
        """
        # a, b, c = [7.60652439e-05, 2.04865987e+02, -6.6e+04]   # Constant c adjusted for very small aircraft
        a, b, c = [7.48446406e-05, 2.04684537e+02, -9.0e4]   # Constant c adjusted for very small aircraft
        power = (a*mtow + b)*mtow + c + self.delta_power
        # power = (0.0197*mtow + 100.6)*mtow
        return power


    def propulsion_mass(self, power_system, total_power):

        energy_type = power_system["energy_type"]
        engine_type = power_system["engine_type"]
        thruster_type = power_system["thruster_type"]

        if engine_type==self.piston:
            propulsion_mass = total_power / self.piston_eng_pw_density
            propulsion_mass += total_power / self.propeller_pw_density
        elif engine_type==self.turboprop:
            propulsion_mass = total_power / self.turboprop_pw_density
            propulsion_mass += total_power / self.propeller_pw_density
        elif engine_type==self.turbofan:
            propulsion_mass = total_power / self.turbofan_pw_density
        elif engine_type==self.emotor:
            propulsion_mass = total_power / self.elec_motor_pw_density
            propulsion_mass += total_power / self.power_elec_pw_density
            if thruster_type==self.fan:
                propulsion_mass += total_power / self.fan_nacelle_pw_density
            elif thruster_type==self.propeller:
                propulsion_mass += total_power / self.propeller_pw_density
            else:
                raise Exception("target power system - thruster type is unknown")
            
            if energy_type in [self.gh2, self.lh2]:
                propulsion_mass += (total_power / self.motor_eff) / self.fuel_cell_gravimetric_index
                propulsion_mass += (total_power / self.motor_eff) / self.fc_system_gravimetric_index
                eff = self.motor_eff*self.fuel_cell_eff
                propulsion_mass += total_power * (1-eff)/eff / self.cooling_gravimetric_index  # All power which is not on the shaft have to be dissipated
            elif energy_type==self.battery:
                pass
            else:
                raise Exception("target power system - energy_type is unknown")
        else:
            raise Exception("engine type is unknown")

        return propulsion_mass


    def standard_mass(self, mtow):
        """Averaged Standard MWE for an airplane with a given MTOW
        """
        # a, b, c = [-1.21221005e-06, 5.18342786e-01, 0]
        a, b, c = [-1.25231743e-06, 5.16730879e-01, -35]   # Constant c adjusted for very small aircraft
        std_mass = (a*mtow + b)*mtow + c + self.delta_mwe
        return std_mass


    def furnishing(self, npax):
        """Semi empirical furnishing mass
        """
        a = 18                              # WARNING : for airplanes with less than 250 pax
        furn = a*npax
        return furn


    def op_item(self, npax, distance):
        """Semi empirical mass for operator items
        """
        a = 5.5e-6
        op_item = a*npax*distance
        return op_item


    def get_pax_allowance(self,distance):
        """Compute passenger mass allowance
        WARNING : continuous function to avoid convergence problem within solvings
        """
        dist_list = [unit.m_km(1000), unit.m_km(20000)]
        mpax_list = [95, 140]
        mpax = lin_interp_1d(distance, dist_list, mpax_list)
        return mpax


    def cruise_altp(self,airplane_type):
        """return [cruise altitude, diversion altitude, holding altitude]
        """
        if airplane_type==self.general:
            mz, dz, hz = unit.m_ft(5000), unit.m_ft(3000), unit.m_ft(1500)
        elif airplane_type==self.commuter:
            mz, dz, hz = unit.m_ft(10000), unit.m_ft(10000), unit.m_ft(1500)
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
            sfc,fhv = self.get_sfc("psfc", max_power, power_system["energy_type"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.prop_eff*lod)))        # piston engine
            enrg = fuel*fhv
        elif power_system["engine_type"]==self.turboprop:
            sfc,fhv = self.get_sfc("psfc", max_power, power_system["energy_type"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(self.prop_eff*lod)))        # turboprop
            enrg = fuel*fhv
        elif power_system["engine_type"]==self.turbofan:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc,fhv = self.get_sfc("tsfc", max_power, power_system["energy_type"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*distance)/(tas*lod)))                  # turbofan
            enrg = fuel*fhv
        elif power_system["engine_type"]==self.emotor:
            if power_system["thruster_type"]==self.propeller:
                if power_system["energy_type"] in [self.gh2, self.lh2]:                 # electroprop + fuel cell
                    eff = self.prop_eff * self.motor_eff * self.fuel_cell_eff
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))
                    enrg = fuel*fhv
                elif power_system["energy_type"]==self.battery:                         # electroprop + battery
                    enrg = start_mass*g*distance / (self.prop_eff*self.motor_eff*lod)
                    fuel = 0.
            elif power_system["thruster_type"]==self.fan:
                if power_system["energy_type"] in [self.gh2, self.lh2]:
                    eff = self.fan_eff * self.motor_eff * self.fuel_cell_eff            # electrofan + fuel cell
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1.-np.exp(-(g*distance)/(eff*fhv*lod)))
                    enrg = fuel*fhv
                elif power_system["energy_type"]==self.battery:
                    enrg = start_mass*g*distance / (self.fan_eff*self.motor_eff*lod)    # electrofan + battery
                    fuel = 0.
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

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
            sfc,fhv = self.get_sfc("psfc", max_power, power_system["energy_type"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.prop_eff*lod)))            # piston
            enrg = fuel*fhv
        elif power_system["engine_type"]==self.turboprop:
            tas = self.get_tas(tamb,speed,speed_type)
            sfc,fhv = self.get_sfc("psfc", max_power, power_system["energy_type"])
            fuel = start_mass*(1.-np.exp(-(sfc*g*tas*time)/(self.prop_eff*lod)))            # turboprop
            enrg = fuel*fhv
        elif power_system["engine_type"]==self.turbofan:
            sfc,fhv = self.get_sfc("tsfc", max_power, power_system["energy_type"])
            fuel = start_mass*(1 - np.exp(-g*sfc*time/lod))                                 # turbofan
            enrg = fuel*fhv
        elif power_system["engine_type"]==self.emotor:
            if power_system["thruster_type"]==self.propeller:
                if power_system["energy_type"] in [self.gh2, self.lh2]:
                    tas = self.get_tas(tamb,speed,speed_type)
                    eff = self.prop_eff * self.motor_eff * self.fuel_cell_eff
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))             # electroprop + fuel cell
                    enrg = fuel*fhv
                elif power_system["energy_type"]==self.battery:
                    tas = self.get_tas(tamb,speed,speed_type)
                    enrg = start_mass*g*tas*time / (self.prop_eff*self.motor_eff*lod)       # electroprop + battery
                    fuel = 0.
            elif power_system["thruster_type"]==self.fan:
                if power_system["energy_type"] in [self.gh2, self.lh2]:
                    tas = self.get_tas(tamb,speed,speed_type)
                    eff = self.fan_eff * self.motor_eff * self.fuel_cell_eff
                    fhv = self.hydrogen_heat
                    fuel = start_mass*(1 - np.exp(-(g*tas*time)/(eff*fhv*lod)))             # electrofan + fuel cell
                    enrg = fuel*fhv
                elif power_system["energy_type"]==self.battery:
                    tas = self.get_tas(tamb,speed,speed_type)
                    enrg = start_mass*g*tas*time / (self.fan_eff*self.motor_eff*lod)        # electrofan + battery
                    fuel = 0.
            else:
                raise Exception("power system - thruster type is unknown")
        else:
            raise Exception("power system - engine type is unknown")

        return fuel,enrg


    def total_fuel(self,tow,distance,cruise_speed,speed_type,mtow,max_power,power_system,altitude_data,reserve_data):
        """Compute the total fuel required for a mission
        WARNING : when fuel is used, returned value is fuel mass (kg)
                  when battery is used, returned value is energy (J)
        """
        cruise_altp = altitude_data["mission"]
        mission_fuel,mission_enrg,mission_lod = self.leg_fuel(tow,distance,cruise_altp,cruise_speed,speed_type,mtow,max_power,power_system)
        if power_system["energy_type"]==self.battery:
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
            speed = 1. * cruise_speed
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

        mpax = self.get_pax_allowance(range)
        payload = npax * mpax

        dict = self.total_fuel(mtow, range, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)

        if power_system["energy_type"]==self.petrol:
            energy_storage_mass = 0.
        elif power_system["energy_type"]==self.gh2:
            h2_density = self.phd.fuel_density("compressed_h2", press=self.initial_gh2_pressure)
            self.gh2_tank_gravimetric_index = 1/(1+self.initial_gh2_pressure/(self.tank_efficiency_factor*h2_density))
            energy_storage_mass = dict["total_fuel"] * (1./self.gh2_tank_gravimetric_index - 1.)
        elif power_system["energy_type"]==self.lh2:
            energy_storage_mass = dict["total_fuel"] * (1./self.lh2_tank_gravimetric_index - 1.)
        elif power_system["energy_type"]==self.battery:
            energy_storage_mass = dict["total_enrg"]/self.battery_enrg_density
        else:
            raise Exception("power system - energy source is unknown")

        owe = mtow - payload - dict["total_fuel"]

        return {"owe":owe,
                "energy_storage_mass":energy_storage_mass,
                "total_energy":dict["total_enrg"],
                "total_fuel":dict["total_fuel"],
                "mpax":mpax,
                "payload":payload,
                "lod":dict["mission_lod"]}


    def owe_structure(self, npax, mtow, distance, total_power, energy_storage_mass, power_system):
        """Compute OWE from the point of view of structures
        """
        standard_mass = self.standard_mass(mtow)               # Standard MWE is MWE without furnishing
        furnishing = self.furnishing(npax)
        operator_items = self.op_item(npax, distance)
        propulsion_mass = self.propulsion_mass(power_system, total_power) + energy_storage_mass

        owe = standard_mass + propulsion_mass + furnishing + operator_items + self.delta_mwe

        return {"owe":owe,
                "furnishing":furnishing,
                "op_item":operator_items,
                "propulsion_mass":propulsion_mass}


    def design_airplane(self, npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system):

        def fct(mtow):
            total_power = self.ref_power(mtow)
            max_power = total_power / n_engine
            dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, power_system, altitude_data, reserve_data)
            dict_s = self.owe_structure(npax, mtow, distance, total_power, dict_p["energy_storage_mass"], power_system)
            return dict_p["owe"]-dict_s["owe"]

        mtow_ini = 1e-3 * npax * distance
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        mtow = output_dict[0][0]
        total_power = self.ref_power(mtow)
        max_power = total_power / n_engine
        dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, power_system, altitude_data, reserve_data)
        dict_s = self.owe_structure(npax, mtow, distance, total_power, dict_p["energy_storage_mass"], power_system)

        payload_max = dict_p["payload"] * self.max_payload_factor
        mzfw = dict_s["owe"] + payload_max
        mlw = mzfw * self.mlw_factor

        return {"airplane_type":altitude_data["airplane_type"],
                "npax":npax,
                "nominal_range":distance,
                "cruise_speed":cruise_speed,
                "altitude_data":altitude_data,
                "reserve_data":reserve_data,

                "n_engine":n_engine,
                "max_power":max_power,
                "total_power":total_power,
                "power_system":power_system,
                "propulsion_mass":dict_s["propulsion_mass"],
                "energy_storage_mass":dict_p["energy_storage_mass"],
                "system_energy_density":dict_p["total_energy"]/(dict_p["energy_storage_mass"]+dict_p["total_fuel"]),

                "mtow":mtow,
                "mlw":mlw,
                "mzfw":mzfw,
                "owe":dict_s["owe"],
                "furnishing":dict_s["furnishing"],
                "op_item":dict_s["op_item"],
                "mpax":dict_p["mpax"],
                "payload":dict_p["payload"],
                "payload_max":payload_max,
                "total_fuel":dict_p["total_fuel"],
                "total_energy":dict_p["total_energy"],

                "pk_o_mass_mini":distance/670,
                "pk_o_mass":npax*distance/mtow,
                "pk_o_enrg":npax*distance/dict_p["total_energy"]}


    def design_from_mtow(self, npax, mtow, cruise_speed, altitude_data, reserve_data, n_engine, power_system=None):

        def fct(distance):
            dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, power_system, altitude_data, reserve_data)
            dict_s = self.owe_structure(npax, mtow, distance, total_power, dict_p["energy_storage_mass"], power_system)
            return dict_p["owe"]-dict_s["owe"]

        total_power = self.ref_power(mtow)
        max_power = total_power / n_engine

        distance_ini = 40*mtow
        output_dict = fsolve(fct, x0=distance_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        distance = output_dict[0][0]
        dict_p = self.owe_performance(npax, mtow, distance, cruise_speed, max_power, power_system, altitude_data, reserve_data)
        dict_s = self.owe_structure(npax, mtow, distance, total_power, dict_p["energy_storage_mass"], power_system)

        payload_max = dict_p["payload"] * self.max_payload_factor
        mzfw = dict_s["owe"] + payload_max
        mlw = mzfw * self.mlw_factor

        return {"airplane_type":altitude_data["airplane_type"],
                "npax":npax,
                "nominal_range":distance,
                "cruise_speed":cruise_speed,
                "altitude_data":altitude_data,
                "reserve_data":reserve_data,

                "n_engine":n_engine,
                "max_power":max_power,
                "total_power":total_power,
                "power_system":power_system,
                "propulsion_mass":dict_s["propulsion_mass"],
                "energy_storage_mass":dict_p["energy_storage_mass"],
                "system_energy_density":dict_p["total_energy"]/(dict_p["energy_storage_mass"]+dict_p["total_fuel"]),

                "mtow":mtow,
                "mlw":mlw,
                "mzfw":mzfw,
                "owe":dict_s["owe"],
                "furnishing":dict_s["furnishing"],
                "op_item":dict_s["op_item"],
                "mpax":dict_p["mpax"],
                "payload":dict_p["payload"],
                "payload_max":payload_max,
                "total_fuel":dict_p["total_fuel"],
                "total_energy":dict_p["total_energy"],

                "pk_o_mass_mini":distance/670,
                "pk_o_mass":npax*distance/mtow,
                "pk_o_enrg":npax*distance/dict_p["total_energy"]}


    def get_best_range(self, range_init, range_step, criterion, npax, cruise_speed, altitude_data, reserve_data, n_engine, power_system):
        """Look for the design range that maximizes the criterion
        """
        def fct(distance):
            dict = self.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
            return dict[criterion]

        dx = range_step
        distance, res, rc = utils.maximize_1d(range_init,dx,[fct])

        dict = self.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
        return dict


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


    def print_design(self, dict, content="all"):

        if content=="all":
            print("")
            print("========================================================")
            print(" Airplane type = ", dict["airplane_type"])
            print(" Number of engine = ", "%.0f"%dict["n_engine"])
            print(" Number of passenger = ", "%.0f"%dict["npax"])
            print(" Design range = ", "%.0f"%unit.convert_to("km", dict["nominal_range"]), " km")
            print(" Cruise speed = ", "%.1f"%unit.convert_to("km/h", dict["cruise_speed"]), " km/h")
            print("")
            print("--------------------------------------------------------")
            print(" Max power = ", "%.0f"%unit.kW_W(dict["max_power"]), " kW")
            print("")
            print(" Engine type = ", dict["power_system"]["engine_type"])
            print(" Thruster type = ", dict["power_system"]["thruster_type"])
            print(" Energy source = ", dict["power_system"]["energy_type"])
            print(" Engine mass = ", "%.0f"%dict["propulsion_mass"])
            print("")
            print(" System energy density = ", "%.0f"%unit.Wh_J(dict["system_energy_density"]), " Wh/kg")
            print("")
            print(" mtow = ", "%.0f"%dict["mtow"], " kg")
            print(" owe = ", "%.0f"%dict["owe"], " kg")
            print(" payload = ", "%.0f"%dict["payload"], " kg")
            print(" payload_max = ", "%.0f"%dict["payload_max"], " kg")
            print(" furnishing = ", "%.0f"%dict["furnishing"], " kg")
            print(" operator items = ", "%.0f"%dict["op_item"], " kg")
            print(" Total fuel = ", "%.0f"%dict["total_fuel"], " kg")
            print(" Total energy = ", "%.0f"%unit.kWh_J(dict["total_energy"]), " kWh")
            print("")
            print(" Mass efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict["pk_o_mass"]), " pax.km/kg")
            print(" Energy efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")
            print("")
            print(" Commercial ratio (P/K maxi) / (P/K) = ", "%.2f"%(0.0146 / (dict["npax"]/unit.km_m(dict["nominal_range"]))))
            print(" Mass efficiency ratio, (P.K/M) / (P.K/M mini) = ", "%.2f"%(dict["pk_o_mass"]/dict["pk_o_mass_mini"]))

        elif content=="criteria":
            print("")
            print(" Design range = ", "%.0f"%unit.convert_to("km", dict["design_range"]), " km")
            print(" Mass efficiency factor, P.K/M = ", "%.2f"%unit.km_m(dict["pk_o_mass"]), " pax.km/kg")
            print(" Energy efficiency factor, P.K/E = ", "%.2f"%(unit.km_m(dict["pk_o_enrg"])/unit.kWh_J(1)), " pax.km/kWh")


    def fly_tow(self, ac_dict, tow, input, input_type="pax"):
        """Compute data from tow & payload
        """
        if input_type=="pax":
            payload = input*ac_dict["mpax"]
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
        power_system = ac_dict["power_system"]
        altitude_data = ac_dict["altitude_data"]
        reserve_data = ac_dict["reserve_data"]

        def fct(distance):
            dict = self.total_fuel(tow, distance, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)
            return tow - (owe + payload + dict["total_fuel"])

        dist_ini = 0.75*ac_dict["nominal_range"]
        output_dict = fsolve(fct, x0=dist_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        distance = output_dict[0][0]
        dict = self.total_fuel(tow, distance, cruise_speed, speed_type, mtow, max_power, power_system, altitude_data, reserve_data)
        dict["payload"] = payload
        return dict


    def fly_distance(self, ac_dict, distance, input, input_type="pax"):
        """Compute data from distance & payload
        """
        if input_type=="pax":
            payload = input*ac_dict["mpax"]
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
        power_system = ac_dict["power_system"]
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

        dict = self.fly_tow(ac_dict, mtow, ac_dict["payload_max"], input_type="mass")
        ac_dict["range_pl_max"] = dict["distance"]  # Range for maximum payload mission

        payload_max_fuel =  ac_dict["payload"] * self.max_fuel_payload_factor
        dict = self.fly_tow(ac_dict, mtow, payload_max_fuel, input_type="mass")
        ac_dict["payload_fuel_max"] = payload_max_fuel  # Payload for max fuel mission
        ac_dict["range_fuel_max"] = dict["distance"]    # Range for max fuel mission

        tow_zero_payload = mtow - payload_max_fuel
        dict = self.fly_tow(ac_dict, tow_zero_payload, 0., input_type="mass")
        ac_dict["range_no_pl"] = dict["distance"]       # Range for zero payload mission

        return ac_dict


    def draw_payload_range(self, ac_dict):
        """Print the payload - range diagram
        """
        plot_title = "Phantom Design"
        window_title = "Payload - Range"

        payload = [ac_dict["payload_max"],
                   ac_dict["payload_max"],
                   ac_dict["payload_fuel_max"],
                   0.]

        range = [0.,
                 unit.km_m(ac_dict["range_pl_max"]),
                 unit.km_m(ac_dict["range_fuel_max"]),
                 unit.km_m(ac_dict["range_no_pl"])]

        nominal = [ac_dict["payload"],
                   unit.km_m(ac_dict["nominal_range"])]

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(plot_title)
        fig.suptitle(window_title, fontsize=14)

        plt.plot(range,payload,linewidth=2,color="blue")
        plt.scatter(range[1:],payload[1:],marker="+",c="orange",s=100)
        plt.scatter(nominal[1],nominal[0],marker="o",c="green",s=50)

        plt.grid(True)

        plt.ylabel('Payload (kg)')
        plt.xlabel('Range (km)')

        plt.show()


    def is_in_plr(self, ac_dict, npax, distance):
        """Detects if a mission is possible
        """
        payload = npax * ac_dict["mpax"]
        out_dict = {"capa":True, "dist":True}

        c1 = ac_dict["payload_max"] - payload                                                                   # Max payload limit
        c2 =  (payload-ac_dict["payload_fuel_max"])*(ac_dict["range_pl_max"]-ac_dict["range_fuel_max"]) \
            - (ac_dict["payload_max"]-ac_dict["payload_fuel_max"])*(distance-ac_dict["range_fuel_max"])         # Max Take off weight limit
        c3 = payload*(ac_dict["range_fuel_max"]-ac_dict["range_no_pl"]) - ac_dict["payload_max"]*(distance-ac_dict["range_no_pl"])  # Max fuel limit
        c4 = ac_dict["range_no_pl"] - distance                                                                  # Max range limit

        if ((c1<0. or c2<0. or c3<0.) and c4>=0.):  # Out of PLR because of capacity
            out_dict["capa"] = False
        elif (c1>=0. and c4<0.):                    # Out of PLR because of range
            out_dict["dist"] = False
        elif (c1<0. and c4<0.):                     # Out of PLR because of range and capacity
            out_dict["capa"] = False
            out_dict["dist"] = False

        return out_dict


    def max_capacity(self, ac_dict, distance):
        """Retrieve the maximum capacity for a given range

        :param ac_dict: Airplane dictionary
        :param distance: Distance to fly
        :return:  capacity
        """
        if distance<=ac_dict["range_pl_max"]:
            capacity = np.floor(ac_dict["payload_max"]/ac_dict["mpax"])
        elif ac_dict["range_pl_max"]<distance and distance<=ac_dict["range_fuel_max"]:
            payload =    ac_dict["payload_fuel_max"] + (ac_dict["payload_max"]-ac_dict["payload_fuel_max"]) * (distance-ac_dict["range_fuel_max"]) / (ac_dict["range_pl_max"]-ac_dict["range_fuel_max"])
            capacity = np.floor(payload/ac_dict["mpax"])
        elif ac_dict["range_fuel_max"]<distance and distance<=ac_dict["range_no_pl"]:
            payload =   ac_dict["payload_fuel_max"]*(distance-ac_dict["range_no_pl"]) / (ac_dict["range_fuel_max"]-ac_dict["range_no_pl"])
            capacity = np.floor(payload/ac_dict["mpax"])
        else:
            capacity = 0.
        return capacity


    def max_range(self, ac_dict, npax):
        """Retrieve the maximum range for a given number of passenger

        :param ac_dict: Airplane dictionary
        :param npax: Number of passenger
        :return:  distance
        """
        payload = ac_dict["mpax"]*npax
        if ac_dict["payload_max"]<payload:
            distance = 0.
        elif ac_dict["payload_fuel_max"]<payload and payload<=ac_dict["payload_max"]:
            distance = ac_dict["range_fuel_max"] + (payload - ac_dict["payload_fuel_max"]) * (ac_dict["range_pl_max"]-ac_dict["range_fuel_max"]) / (ac_dict["payload_max"]-ac_dict["payload_fuel_max"])
        else:
            distance = ac_dict["range_no_pl"] + payload * (ac_dict["range_fuel_max"]-ac_dict["range_no_pl"]) / ac_dict["payload_fuel_max"]
        return distance


if __name__ == '__main__':

    phd = PhysicalData()
    ddm = DDM(phd)


    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "../../../data/All_Data_v5.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    un1 = un.copy()

    df1 = df1[df1['mtow']<100000].reset_index(drop=True)    # Remove all airplane with MTOW > 100t



    # # Regressions
    # #----------------------------------------------------------------------------------
    # abs = "mtow"
    # ord = "total_power"                           # Name of the new column
    #
    # df[ord] = df['max_power']*df['n_engine']      # Add the new column to the dataframe
    # un[ord] = un['max_power']                     # Add its unit
    #
    # # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    #
    # order = [2, 1]
    # dict = do_regression(df, un, abs, ord, coloration, order)




    # #-------------------------------------------------------------------------------------------------------------------
    # abs = "mtow"
    # ord = "owe"
    #
    # # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    # # df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t
    #
    # # order = [1]
    # order = [2, 1]
    # dict_owe = do_regression(df1, un1, abs, ord, coloration, order)



    #
    #-------------------------------------------------------------------------------------------------------------------
    # abs = "1e-2*weight/area"
    # ord = "10*power/weight"
    #
    # df1[abs] = (df1['mtow'] / df1['wing_area']).multiply(1e-2)
    # un1[abs] = "kg/m2"                    # Add its unit
    #
    # df1[ord] = ((df1['max_power']*df1['n_engine']) / df1['mtow']).multiply(10)
    # un1[ord] = "kN/kg"                    # Add its unit
    #
    # # dict = draw_reg(df1, un1, ord, abs, [[],[]], coloration, leg_loc="upper right")
    # #
    # order = [1, 0.25, 0]
    # dict_owe = do_regression(df1, un1, abs, ord, coloration, order)





    # MWE
    #-------------------------------------------------------------------------------------------------------------------
    # abs = "mtow"
    # ord = "MWE"                           # Name of the new column
    # ord1 = "mfurn+mopit"
    #
    # df1['m_furnishing'] = (df1['n_pax']**2).multiply(0.06) + df1['n_pax'].multiply(10)
    # df1['m_op_item'] = (df1['n_pax']*df1['nominal_range']).multiply(5.5e-6)
    #
    # # df1[ord1] = df1['mtow'].multiply(0.095) + (df1['mtow']**0.40).multiply(2.5)
    # # df1[ord] = df1['owe'] - df1[ord1]     # Add the new column to the dataframe
    #
    # # df1[ord] = df1['owe'] - df1['mtow'].multiply(0.0582)      # Raymer p591
    # # df1[ord] = df1['owe'] - df1['mtow'].multiply(0.1)      # Add the new column to the dataframe
    # df1[ord] = df1['owe'] - df1['m_furnishing'] - df1['m_op_item']      # Add the new column to the dataframe
    #
    # un1[ord] = un1['owe']                     # Add its unit
    #
    # # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    #
    # order = [2, 1]
    # dict = do_regression(df1, un1, abs, ord, coloration, order)
    #
    # # order = [0.5, 1]
    # # dict = do_regression(df1, un1, abs, ord1, coloration, order)





    # Build basic mass
    #-------------------------------------------------------------------------------------------------------------------
    # nap = df1.shape[0]
    #
    # thruster = {ddm.piston:ddm.propeller, ddm.turboprop:ddm.propeller, ddm.turbofan:ddm.fan}
    #
    # ord = "strw"
    #
    # df1[ord] = df1['owe']
    # un1[ord] = un1['owe']
    #
    # for n in range(nap):
    #     # print(df1["name"][n])
    #     npax = df1["n_pax"][n]
    #     distance = df1["nominal_range"][n]
    #     cruise_speed = df1["cruise_speed"][n]
    #     airplane_type = df1["airplane_type"][n]
    #     n_engine = df1["n_engine"][n]
    #     max_power = df1["max_power"][n]
    #     total_power = max_power*n_engine
    #     engine_type = df1["engine_type"][n]
    #     thruster_type = thruster[engine_type]
    #     mtow = df1["mtow"][n]
    #
    #     owe = df1["owe"][n]
    #
    #     tofl = df1["tofl"][n]
    #     vapp = df1["approach_speed"][n]
    #
    #     altitude_data = ddm.cruise_altp(airplane_type)
    #     reserve_data = ddm.reserve_data(airplane_type)
    #
    #     furnishing = ddm.furnishing(npax)
    #     operator_items = ddm.op_item(npax, distance)
    #
    #     power_system = {"energy_type":ddm.petrol, "engine_type":engine_type, "thruster_type":thruster_type}
    #     propulsion_mass = ddm.propulsion_mass(power_system, total_power)
    #
    #     df1[ord][n] = owe - propulsion_mass - furnishing - operator_items
    #
    # abs = "mtow"
    #
    # # dict = draw_reg(df1, un1, abs, ord, [[0,amp],[0,amp]], coloration)
    #
    # order = [2, 1]
    # dict_owe = do_regression(df1, un1, abs, ord, coloration, order)




    # Design Analysis
    #-------------------------------------------------------------------------------------------------------------------
    # nap = df1.shape[0]
    #
    # var = "mtow"
    # # var = "nominal_range"
    #
    # df1["guessed_"+var] = df1[var]
    # un1["guessed_"+var] = un1[var]
    #
    # thruster = {ddm.piston:ddm.propeller, ddm.turboprop:ddm.propeller, ddm.turbofan:ddm.fan}
    #
    # abs = var
    # ord = "guessed_"+var
    #
    # for n in range(nap):
    #     # print(df1["name"][n])
    #     npax = df1["n_pax"][n]
    #     distance = df1["nominal_range"][n]
    #     mtow = df1["mtow"][n]
    #     cruise_speed = df1["cruise_speed"][n]
    #     tofl = df1["tofl"][n]
    #     vapp = df1["approach_speed"][n]
    #     n_engine = df1["n_engine"][n]
    #     airplane_type = df1["airplane_type"][n]
    #     altitude_data = ddm.cruise_altp(airplane_type)
    #     reserve_data = ddm.reserve_data(airplane_type)
    #     power_system = {"thruster_type":thruster[df1["engine_type"][n]],
    #                     "engine_type":df1["engine_type"][n],
    #                     "energy_type":ddm.petrol}
    #
    #     if var=="mtow":
    #         ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
    #     elif var=="nominal_range":
    #         ac_dict = ddm.design_from_mtow(npax, mtow, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
    #
    #     df1[ord][n] = ac_dict[abs]
    #
    # amp = {"mtow":5e5, "nominal_range":20e6}.get(var)
    #
    # dict = draw_reg(df1, un1, abs, ord, [[0,amp],[0,amp]], coloration)
    # #
    # # order = [1]
    # # dict_owe = do_regression(df1, un1, abs, ord, coloration, order)







    # Airplane design analysis
    #-------------------------------------------------------------------------------------------------------------------



    # A320
    #-------------------------------------------------------------------------------------------------------------------
    # airplane_type = "narrow_body"
    # altitude_data = ddm.cruise_altp(airplane_type)
    # reserve_data = ddm.reserve_data(airplane_type)
    #
    # power_system = {"thruster_type":ddm.fan, "engine_type":ddm.turbofan, "energy_type":ddm.petrol}
    # n_engine = 2
    #
    # npax = 150
    # distance = unit.m_NM(3000)
    # cruise_speed = unit.mps_kmph(871)
    #
    # ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
    # ddm.print_design(ac_dict)




    # ATR72
    #-------------------------------------------------------------------------------------------------------------------
    airplane_type = "commuter"
    altitude_data = ddm.cruise_altp(airplane_type)
    reserve_data = ddm.reserve_data(airplane_type)

    power_system = {"thruster_type":ddm.propeller, "engine_type":ddm.turboprop, "energy_type":ddm.petrol}
    n_engine = 2

    npax = 70
    distance = unit.m_NM(600)
    cruise_speed = unit.mps_kmph(650)

    ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
    ddm.print_design(ac_dict)

    print(" Test missions")
    print("-----------------------------------------------------------------------")
    distance = unit.m_NM(350)
    input_type = "pax"
    input = 55

    tow_dict = ddm.fly_distance(ac_dict, distance, input, input_type)

    tow = tow_dict["tow"]

    dist_dict = ddm.fly_tow(ac_dict, tow, input, input_type)

    print("Initial distance = ", "%.2f"%unit.NM_m(distance), " NM")
    print("Mission take off weight = ", "%.1f"%tow, " kg")
    print("Mission fuel = ", "%.1f"%dist_dict["mission_fuel"], " kg")
    print("Reserve fuel = ", "%.1f"%dist_dict["reserve_fuel"], " kg")
    print("Recomputed distance = ", "%.2f"%unit.NM_m(dist_dict["distance"]), " NM")

    print(" Test payload-range")
    print("-----------------------------------------------------------------------")
    ac_dict = ddm.build_payload_range(ac_dict)
    ddm.draw_payload_range(ac_dict)

    print("-----------------------------------------------------------------------")
    npax = 78
    distance = unit.m_NM(200)

    print("capa = ", npax, "dist = ", "%.0f"%unit.km_m(distance))
    print(ddm.is_in_plr(ac_dict, npax, distance))

    print("-----------------------------------------------------------------------")
    npax = 78
    distance = unit.m_NM(400)

    print("capa = ", npax, "dist = ", "%.0f"%unit.km_m(distance))
    print(ddm.is_in_plr(ac_dict, npax, distance))
    print("max capa = ", "%.0f"%ddm.max_capacity(ac_dict, distance))

    print("-----------------------------------------------------------------------")
    npax = 60
    distance = unit.m_NM(1000)

    print("capa = ", npax, "dist = ", "%.0f"%unit.km_m(distance))
    print(ddm.is_in_plr(ac_dict, npax, distance))
    print("max capa = ", "%.0f"%ddm.max_capacity(ac_dict, distance))

    print("-----------------------------------------------------------------------")
    npax = 60
    distance = unit.m_NM(4000)

    print("capa = ", npax, "dist = ", "%.0f"%unit.km_m(distance))
    print(ddm.is_in_plr(ac_dict, npax, distance))
    print("max distance = ", "%.0f"%unit.km_m(ddm.max_range(ac_dict, npax)), " km")

    # TB20
    #-------------------------------------------------------------------------------------------------------------------
    # airplane_type = "general"
    # altitude_data = ddm.cruise_altp(airplane_type)
    # reserve_data = ddm.reserve_data(airplane_type)
    #
    # power_system = {"thruster_type":ddm.propeller, "engine_type":ddm.piston, "energy_type":ddm.petrol}
    # n_engine = 1
    #
    # npax = 4
    # distance = unit.m_km(1300)
    # cruise_speed = unit.mps_kmph(280)
    #
    # ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
    # ddm.print_design(ac_dict)



    # Bristell Energic
    # -------------------------------------------------------------------------------------------------------------------
    # airplane_type = "general"
    # altitude_data = ddm.cruise_altp(airplane_type)
    # reserve_data = ddm.reserve_data(airplane_type)
    #
    # power_system = {"thruster_type":ddm.propeller, "engine_type":ddm.emotor, "energy_type":ddm.battery}
    # n_engine = 1
    #
    # npax = 2
    # distance = unit.m_km(140)
    # cruise_speed = unit.mps_kmph(140)
    #
    # ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, power_system)
    # ddm.print_design(ac_dict)



    # Test
    #-------------------------------------------------------------------------------------------------------------------
    # airplane_type = "general"
    # altitude_data = ddm.cruise_altp(airplane_type)
    # reserve_data = ddm.reserve_data(airplane_type)
    #
    # initial_power_system = {"thruster_type":ddm.propeller, "engine_type":ddm.piston, "energy_type":ddm.petrol}
    # n_engine = 1
    #
    # npax = 4
    # distance = unit.m_km(200)
    # cruise_speed = unit.mps_kmph(180)
    #
    # tpws = [{"thruster_type":ddm.propeller, "engine_type":ddm.piston, "energy_type":ddm.petrol},
    #         {"thruster_type":ddm.propeller, "engine_type":ddm.piston, "energy_type":ddm.gh2},
    #         {"thruster_type":ddm.propeller, "engine_type":ddm.piston, "energy_type":ddm.lh2},
    #         {"thruster_type":ddm.propeller, "engine_type":ddm.emotor, "energy_type":ddm.battery},
    #         {"thruster_type":ddm.propeller, "engine_type":ddm.emotor, "energy_type":ddm.gh2},
    #         {"thruster_type":ddm.propeller, "engine_type":ddm.emotor, "energy_type":ddm.lh2}]
    #
    # for target_power_system in tpws:
    #     ac_dict = ddm.design_airplane(npax, distance, cruise_speed, altitude_data, reserve_data, n_engine, initial_power_system, target_power_system)
    #     ddm.print_design(ac_dict)











