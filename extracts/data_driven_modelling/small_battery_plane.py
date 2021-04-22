#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve

import unit



class SmallPlane(object):

    def __init__(self, npax=4, alt=unit.m_ft(3000), dist=unit.m_km(200), tas=unit.mps_kmph(150)):
        self.g = 9.80665

        self.disa = 0.

        self.alt = alt
        self.vtas = tas
        self.distance = dist

        self.n_pax = npax
        self.m_pax = 90
        self.diversion_time = unit.s_min(30)

        self.lod = 14   # Aerodynamic efficiency

        self.prop_eff = 0.80
        self.motor_eff = 0.95       # MAGNIX

        self.psfc = unit.convert_from("lb/shp/h",0.6)
        self.fuel_hv = unit.J_MJ(43)    # gasoline

        self.piston_eng_pw_density = unit.W_kW(1)   # W/kg
        self.elec_motor_pw_density = unit.W_kW(4.5) # W/kg   MAGNIX
        self.power_elec_pw_density = unit.W_kW(10)  # W/kg
        self.battery_enrg_density = unit.J_Wh(400)  # Wh/kg

    def breguet(self, tow):
        """Used for classical airplane burning gasoline
        """
        fuel_mission = tow*(1.-np.exp(-(self.psfc*self.g*self.distance)/(self.prop_eff*self.lod)))   # piston engine
        fuel_reserve = (tow-fuel_mission)*(1.-np.exp(-(self.psfc*self.g*self.vtas*self.diversion_time)/(self.prop_eff*self.lod)))
        return fuel_mission, fuel_reserve

    def classic_design(self, mtow, type):
        """Aggregate all discipline to compute design point characteristics of a classicla airplane burning gasoline
        """
        pw_max = self.max_power(mtow)
        owe = self.basic_owe(mtow)
        fuel_mission, fuel_reserve = self.breguet(mtow)
        fuel_total = fuel_mission + fuel_reserve
        payload = self.n_pax * self.m_pax
        total_energy = fuel_total * self.fuel_hv
        return {"airplane_type":type,
                "pw_max":pw_max,
                "mission_fuel":fuel_mission,
                "reserve_fuel":fuel_reserve,
                "total_fuel":fuel_total,
                "mission_energy":fuel_mission * self.fuel_hv,
                "reserve_energy":fuel_reserve*self.fuel_hv,
                "total_energy":total_energy,
                "mtow":mtow,
                "owe":owe,
                "payload":payload,
                "pk_o_m_min":unit.km_m(self.distance)/500,
                "pk_o_m":self.n_pax*unit.km_m(self.distance)/mtow,
                "pk_o_e":self.n_pax*unit.km_m(self.distance)/unit.kWh_J(total_energy)}

    def aerodynamic(self, mass):
        """Compute required thrust at cruise point
        """
        fn = mass * self.g / self.lod
        return fn

    def propulsion(self, pw_max, fn):
        """Copute required power at cruise point and engine mass
        """
        pw = fn * self.vtas / (self.prop_eff*self.motor_eff)
        m_engine = pw_max / self.elec_motor_pw_density
        return pw, m_engine

    def energy_system(self, pw_max, pw, mass):
        """Compute added system and battery masses
        """
        m_system = pw_max / self.power_elec_pw_density
        m_battery = (  pw * (self.distance/self.vtas)
                     + mass*self.g*self.alt/(self.motor_eff*self.prop_eff)
                    ) / self.battery_enrg_density
        return m_system, m_battery

    def regulation(self, pw):
        """Compute additional battery mass for reserve
        """
        m_reserve = (pw * self.diversion_time) / self.battery_enrg_density
        return m_reserve

    def max_power(self, mtow):
        """Estimate max installed power
        """
        pw_max = (0.0197*mtow + 100.6)*mtow
        return pw_max

    def basic_owe(self, mtow):
        """Estimate classical airplane empty weight
        """
        owe_basic = (-9.6325e-07 * mtow + 6.1041e-01) * mtow
        # owe_basic = 0.606 * mtow
        return owe_basic

    def full_elec_design(self, mtow, type):
        """Aggregate all discipline outputs to compute design point characteristics of a full electric airplane
        """
        pw_max = self.max_power(mtow)
        owe_basic = self.basic_owe(mtow)
        fn = self.aerodynamic(mtow)
        pw, m_engine = self.propulsion(pw_max,fn)
        m_system, m_battery = self.energy_system(pw_max, pw, mtow)
        m_reserve = self.regulation(pw)
        owe =  owe_basic \
             - pw_max / self.piston_eng_pw_density \
             + m_engine + m_system + m_battery + m_reserve
        payload = self.n_pax * self.m_pax
        total_energy = (m_battery + m_reserve) * self.battery_enrg_density
        return {"airplane_type":type,
                "pw_max":pw_max,
                "pw_cruise":pw,
                "fn_cruise":fn,
                "battery_mass":m_battery+m_reserve,
                "mission_energy":m_battery*self.battery_enrg_density,
                "reserve_energy":m_reserve*self.battery_enrg_density,
                "total_energy":total_energy,
                "mtow":mtow,
                "owe":owe,
                "payload":payload,
                "pk_o_m_min":unit.km_m(self.distance)/500,
                "pk_o_m":self.n_pax*unit.km_m(self.distance)/mtow,
                "pk_o_e":self.n_pax*unit.km_m(self.distance)/unit.kWh_J(total_energy)}

    def design_solver(self, type="classic"):
        """Compute the design point
        """
        def fct(mtow):
            if type=="classic":
                dict = self.classic_design(mtow, type)
                return mtow - dict["owe"] - dict["payload"] - dict["total_fuel"]
            elif type=="electric":
                dict = self.full_elec_design(mtow, type)
                return mtow - dict["owe"] - dict["payload"]
            else:
                raise Exception("Aircraft type is unknown")

        mtow_ini = 2000
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        mtow = output_dict[0][0]

        if type=="classic":
            return self.classic_design(mtow, type)
        elif type=="electric":
            return self.full_elec_design(mtow, type)

    def max_distance(self, type="classic"):
        """Compute the design that brings the minimum value for the PK/M criterion
        """
        def fct(dist):
            self.distance = dist
            dict = self.design_solver(type)
            return dict["pk_o_m"] - dict["pk_o_m_min"]

        dist_ini = self.distance * 2.
        output_dict = fsolve(fct, x0=dist_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.distance = output_dict[0][0]

        return self.design_solver(type)


    def print(self, dict):
        """Print main figures
        """
        print("")
        print("Airplane type = ", dict["airplane_type"])
        print("-----------------------------------------------")
        print("Max power = ", "%.0f"%unit.kW_W(dict["pw_max"]), " kW")

        if dict["airplane_type"]=="classic":
            print("Mission fuel = ", "%.0f"%dict["mission_fuel"], " kg")
            print("Reserve fuel = ", "%.0f"%dict["reserve_fuel"], " kg")
            print("Total fuel = ", "%.0f"%dict["total_fuel"], " kg")
        elif dict["airplane_type"]=="electric":
            print("Cruise power = ", "%.0f"%unit.kW_W(dict["pw_cruise"]), " kW")
            print("Battery mass = ", "%.0f"%dict["battery_mass"], " kg")
        else:
            raise Exception("Aircraft type is unknown")

        print("Mission energy = ", "%.0f"%unit.kWh_J(dict["mission_energy"]), " kWh")
        print("Reserve energy = ", "%.0f"%unit.kWh_J(dict["reserve_energy"]), " kWh")
        print("Total energy = ", "%.0f"%unit.kWh_J(dict["total_energy"]), " kWh")
        print("MTOW = ", "%.0f"%dict["mtow"], " kg")
        print("OWE = ", "%.0f"%dict["owe"], " kg")
        print("Payload = ", "%.0f"%dict["payload"], " kg")
        print("")
        print("PK / MTOW minimum = ", "%.2f"%dict["pk_o_m_min"], " pk/kg")
        print("PK / MTOW = ", "%.2f"%dict["pk_o_m"], " pk/kg")
        print("PK / Energy = ", "%.2f"%dict["pk_o_e"], " pk/kWh")



if __name__ == '__main__':

    npax = 9
    dist = 200
    vtas = 200

    spc = SmallPlane(npax=npax, dist=unit.m_km(dist), tas=unit.mps_kmph(vtas))

    spc_dict = spc.design_solver("classic")

    spc.print(spc_dict)

    # spc.max_distance(type="classic")
    #
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spc.distance), " km")



    spe = SmallPlane(npax=npax, dist=unit.m_km(dist), tas=unit.mps_kmph(vtas))

    spe.battery_enrg_density = unit.J_Wh(200)

    spe_dict = spe.design_solver("electric")

    spe.print(spe_dict)

    # spe.max_distance(type="electric")
    #
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spe.distance), " km")

    print("")
    print("Criteria = ", "%.3f"%(spe_dict["pk_o_m"]/spc_dict["pk_o_m"]))






