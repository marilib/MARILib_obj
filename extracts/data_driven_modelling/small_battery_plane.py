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

    def __init__(self, npax=4, alt=unit.m_ft(3000), dist=unit.m_km(500), tas=unit.mps_kmph(180), mode="classic"):
        # Earth and atmosphere
        self.g = 9.80665
        self.disa = 0.

        # Top level requirements
        self.alt = alt
        self.vtas = tas
        self.distance = dist
        self.n_pax = npax
        self.mode = mode  # "classic" or "electric"

        # Additionnal informations
        self.m_pax = 90
        self.diversion_time = unit.s_min(30)
        self.lod = 14  # Aerodynamic efficiency

        # Propulsion
        self.prop_eff = 0.80
        self.motor_eff = 0.95  # MAGNIX

        # Energy storage
        self.psfc = unit.convert_from("lb/shp/h", 0.6)
        self.fuel_hv = unit.J_MJ(43)  # gasoline
        self.piston_eng_pw_density = unit.W_kW(1)  # W/kg
        self.elec_motor_pw_density = unit.W_kW(4.5)  # W/kg   MAGNIX
        self.power_elec_pw_density = unit.W_kW(10)  # W/kg
        self.battery_enrg_density = unit.J_Wh(200)  # Wh/kg

        # design results
        self.design = None

    def __str__(self):
        """Print main figures
        """
        s = ["\nAirplane : %s" %self.mode,
             "Npax     %.1f" % self.n_pax,
             "Distance %d km" % unit.km_m(self.distance),
             "TAS      %d km/h" % unit.kmph_mps(self.vtas),
             "Altitude %d ft" % unit.ft_m(self.alt),
             "---------------------------------"]

        if self.design==None:
            s.append(">>  NO DESIGN  <<")
            return "\n".join(s) # print only recquirements

        s.append("Max power = %.0f kW" %unit.kW_W(self.design["pw_max"]))
        if self.design["airplane_mode"]=="classic":
            s.append("Mission fuel = %.0f kg" %self.design["mission_fuel"])
            s.append("Reserve fuel = %.0f kg" %self.design["reserve_fuel"])
            s.append("Total fuel = %.0f kg" %self.design["total_fuel"])
        elif self.design["airplane_mode"]=="electric":
            s.append("Cruise power = %.0f kW" %unit.kW_W(self.design["pw_cruise"]))
            s.append("Battery mass = %.0f kg" %self.design["battery_mass"])
        else:
            raise Exception("Aircraft mode is unknown")
        s.append("Mission energy = %.0f kWh" %unit.kWh_J(self.design["mission_energy"]))
        s.append("Reserve energy = %.0f kWh" %unit.kWh_J(self.design["reserve_energy"]))
        s.append("Total energy = %.0f kWh" %unit.kWh_J(self.design["total_energy"]))
        s.append("MTOW = %.0f kg"%self.design["mtow"])
        s.append("OWE = %.0f kg"%self.design["owe"])
        s.append("Payload = %.0f kg"%self.design["payload"])
        s.append("")
        s.append("PK / MTOW minimum = %.2f pk/km" %self.design["pk_o_m_min"])
        s.append("PK / MTOW = %.2f pk/kg" %self.design["pk_o_m"])
        s.append("PK / Energy = %.2f pk/kWh"%self.design["pk_o_e"])
        return "\n|\t".join(s)

    def breguet(self, tow):
        """Used for classical airplane burning gasoline
        """
        fuel_mission = tow*(1.-np.exp(-(self.psfc*self.g*self.distance)/(self.prop_eff*self.lod)))   # piston engine
        fuel_reserve = (tow-fuel_mission)*(1.-np.exp(-(self.psfc*self.g*self.vtas*self.diversion_time)/(self.prop_eff*self.lod)))
        return fuel_mission, fuel_reserve

    def classic_design(self, mtow):
        """Aggregate all discipline to compute design point characteristics of a classicla airplane burning gasoline
        """
        pw_max = self.max_power(mtow)
        owe = self.basic_owe(mtow)
        fuel_mission, fuel_reserve = self.breguet(mtow)
        fuel_total = fuel_mission + fuel_reserve
        payload = self.n_pax * self.m_pax
        total_energy = fuel_total * self.fuel_hv
        return {"airplane_mode":self.mode,
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
                "pk_o_m_min":unit.km_m(self.distance)/670,
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
        # owe_basic = (-9.6325e-07 * mtow + 6.1041e-01) * mtow
        owe_basic = 0.606 * mtow
        return owe_basic

    def full_elec_design(self, mtow):
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
        return {"airplane_mode":self.mode,
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
                "pk_o_m_min":unit.km_m(self.distance)/670,
                "pk_o_m":self.n_pax*unit.km_m(self.distance)/mtow,
                "pk_o_e":self.n_pax*unit.km_m(self.distance)/unit.kWh_J(total_energy)}

    def design_solver(self):
        """Compute the design point
        """
        def fct(mtow):
            if self.mode=="classic":
                dict = self.classic_design(mtow)
                return mtow - dict["owe"] - dict["payload"] - dict["total_fuel"]
            elif self.mode=="electric":
                dict = self.full_elec_design(mtow)
                return mtow - dict["owe"] - dict["payload"]
            else:
                raise Exception("Aircraft mode is unknown")

        mtow_ini = 100000   # Use very high init to avoid negative root
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        mtow = output_dict[0][0]

        if self.mode=="classic":
            self.design = self.classic_design(mtow)
        elif self.mode=="electric":
            self.design = self.full_elec_design(mtow)


    def max_distance(self):
        """Compute the design that brings the minimum value for the PK/M criterion
        """
        def fct(dist):
            self.distance = dist
            dict = self.design_solver()
            return dict["pk_o_m"] - dict["pk_o_m_min"]

        dist_ini = self.distance * 2.
        output_dict = fsolve(fct, x0=dist_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.distance = output_dict[0][0]

        return self.design_solver()


    def compute_PKoM_on_grid(self, X, Y, **kwargs):
        """
        Compute the ratio between the Passenger.distance/MTOW (PKoM) and the minimum acceptable PKoM for a set of design range an number of passenger.
        :param X: 2D array of distance
        :param Y: 2D array of Npax
        :param kwargs: SmallPlane attribut that must be changed
        :return: 2D array of computed values
        """
        pkm=[]
        for key,val in kwargs.items(): # iterate over the kwargs list
            if key not in self.__dict__:
                raise KeyError('%s is not a SmallPlane attribut' %key)
            setattr(self,key,val) # change the attribut value. Raises a KeyError if invalid key is entered

        for x,y in zip(X.flatten(),Y.flatten()):
            self.distance = x
            self.n_pax = y
            self.design_solver()
            pkm.append(self.design["pk_o_m"]/self.design["pk_o_m_min"])
        # reshape pkm to 2D array
        pkm = np.array(pkm)
        return pkm.reshape(np.shape(X))


if __name__ == '__main__':

    #-------------------------------------------------------------------------------------------------------------------
    # Validation examples

    spc = SmallPlane(npax=4.5, dist=unit.m_km(1300), tas=unit.mps_kmph(280),mode="classic")     # TB20
    spc.design_solver()
    print(spc)

    # spc.max_distance(mode="classic")
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spc.distance), " km")


    spe = SmallPlane(npax=2, dist=unit.m_km(130), tas=unit.mps_kmph(130),mode="electric")       # H55
    spe.battery_enrg_density = unit.J_Wh(200)
    spe.design_solver()
    print(spe)

    # spe.max_distance(mode="electric")
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spe.distance), " km")






