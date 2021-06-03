#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
import unit, utils


from physical_data import PhysicalData

import matplotlib.pyplot as plt


class SmallPlane(object):

    def __init__(self, phd, npax=4, altp=unit.m_ft(3000), dist=unit.m_km(500), tas=unit.mps_kmph(180), mode="classic", fuel="gasoline"):
        # Earth and atmosphere
        self.phd = phd
        _,_,g = self.phd.atmosphere(0.,0.)
        self.g = g
        self.disa = 0.

        # Top level requirements
        self.altp = altp
        self.vtas = tas
        self.distance = dist
        self.n_pax = npax
        self.mode = mode  # "classic", "battery" or "fuel_cell"
        self.fuel = fuel  # "gasoline", "gh2" or "lh2"

        # Additionnal informations
        self.m_pax = 90
        self.diversion_time = unit.s_min(30)
        self.lod = 14           # Aerodynamic efficiency
        self.owe_factor = 1.    # Factor on OWE

        # Propulsion
        self.prop_efficiency = 0.80

        self.piston_eng_pw_density = unit.W_kW(1)  # W/kg
        self.psfc_gasoline = unit.convert_from("lb/shp/h", 0.6)
        
        self.elec_motor_pw_density = unit.W_kW(4.5) # W/kg   MAGNIX
        self.elec_motor_efficiency = 0.95           # MAGNIX

        # Energy systems
        self.cooling_pw_density = unit.W_kW(2)      # W/kg

        self.power_elec_pw_density = unit.W_kW(40)  # W/kg
        self.power_elec_efficiency = 0.99

        self.fc_system_pw_density = unit.W_kW(2)    # W/kg
        self.fuel_cell_pw_density = unit.W_kW(5)    # W/kg
        self.fuel_cell_efficiency = 0.50

        # Energy storage
        self.gasoline_lhv = unit.J_MJ(42)           # 42 MJ/kg
        self.hydrogen_lhv = unit.J_MJ(121)          # 121 MJ/kg
        self.lh2_tank_index = 0.2                   # Liquid H2 tank gravimetric index :     H2_mass / (H2_mass + Tank_mass)
        self.gh2_tank_index = 0.1                   # 700 bar H2 tank gravimetric index :     H2_mass / (H2_mass + Tank_mass)
        self.battery_enrg_density = unit.J_Wh(200)  # Wh/kg

        # design results
        self.design = None


    def __str__(self):
        """Print main figures
        """
        s = ["Airplane : %s" %self.mode,
             "Npax     %.1f" % self.n_pax,
             "Distance %d km" % unit.km_m(self.distance),
             "TAS      %d km/h" % unit.kmph_mps(self.vtas),
             "Altitude %d ft" % unit.ft_m(self.altp),
             "---------------------------------"]

        if self.design==None:
            s.append(">>  NO DESIGN  <<")
            return "\n".join(s) # print only recquirements

        if self.design["wing_area"] is not None:
            s.append("Wing area = %.2f m2" %self.design["wing_area"])
            s.append("Wing span = %.2f m" %self.design["wing_span"])
        s.append("Max power = %.0f kW" %unit.kW_W(self.design["pw_max"]))
        if self.design["airplane_mode"]=="classic":
            s.append("Special tank mass = %.0f kg" %self.design["special_tank"])
            s.append("Total fuel = %.0f kg" %self.design["fuel_total"])
        elif self.design["airplane_mode"]=="battery":
            s.append("Cruise power = %.0f kW" %unit.kW_W(self.design["pw_cruise"]))
            s.append("Cruise thrust = %.0f N" %self.design["fn_cruise"])
            s.append("Battery mass = %.0f kg" %self.design["battery_mass"])
            s.append("system energy density = %.2f Wh/kg" %unit.Wh_J(self.design["system_enrg_density"]))
        elif self.design["airplane_mode"]=="fuel_cell":
            s.append("Cruise power = %.0f kW" %unit.kW_W(self.design["pw_cruise"]))
            s.append("Cruise thrust = %.0f N" %self.design["fn_cruise"])
            s.append("System mass = %.0f kg" %self.design["system_mass"])
            s.append("system energy density = %.2f Wh/kg" %unit.Wh_J(self.design["system_enrg_density"]))
            s.append("Total fuel = %.0f kg" %self.design["fuel_total"])
        else:
            raise Exception("Aircraft mode is unknown")
        s.append("Total energy = %.0f kWh" %unit.kWh_J(self.design["total_energy"]))
        s.append("MTOW = %.0f kg"%self.design["mtow"])
        s.append("OWE = %.0f kg"%self.design["owe"])
        s.append("Payload = %.0f kg"%self.design["payload"])
        s.append("")
        s.append("PK / MTOW minimum = %.2f pk/kg" %self.design["pk_o_m_min"])
        s.append("PK / MTOW = %.2f pk/kg" %self.design["pk_o_m"])
        s.append("PK / Energy = %.2f pk/kWh"%self.design["pk_o_e"])
        s.append("")
        return "\n|\t".join(s)


    def set_aero_data(self, area, span, stall_speed):
        if self.design==None:
            raise Exception("No design available, Wing data cannot be loaded")
        self.design["wing_area"] = area
        self.design["wing_span"] = span
        self.design["stall_speed"] = stall_speed


    def get_aero_model(self, full_output=False):
        if self.design==None:
            raise Exception("No design available, Polar cannot be assessed")

        if self.design["wing_area"]==None:
            raise Exception("No wing area available, set wing area before asking for polar")

        if self.mode in ["classic", "fuel_cell"]:
            mass = self.design["mtow"] - 0.5*self.design["fuel_mission"]
        elif self.mode=="battery":
            mass = self.design["mtow"]
        else:
            raise Exception("Aircraft mode is unknown")

        pamb,tamb,g = self.phd.atmosphere(self.altp, self.disa)
        rho = self.phd.gas_density(pamb,tamb)
        vtas = self.vtas
        ar = self.design["wing_span"]**2 / self.design["wing_area"]
        cz = (mass*g) / (0.5*rho*self.design["wing_area"]*vtas**2)
        ki = 1.1 / (np.pi*ar)
        cx0 = cz/self.lod - ki*cz**2
        cza = (np.pi*ar) / (1+np.sqrt(1+(ar/2)**2))
        rho0 = 1.225
        czmax = (self.design["mtow"]*g) / (0.5*rho0*self.design["wing_area"]*self.design["stall_speed"]**2)

        area, span = [self.design["wing_area"], self.design["wing_span"]]
        wa,ha,va,fa = [area, 0.253*area-0.79, 0.213*area-1.34, 1.248*area-8.33]
        wrc,hrc,vrc,frl = [area/span, np.sqrt(ha/5), np.sqrt(va/1.7), 1.296*span -5.36]
        wwa,hwa,vwa,fwa = [1.7*wa, 1.9*ha, 2.0*va, 2.7*fa]
        lref = (wrc*wwa+hrc*hwa+vrc*vwa+frl*fwa) / (wwa+hwa+vwa+fwa)

        re = self.phd.reynolds_number(pamb,tamb,vtas)
        kre = (1/np.log(re*lref))**2.58

        self.design["aero"] = {"cx0":cx0, "ki":ki, "cza":cza, "czmax":czmax, "kre":kre, "lref":lref}

        if full_output:
            print("Aerodynamic data")
            print("|   cx0   ", "%.4f"%cx0)
            print("|   ki    ", "%.3f"%ki)
            print("|   cza   ", "%.2f"%cza)
            print("|   czmax ", "%.2f"%czmax)
            print("|     kre ", "%.6f"%(kre*1000), " kre*1e3")
            print("|    lref ", "%.1f"%lref, " m")
            cz_list = np.linspace(0., 1.5, 50)
            lod_list = [cz/(cx0+ki*cz**2) for cz in cz_list]
            plt.plot(cz_list, lod_list)
            plt.scatter(cz, cz/(cx0+ki*cz**2), marker="o",c="green",s=50)
            plt.grid(True)
            plt.show()

    def get_thrust(self, mass, altp, disa, vtas):
        pamb,tamb,g = self.phd.atmosphere(altp, disa)
        rho = self.phd.gas_density(pamb,tamb)
        area = self.design["wing_area"]
        czmax = self.design["aero"]["czmax"]
        cx0 = self.design["aero"]["cx0"]
        ki = self.design["aero"]["ki"]
        q = 0.5*rho*vtas**2
        vs1g = np.sqrt((2*mass*g) / (rho*area*czmax))
        kvs1g = vtas / vs1g
        fn = q*area*cx0 + ki*(mass*g)**2 / (q*area)

        fnmax = 0.8 * self.design["pw_max"] * self.prop_efficiency / (0.7*1.13*vs1g)
        cz = czmax / 1.13**2
        mlfac = mass**2 / (fnmax*cz*area*(rho/1.225)**0.8)
        tofl = 12*mlfac + 300

        return fn, kvs1g, tofl

    def get_criteria(self, pod_mass_init, pod_mass, vtas, pw):
        mtow = self.design["mtow"]
        ft =  (self.design["battery_mass"] + pod_mass_init - pod_mass) \
            * self.battery_enrg_density \
            / pw
        fd = ft * vtas

    def get_tank_index(self):
        if self.fuel=="gasoline":
            return 1.
        elif self.fuel=="gh2":
            return self.gh2_tank_index
        elif self.fuel=="lh2":
            return self.lh2_tank_index
        else:
            raise Exception("Fuel type is unknown")


    def get_psfc(self):
        if self.fuel=="gasoline":
            psfc = self.psfc_gasoline
        elif self.fuel in ["gh2", "lh2"]:
            psfc = self.psfc_gasoline * self.gasoline_lhv / self.hydrogen_lhv
        else:
            raise Exception("Fuel type is unknown")
        return psfc


    def max_power(self, mtow):
        """Estimate max installed power
        """
        pw_max = (0.0197*mtow + 100.6)*mtow
        return pw_max


    def basic_owe(self, mtow):
        """Estimate classical airplane empty weight
        """
        # owe_basic = (-9.6325e-07 * mtow + 6.1041e-01) * mtow
        owe_basic = 0.606 * mtow * self.owe_factor
        return owe_basic


    def aerodynamic(self, mass):
        """Compute required thrust at cruise point
        """
        fn = mass * self.g / self.lod
        return fn


    def elec_propulsion(self, pw_max, fn):
        """Copute required power at cruise point and engine mass
        """
        pw = fn * self.vtas / (self.prop_efficiency*self.elec_motor_efficiency)
        m_engine = pw_max / self.elec_motor_pw_density
        return pw, m_engine


    def breguet(self, tow):
        """Used for classical airplane burning gasoline
        """
        psfc = self.get_psfc()
        fuel_mission = tow*(1.-np.exp(-(psfc*self.g*self.distance)/(self.prop_efficiency*self.lod)))   # piston engine
        fuel_reserve = (tow-fuel_mission)*(1.-np.exp(-(psfc*self.g*self.vtas*self.diversion_time)/(self.prop_efficiency*self.lod)))
        return fuel_mission, fuel_reserve


    def classic_design(self, mtow):
        """Aggregate all discipline to compute design point characteristics of a classicla airplane burning gasoline
        """
        pw_max = self.max_power(mtow)
        fuel_mission, fuel_reserve = self.breguet(mtow)
        fuel_total = fuel_mission + fuel_reserve
        special_tank = fuel_total * (1/self.get_tank_index()-1)
        owe =  self.basic_owe(mtow) + special_tank
        payload = self.n_pax * self.m_pax
        total_energy = fuel_total * self.gasoline_lhv
        return {"airplane_mode":self.mode,
                "wing_area":None,
                "wing_span":None,
                "pw_max":pw_max,
                "special_tank":special_tank,
                "fuel_mission":fuel_mission,
                "fuel_total":fuel_total,
                "total_energy":total_energy,
                "mtow":mtow,
                "owe":owe,
                "payload":payload,
                "pk_o_m_min":unit.km_m(self.distance)/670,
                "pk_o_m":self.n_pax*unit.km_m(self.distance)/mtow,
                "pk_o_e":self.n_pax*unit.km_m(self.distance)/unit.kWh_J(total_energy)}


    def battery_energy_system(self, pw_max, pw, mass):
        """Compute added system and battery masses
        """
        elec_pw_max = pw_max / (self.prop_efficiency*self.elec_motor_efficiency)
        m_system = elec_pw_max / self.power_elec_pw_density
        m_battery = (  pw * (self.distance/self.vtas)
                     + mass*self.g*self.altp/(self.elec_motor_efficiency*self.prop_efficiency)
                    ) / self.battery_enrg_density
        m_battery += (pw * self.diversion_time) / self.battery_enrg_density
        return m_system, m_battery


    def battery_elec_design(self, mtow):
        """Aggregate all discipline outputs to compute design point characteristics of a full electric airplane
        """
        pw_max = self.max_power(mtow)
        owe_basic = self.basic_owe(mtow)
        fn = self.aerodynamic(mtow)
        pw, m_engine = self.elec_propulsion(pw_max,fn)
        m_system, m_battery = self.battery_energy_system(pw_max, pw, mtow)
        owe =  owe_basic \
             - pw_max / self.piston_eng_pw_density \
             + m_engine + m_system + m_battery
        payload = self.n_pax * self.m_pax
        total_energy = m_battery * self.battery_enrg_density
        return {"airplane_mode":self.mode,
                "wing_area":None,
                "wing_span":None,
                "pw_max":pw_max,
                "pw_cruise":pw,
                "fn_cruise":fn,
                "battery_mass":m_battery,
                "system_enrg_density":total_energy/(m_system+m_battery),
                "fuel_mission":0.,
                "fuel_total":0.,
                "total_energy":total_energy,
                "mtow":mtow,
                "owe":owe,
                "payload":payload,
                "pk_o_m_min":unit.km_m(self.distance)/670,
                "pk_o_m":self.n_pax*unit.km_m(self.distance)/mtow,
                "pk_o_e":self.n_pax*unit.km_m(self.distance)/unit.kWh_J(total_energy)}


    def fuel_cell_energy_system(self, pw_max, mass):
        """Compute added system and battery masses
        """
        elec_pw_max = pw_max / (self.prop_efficiency*self.elec_motor_efficiency)
        m_system = elec_pw_max / self.power_elec_pw_density
        m_system += elec_pw_max / self.fuel_cell_pw_density
        m_system += elec_pw_max / self.fc_system_pw_density
        thermal_pw = elec_pw_max * (1 - self.fuel_cell_efficiency) \
                                 / (self.fuel_cell_efficiency*self.elec_motor_efficiency*self.power_elec_efficiency)
        m_system += thermal_pw / self.cooling_pw_density

        fhv = self.hydrogen_lhv
        eff = self.prop_efficiency * self.elec_motor_efficiency * self.fuel_cell_efficiency
        fuel_mission = mass*(1.-np.exp(-(self.g*self.distance)/(eff*fhv*self.lod)))
        fuel_reserve = (mass-fuel_mission)*(1-np.exp(-(self.g*self.vtas*self.diversion_time)/(eff*fhv*self.lod)))

        m_system += (fuel_mission + fuel_reserve) * (1/self.get_tank_index() - 1)
        return m_system, fuel_mission, fuel_reserve


    def fuel_cell_elec_design(self, mtow):
        """Aggregate all discipline outputs to compute design point characteristics of a full electric airplane
        """
        pw_max = self.max_power(mtow)
        owe_basic = self.basic_owe(mtow)
        fn = self.aerodynamic(mtow)
        pw, m_engine = self.elec_propulsion(pw_max,fn)
        m_system, fuel_mission, fuel_reserve = self.fuel_cell_energy_system(pw_max, mtow)
        fuel_total = fuel_mission + fuel_reserve
        owe =  owe_basic \
             - pw_max / self.piston_eng_pw_density \
             + m_engine + m_system + fuel_total
        payload = self.n_pax * self.m_pax
        total_energy = fuel_total * self.hydrogen_lhv
        return {"airplane_mode":self.mode,
                "wing_area":None,
                "wing_span":None,
                "pw_max":pw_max,
                "pw_cruise":pw,
                "fn_cruise":fn,
                "system_mass":m_system,
                "system_enrg_density":total_energy/m_system,
                "total_energy":total_energy,
                "fuel_mission":fuel_mission,
                "fuel_total":fuel_total,
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
            elif self.mode=="battery":
                dict = self.battery_elec_design(mtow)
            elif self.mode=="fuel_cell":
                dict = self.fuel_cell_elec_design(mtow)
            else:
                raise Exception("Aircraft mode is unknown")
            return mtow - dict["owe"] - dict["payload"] - dict["fuel_total"]

        mtow_ini = 5000   # Use very high init to avoid negative root
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1):
            xres, yres, rc = utils.maximize_1d(200, 200, [fct])
            if yres<0: return
        elif output_dict[0][0]<0:
            return
        mtow = output_dict[0][0]

        if self.mode=="classic":
            self.design = self.classic_design(mtow)
        elif self.mode=="battery":
            self.design = self.battery_elec_design(mtow)
        elif self.mode=="fuel_cell":
            self.design = self.fuel_cell_elec_design(mtow)


    def test_convergence(self):
        mtow = np.linspace(200, 10000, 50)
        diff = []
        for m in mtow:
            if self.mode=="classic":
                dict = self.classic_design(m)
            elif self.mode=="battery":
                dict = self.battery_elec_design(m)
            elif self.mode=="fuel_cell":
                dict = self.fuel_cell_elec_design(m)
            else:
                raise Exception("Aircraft mode is unknown")
            diff.append(m - dict['owe'] - dict['payload'] - dict['fuel_total'])
        plt.plot(mtow,diff)
        plt.grid(True)
        plt.show()


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

    phd = PhysicalData()

    #-------------------------------------------------------------------------------------------------------------------
    # Validation examples

    # print("---------------------------------------------------")
    # print("SOCATA TB20 TRINIDAD")
    # print("---------------------------------------------------")
    # spc = SmallPlane(phd, npax=4.5, dist=unit.m_km(1300), tas=unit.mps_kmph(280), mode="classic", fuel="gasoline")     # TB20
    # spc.design_solver()
    # print(spc)

    # spc.max_distance(mode="classic")
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spc.distance), " km")


    # print("---------------------------------------------------")
    # print("BRISTEL H55 ENERGIC")
    # print("---------------------------------------------------")
    # spe = SmallPlane(phd, npax=2, dist=unit.m_km(130), altp=unit.m_ft(1000), tas=unit.mps_kmph(130), mode="battery")       # H55
    # spe.battery_enrg_density = unit.J_Wh(200)
    # spe.lod = 13.2
    # spe.design_solver()
    # print(spe)
    #
    # wing_area = 11.75
    # wing_span = 9.27
    # spe.set_aero_data(wing_area, wing_span, unit.mps_kmph(87))
    # spe.get_aero_model(full_output=True)
    #
    # mass = spe.design["mtow"]
    # altp = 0
    # disa = 0
    # vtas = unit.mps_kmph(130)
    #
    # fn,kvs1g,tofl = spe.get_thrust(mass, altp, disa, vtas)
    #
    # print("--------------------------------")
    # print("fn = ", fn)
    # print("kvs1g = ", kvs1g)
    # print("tofl = ", tofl)
    #
    # spe.max_distance(mode="battery")
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spe.distance), " km")


    # print("---------------------------------------------------")
    # print("EVIATION ALICE (constructeur)")
    # print("---------------------------------------------------")
    # spe = SmallPlane(phd, npax=11, altp=unit.m_ft(10000), dist=unit.m_km(800), tas=unit.mps_kmph(440), mode="battery")       # H55
    # spe.battery_enrg_density = unit.J_Wh(240)   # Outstanding capacity for state of the art Li-ion
    # spe.prop_efficiency = 0.82                  # Very good propeller efficiency
    # spe.owe_factor = 0.75                       # 25% airframe weight improvement
    # spe.lod = 29                                # Glider level aerodynamic efficiency
    # spe.design_solver()
    # print(spe)
    #
    #
    # print("---------------------------------------------------")
    # print("EVIATION ALICE (most probable)")
    # print("---------------------------------------------------")
    # spe = SmallPlane(phd, npax=11, altp=unit.m_ft(10000), dist=unit.m_km(500), tas=unit.mps_kmph(440), mode="battery")       # H55
    # spe.battery_enrg_density = unit.J_Wh(200)   # Outstanding capacity for state of the art Li-ion
    # spe.prop_efficiency = 0.82                  # Very good propeller efficiency
    # spe.owe_factor = 0.75                       # 25% airframe weight improvement
    # spe.lod = 25                                # Glider level aerodynamic efficiency
    # spe.design_solver()
    # print(spe)


    #-------------------------------------------------------------------------------------------------------------------
    # sph = SmallPlane(phd, npax=4, dist=unit.m_km(600), tas=unit.mps_kmph(250), mode="fuel_cell", fuel="gh2")       # H55
    # sph.cooling_pw_density = unit.W_kW(2)      # W/kg
    # sph.fc_system_pw_density = unit.W_kW(2)
    # sph.fuel_cell_efficiency = 0.5
    # # sph.test_convergence()
    # sph.design_solver()
    # print(sph)

    # spe.max_distance(mode="battery")
    # print("")
    # print("Max distance vs PK/M = ", "%.0f"%unit.km_m(spe.distance), " km")






