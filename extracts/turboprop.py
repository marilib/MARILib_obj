#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

import unit

from physical_data import PhysicalData



class TurboPropNacelle(object):

    def __init__(self, phd):
        self.phd = phd

        self.rating_factor = {"MTO":1.00, "MCN":0.95, "MCL":0.90, "MCR":0.70, "FID":0.05}
        self.nacelle_pw_density = unit.W_kW(10.)    # W/kg
        self.form_factor = 1.15
        self.propeller_efficiency = 0.82
        self.propeller_disk_load = 1000.    # N/m2
        self.psfc_reference = unit. kgpWps_lbpshpph(0.9)    # kerosene
        self.fuel_type = "kerosene"     # Can be kerosene or liquid_h2 or compressed_h2

        self.psfc_effective = None
        self.reference_power = None
        self.propeller_width = None
        self.width = None
        self.length = None

        self.gross_wet_area = None
        self.net_wet_area = None
        self.aero_length = None

        self.propeller_mass = 0.
        self.engine_mass = 0.
        self.pylon_mass = 0.
        self.mass = None

    def design(self, reference_power, reference_speed):
        self.reference_power = reference_power
        self.reference_speed = reference_speed
        self.psfc_effective = self.psfc_reference * self.phd.fuel_heat("kerosene") / self.phd.fuel_heat(self.fuel_type)

        # info : reference_thrust is defined by thrust(mach=0.25, altp=0, disa=15) / 0.80
        disa = 15.
        altp = 0.

        pamb,tamb,g = self.phd.atmosphere(altp, disa)
        rating = "MTO"
        throttle = 1.

        dict = self.operate_from_throttle(pamb,tamb,reference_speed,rating,throttle)
        self.reference_thrust = dict["thrust"]

        self.propeller_width = np.sqrt((4./np.pi)*(self.reference_thrust/self.propeller_disk_load))      # Assuming 3000 N/m2

        self.width = 0.25*(reference_power/1.e3)**0.2        # statistical regression
        self.length = 0.84*(reference_power/1.e3)**0.2       # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)   # statistical regression
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length

        self.engine_mass = (0.633*(reference_power/1.e3)**0.9)  # statistical regression
        self.nacelle_mass = (1./self.nacelle_pw_density) * reference_power
        self.propeller_mass = (165./1.5e6)*reference_power

        self.mass = self.nacelle_mass + self.engine_mass + self.propeller_mass

    def operate_from_throttle(self,pamb,tamb,vair,rating,throttle):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        reference_power = self.reference_power
        factor = self.rating_factor[rating]
        eta_prop = self.propeller_efficiency

        rho = self.phd.gas_density(pamb,tamb)
        sig = rho / 1.225
        shaft_power = throttle*factor*reference_power*sig**0.5
        thrust = eta_prop*shaft_power/vair
        psfc = self.psfc_effective

        return {"throttle":throttle, "thrust":thrust, "power":shaft_power, "psfc":psfc}

    def operate_from_thrust(self,pamb,tamb,vair,rating,thrust):
        """Unitary thrust of a pure turboprop engine (semi-empirical model)
        """
        reference_power = self.reference_power
        factor = self.rating_factor[rating]
        eta_prop = self.propeller_efficiency

        rho = self.phd.gas_density(pamb,tamb)
        sig = rho / 1.225
        shaft_power = thrust*vair/eta_prop
        throttle = shaft_power / (factor*reference_power*sig**0.5)
        psfc = self.psfc_effective

        return {"throttle":throttle, "thrust":thrust, "power":shaft_power, "psfc":psfc}

    def print_operation(self, dict):
        print("")
        print("Electric nacelle operation")
        print("---------------------------------------------------------")
        print("Throttle = ", "%.3f"%dict["throttle"])
        print("Thrust = ", "%.2f"%unit.daN_N(dict["thrust"]), " daN")
        print("Shaft power = ", "%.2f"%unit.kW_W(dict["power"]), " kW")
        print("Power Specific Fuel Consumption = ", "%.3f"%(dict["psfc"]/unit.h_s(1.)/unit.kW_W(1.)), " kg/h/kW")

    def print(self):
        print("")
        print("Turboprop nacelle")
        print("---------------------------------------------------------")
        print("Propeller efficiency = ", "%.3f"%self.propeller_efficiency)
        print("Propeller disk load = ", "%.0f"%self.propeller_disk_load, " N/m2")
        print("Propeller diameter = ", "%.2f"%self.propeller_width, " m")
        print("Propeller mass = ", "%.2f"%self.propeller_mass, " kg")
        print("Engine mass = ", "%.2f"%self.engine_mass, " kg")
        print("")
        print("Nacelle power density = ", "%.2f"%unit.kW_W(self.nacelle_pw_density), " kW/kg")
        print("Form factor = ", "%.3f"%self.form_factor)
        print("Nacelle width = ", "%.2f"%self.width, " m")
        print("Nacelle length = ", "%.2f"%self.length, " m")
        print("Nacelle gross wetted area = ", "%.2f"%self.gross_wet_area, " m2")
        print("Nacelle net wetted area = ", "%.2f"%self.net_wet_area, " m2")
        print("Nacelle aerodynamic length = ", "%.0f"%self.aero_length, " m")
        print("Nacelle mass = ", "%.2f"%self.nacelle_mass, " kg")
        print("")
        print("Total mass = ", "%.2f"%self.mass, " kg")


if __name__ == "__main__":

    # Design time
    #-----------------------------------------------------------------
    phd = PhysicalData()        # Create phd object

    fcs = TurboPropNacelle(phd)    # Create electroprop nacelle object

    ref_power = unit.W_kW(100.)     # Design power
    ref_speed = unit.mps_kmph(120.) # Design speed

    fcs.design(ref_power, ref_speed)       # Design the Stack

    fcs.print()                 # Print stack data

    # Operation time
    #-----------------------------------------------------------------
    speed = unit.mps_kmph(100.)
    rating = "MCR"
    thrtl = 1.

    disa = 0.
    altp = unit.m_ft(10000)

    pamb,tamb,g = phd.atmosphere(altp, disa)

    dict = fcs.operate_from_throttle(pamb,tamb,speed,rating, thrtl)

    fcs.print_operation(dict)

    thrust = dict["thrust"]

    dict = fcs.operate_from_thrust(pamb,tamb,speed,rating, thrust)

    fcs.print_operation(dict)


