#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

import unit

from physical_data import PhysicalData



class ElectroPropNacelle(object):

    def __init__(self, phd):
        self.phd = phd

        self.propeller_efficiency = 0.82
        self.propeller_disk_load = 1000.    # N/m2

        self.controller_efficiency = 0.99
        self.controller_pw_density = unit.W_kW(15.) # W/kg

        self.motor_efficiency = 0.95
        self.motor_pw_density = unit.W_kW(5.)       # W/kg

        self.nacelle_pw_density = unit.W_kW(10.)    # W/kg
        self.form_factor = 1.15

        self.reference_power = None

        self.propeller_width = None
        self.width = None
        self.length = None

        self.gross_wet_area = None
        self.net_wet_area = None
        self.aero_length = None

        self.propeller_mass = None
        self.engine_mass = None
        self.nacelle_mass = None
        self.mass = None

    def print(self):
        print("")
        print("Electric nacelle")
        print("---------------------------------------------------------")
        print("Propeller efficiency = ", "%.3f"%self.propeller_efficiency)
        print("Propeller disk load = ", "%.0f"%self.propeller_disk_load, " N/m2")
        print("Propeller diameter = ", "%.2f"%self.propeller_width, " m")
        print("Propeller mass = ", "%.2f"%self.propeller_mass, " kg")
        print("")
        print("Controller efficiency = ", "%.3f"%self.controller_efficiency)
        print("Controller power density = ", "%.2f"%unit.kW_W(self.controller_pw_density), " kW/kg")
        print("")
        print("Motor efficiency = ", "%.3f"%self.motor_efficiency)
        print("Motor power density = ", "%.2f"%unit.kW_W(self.motor_pw_density), " kW/kg")
        print("Motor mass = ", "%.2f"%self.engine_mass, " kg")
        print("")
        print("Nacelle power density = ", "%.2f"%unit.kW_W(self.nacelle_pw_density), " kW/kg")
        print("Form factor = ", "%.3f"%self.form_factor)
        print("Nacelle width = ", "%.2f"%self.width, " m")
        print("Nacelle length = ", "%.2f"%self.length, " m")
        print("Nacelle gross wetted area = ", "%.2f"%self.gross_wet_area, " m2")
        print("Nacelle net wetted area = ", "%.2f"%self.length, " m2")
        print("Nacelle aerodynamic length = ", "%.0f"%self.aero_length, " m")
        print("Nacelle mass = ", "%.2f"%self.nacelle_mass, " kg")
        print("")
        print("Total mass = ", "%.2f"%self.mass, " kg")

    def operate_from_throttle(self,vair,throttle):
        """Unitary thrust of the electroprop
        """
        pw_shaft = self.reference_power * throttle
        pw_elec = pw_shaft / (self.motor_efficiency*self.controller_efficiency)
        thrust = self.propeller_efficiency*pw_shaft/vair
        return {"throttle":throttle, "thrust":thrust, "power":pw_elec, "sec":pw_elec/thrust}

    def operate_from_thrust(self,vair,thrust):
        """Unitary thrust of the electroprop
        """
        pw_shaft = thrust * vair / self.propeller_efficiency
        throttle = pw_shaft / self.reference_power
        pw_elec = pw_shaft / (self.motor_efficiency*self.controller_efficiency)
        return {"throttle":throttle, "thrust":thrust, "power":pw_elec, "sec":pw_elec/thrust}

    def print_operation(self, dict):
        print("")
        print("Electric nacelle operation")
        print("---------------------------------------------------------")
        print("Throttle = ", "%.3f"%dict["throttle"])
        print("Thrust = ", "%.2f"%unit.daN_N(dict["thrust"]), " daN")
        print("Input power = ", "%.2f"%unit.kW_W(dict["power"]), " kW")
        print("Specific Energyy Consumption = ", "%.2f"%unit.kW_W(dict["sec"]/unit.daN_N(1.)), " kW/daN")

    def design(self, reference_power, reference_speed):
        """Computes"""
        self.reference_power = reference_power

        # Design condition
        throttle = 1.

        dict = self.operate_from_throttle(reference_speed, throttle)

        self.propeller_width = np.sqrt((4./np.pi)*((dict["thrust"]/0.80)/self.propeller_disk_load))      # Assuming 3000 N/m2

        self.width = 0.15*(reference_power/1.e3)**0.2       # statistical regression
        self.length = 0.55*(reference_power/1.e3)**0.2      # statistical regression

        self.gross_wet_area = 2.8*(self.width*self.length)  # statistical regression
        self.net_wet_area = self.gross_wet_area
        self.aero_length = self.length

        self.engine_mass = (1./self.controller_pw_density + 1./self.motor_pw_density) * reference_power
        self.nacelle_mass = (1./self.nacelle_pw_density) * reference_power
        self.propeller_mass = (165./1.5e6)*reference_power

        self.mass = self.nacelle_mass + self.engine_mass + self.propeller_mass



if __name__ == "__main__":

    # Design time
    #-----------------------------------------------------------------
    phd = PhysicalData()        # Create phd object

    fcs = ElectroPropNacelle(phd)    # Create electroprop nacelle object

    ref_power = unit.W_kW(100.)     # Design power
    ref_speed = unit.mps_kmph(120.) # Design speed

    fcs.design(ref_power, ref_speed)       # Design the Stack

    fcs.print()                 # Print stack data

    # Operation time
    #-----------------------------------------------------------------
    speed = unit.mps_kmph(100.)
    thrtl = 1.

    dict = fcs.operate_from_throttle(speed, thrtl)

    fcs.print_operation(dict)

    thrust = dict["thrust"]

    dict = fcs.operate_from_thrust(speed, thrust)

    fcs.print_operation(dict)


