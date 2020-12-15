#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

import unit

from physical_data import PhysicalData



class FuelCellStack(object):

    def __init__(self, phd):
        self.phd = phd

        # Design data
        self.fuel_type = "liquid_h2"
        self.controller_efficiency = 0.99
        self.motor_efficiency = 0.99
        self.design_disa = 25.
        self.design_altp = 0.

        # Component data
        self.fuel_cell_efficiency = 0.5
        self.fuel_cell_pw_density = unit.W_kW(2.)   # 2. kW/kg
        self.fuel_cell_output_power_ref = None      # Design output power
        self.fuel_cell_mass = None

        self.compressor_efficiency = 0.80
        self.compressor_pw_density = unit.W_kW(1.)      # 1. kW/kg
        self.compressor_over_pressure = unit.Pa_bar(1.) # Fuel cell air feed pressure
        self.compressor_power_ref = None
        self.compressor_mass = None

        self.cooling_gravimetric_index = unit.W_kW(5.)  # 5. kW/kg
        self.cooling_power_index = 0.005    # Required power per kW of heat dissipated
        self.cooling_heat_power_ref = None  # Design dissipated heat power
        self.cooling_power_ref = None       # Cooling system power
        self.cooling_mass = None

        self.wiring_efficiency = 0.999
        self.wiring_pw_density = unit.W_kW(10.)   # 10. kW/kg
        self.wiring_mass = None

        self.power_chain_mass = None    # Without controleur, motor ande propoller

    def print(self):
        print("")
        print("Power chain")
        print("---------------------------------------------------------")
        print("Fuel type = ", self.fuel_type)
        print("Controller efficiency = ", "%.3f"%self.controller_efficiency)
        print("Motor efficiency = ", "%.3f"%self.motor_efficiency)
        print("Design disa = ", "%.1f"%self.design_disa, " degK")
        print("Design altp = ", "%.0f"%unit.ft_m(self.design_altp), " ft")
        print("")
        print("Fuel cell efficiency = ", "%.3f"%self.fuel_cell_efficiency)
        print("Fuel cell power density = ", "%.1f"%unit.kW_W(self.fuel_cell_pw_density), " kW/kg")
        print("Fuel cell output power density = ", "%.1f"%unit.kW_W(self.fuel_cell_output_power_ref), " kW")
        print("Fuel cell stack mass = ", "%.0f"%self.fuel_cell_mass, " kg")
        print("")
        print("Compressor efficiency = ", "%.3f"%self.compressor_efficiency)
        print("Compressor power density = ", "%.1f"%unit.kW_W(self.compressor_pw_density), " kW/kg")
        print("Compressor over pressure = ", "%.1f"%unit.bar_Pa(self.compressor_over_pressure), " bar")
        print("Compressor design power = ", "%.1f"%unit.kW_W(self.compressor_power_ref), " kW")
        print("Compressor mass = ", "%.0f"%self.compressor_mass, " kg")
        print("")
        print("Cooling gravimetric index = ", "%.1f"%unit.kW_W(self.cooling_gravimetric_index), " kW/kg")
        print("Cooling power index = ", "%.1f"%(self.cooling_power_index), " kW/kW")
        print("Cooling design heat power = ", "%.1f"%unit.kW_W(self.cooling_heat_power_ref), " kW")
        print("Cooling design power = ", "%.1f"%unit.kW_W(self.cooling_power_ref), " kW")
        print("Cooling mass = ", "%.0f"%self.cooling_mass, " kg")
        print("")
        print("Wiring efficiency = ", "%.3f"%self.wiring_efficiency)
        print("Wiring power density = ", "%.1f"%unit.kW_W(self.wiring_pw_density), " kW/kg")
        print("Wiring mass = ", "%.0f"%self.wiring_mass, " kg")
        print("")
        print("Total power chain mass = ", "%.0f"%self.power_chain_mass, " kg")

    def operate(self, disa,altp,required_power):
        """Fuel cell stack system operation
        required_power is the TOTAL REQUIRED POWER ON ALL MOTOR SHAFTS
        """
        pamb,tamb,g = self.phd.atmosphere(altp,disa)

        eff_required_power = required_power / self.power_chain_efficiency

        dict = self.eval_fuel_cell_power(eff_required_power,pamb,tamb)

        return dict

    def design(self, reference_power):
        """Fuel cell stack system design
        reference_power is the TOTAL REQUIRED POWER ON ALL MOTOR SHAFTS
        """
        self.power_chain_efficiency =   self.wiring_efficiency \
                                      * self.controller_efficiency \
                                      * self.motor_efficiency

        # Fuell cell stack is designed for take off
        pamb,tamb,g = self.phd.atmosphere(self.design_altp, self.design_disa)

        required_power = reference_power / self.power_chain_efficiency

        dict = self.eval_fuel_cell_power(required_power,pamb,tamb)

        self.fuel_cell_output_power_ref = dict["fuel_cell_power"]
        self.compressor_power_ref = dict["compressor_power"]
        self.cooling_power_ref = dict["cooling_power"]

        # Heat dissipated by wiring and nacelles must be added to heat dissipated by fuell cells
        self.cooling_heat_power_ref = dict["heat_power"] + reference_power*(1. - self.wiring_efficiency +
                                                                            1. - self.controller_efficiency +
                                                                            1. - self.motor_efficiency)

        # Evaluate masses
        self.fuel_cell_mass = self.fuel_cell_output_power_ref / self.fuel_cell_pw_density
        self.compressor_mass = self.compressor_power_ref / self.compressor_pw_density
        self.cooling_mass = self.cooling_heat_power_ref / self.cooling_gravimetric_index
        self.wiring_mass = self.fuel_cell_output_power_ref / self.wiring_pw_density

        self.power_chain_mass =   self.fuel_cell_mass \
                                + self.compressor_mass \
                                + self.wiring_mass \
                                + self.cooling_mass

    def eval_fuel_cell_power(self,required_power,pamb,tamb):
        r,gam,Cp,Cv = self.phd.gas_data()

        fuel_heat = phd.fuel_heat(self.fuel_type)

        # air_mass_flow = fuel_cell_power * relative_air_mass_flow
        st_mass_ratio = phd.stoichiometry("air","hydrogen")
        relative_fuel_flow = (1./self.fuel_cell_efficiency) / fuel_heat
        relative_air_mass_flow = relative_fuel_flow * st_mass_ratio
        relative_compressor_power = (1./self.compressor_efficiency)*(relative_air_mass_flow*Cv)*tamb*(((pamb+self.compressor_over_pressure)/pamb)**((gam-1.)/gam)-1.)

        # heat_power = fuel_cell_power * relative_heat_power
        relative_heat_power = (1.-self.fuel_cell_efficiency)/self.fuel_cell_efficiency
        relative_cooling_power = relative_heat_power*self.cooling_power_index

        fuel_cell_power = required_power / (1. - relative_compressor_power - relative_cooling_power)
        fuel_flow = fuel_cell_power * relative_fuel_flow

        compressor_power = fuel_cell_power * relative_compressor_power
        heat_power = fuel_cell_power * relative_heat_power
        cooling_power = heat_power * self.cooling_power_index

        return {"fuel_cell_power":fuel_cell_power,
                "compressor_power":compressor_power,
                "cooling_power":cooling_power,
                "heat_power":heat_power,
                "fuel_flow":fuel_flow}

    def print_operation(self, dict):
        print("")
        print("Power chain operation")
        print("---------------------------------------------------------")
        print("Fuel cell power = ", "%.2f"%unit.kW_W(dict["fuel_cell_power"]), " kW")
        print("Compressor power = ", "%.2f"%unit.kW_W(dict["compressor_power"]), " kW")
        print("Cooling power = ", "%.2f"%unit.kW_W(dict["cooling_power"]), " kW")
        print("Heat power = ", "%.2f"%unit.kW_W(dict["heat_power"]), " kW")
        print("Fuel flow = ", "%.2f"%unit.convert_to("kg/h",dict["fuel_flow"]), " kg/h")



if __name__ == "__main__":

    # Design time
    #-----------------------------------------------------------------
    phd = PhysicalData()        # Create phd object

    fcs = FuelCellStack(phd)    # Create fuel cell stack object

    ref_power = unit.W_kW(500.) # Design power

    fcs.design(ref_power)       # Design the Stack

    fcs.print()                 # Print stack data

    # Operation time
    #-----------------------------------------------------------------
    disa = 10.
    altp = unit.m_ft(5000.)
    power = unit.W_kW(175.)

    dict = fcs.operate(disa,altp,power)

    fcs.print_operation(dict)




