#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Thomas Ligeois, Thierry DRUOT
"""

import numpy as np
from scipy.optimize import fsolve

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

import unit, utils
from physical_data import PhysicalData



class AirCompressor(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system
        self.phd = phd

        self.output_pressure   = unit.convert_from("bar", 1.5)

        self.adiabatic_efficiency  = 0.8
        self.mechanical_efficiency = 0.9
        self.electrical_efficiency = 0.85

    def operate_compressor(self, p_in, t_in, air_flow):
        r,gam,cp,cv = self.phd.gas_data()


        pressure_ratio = self.air_outlet_comp_pressure / p_in
       # required_h2_flow    = self.power_max /  efficiency / h2_heat
       # required_air_flow   = self.adiabatic_comp_efficiency * self.stoechimoetrie_air_over_h2 * required_h2_flow
        self.adiab_comp_power     = self.nominal_air_flow * self.air_constant_pressure_heat \
                                    * self.air_inlet_comp_temperature \
                                    * (pressure_ratio^((phd.atmosphere(self.gam)-1)/phd.atmosphere("gam"))-1)
        self.real_comp_power      = self.adiab_comp_power / self.adiabatic_comp_efficiency
        self.air_outlet_comp_temperature = (self.real_comp_power / self.air_constant_pressure_heat) \
                                           + self.air_inlet_comp_temperature

        self.real_comp_mechanical_power  = self.real_comp_power / self.mechanical_comp_efficiency
        self.real_comp_electrical_power  = self.real_comp_mechanical_power / self.electrical_comp_efficiency




