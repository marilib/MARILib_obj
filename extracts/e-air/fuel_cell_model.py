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



class FuelCell_PEMBT(object):

    def __init__(self, phd):
        self.phd = phd

        self.cell_area = unit.convert_from("cm2", 400)              # m2
        self.max_current_density = 5./unit.convert_from("cm2",1)    # A/m2
        self.cell_entry_pressure = unit.convert_from("bar", 1.5)    # Gas pressure at electrode entry
        self.working_temperature = 273.15 + 65                      # Cell working temperature
        self.air_over_feeding = 3                                   # air flow ratio over stoechiometry
        self.power_margin = 0.8                                     # Ratio allowed power over max power

        self.n_stack = None
        self.n_cell = None
        self.power_max = None
        self.nominal_thermal_power = None
        self.nominal_voltage = None
        self.nominal_current = None
        self.nominal_h2_flow = None
        self.nominal_air_flow = None

        self.h2_heat_value = unit.convert_from("MJ", 121)    # MJ/kg
        self.h2_molar_mass = unit.convert_from("g", 2.01588)    # kg/mol
        self.air_molar_mass = unit.convert_from("g", 28.965)    # kg/mol
        self.air_o2_ratio = 0.20946                             # Mole ratio of dioxygen in the air

    def faraday_constant(self):
        return 96485.3329   # Coulomb/mole

    def ideal_gas_constant(self):
        return 8.3144621   # J/K/mol


    def enernst(self, temp, p_h2, p_o2):
        """Determination de l'enthalpis de Gibbs : DeltaG_0
        Hypothèse: l'eau produite est liquide
        """
        temp_ref = 298          # Température de référence

        dh_std_h2o = -285.83e3  # Enthalpie standard de l'eau liquide
        dh_std_h2 = 0           # Enthalpie standard de l'hydrogène gazeux
        dh_std_o2 = 0           # Enthalpie standard de l'oxygène gazeux

        ds_std_h2o = 69.91  # Enthropie standard de l'eau liquide
        ds_std_h2 = 130.684 # Enthropie standard de l'hydrogène gazeux
        ds_std_o2 = 205.138 # Enthropie standard de l'oxygène gazeux

        # Coefficients de calcul des capacités thermique molaire
        a_h2o = 30.33
        b_h2o = 0.0096056
        c_h2o = 0.0000011829
        
        a_h2 = 29.038
        b_h2 = -0.0008356
        c_h2 = 0.0000020097
        
        a_o2 = 25.699
        b_o2 = 0.012966
        c_o2 = -0.0000038581
        
        # Calcul de l'enthalpie globale
        dh_h2o =  dh_std_h2o + a_h2o*(temp-temp_ref) \
                + (1/2) * b_h2o*(temp**2-temp_ref**2) \
                + (1/3) * c_h2o*(temp**3-temp_ref**3)

        dh_h2 =  dh_std_h2 + a_h2*(temp-temp_ref) \
               + (1/2) * b_h2*(temp**2-temp_ref**2) \
               + (1/3) * c_h2*(temp**3-temp_ref**3)

        dh_o2 =  dh_std_o2 + a_o2*(temp-temp_ref) \
               + (1/2) * b_o2*(temp**2-temp_ref**2) \
               + (1/3) * c_o2*(temp**3-temp_ref**3)

        delta_h = dh_h2o - (1/2) * dh_o2 - dh_h2
            
        # Calcul de l'entropie globale
        ds_h2o =  ds_std_h2o + a_h2o*np.log(temp/temp_ref) \
                + b_h2o*(temp-temp_ref) + (1/2) * c_h2o*(temp**2-temp_ref**2)

        ds_h2 =  ds_std_h2 + a_h2*np.log(temp/temp_ref) \
               + b_h2*(temp-temp_ref) + (1/2) * c_h2*(temp**2-temp_ref**2)

        ds_o2 =  ds_std_o2 + a_o2*np.log(temp/temp_ref) \
               + b_o2*(temp-temp_ref) + (1/2) * c_o2*(temp**2-temp_ref**2)

        delta_s = ds_h2o - (1/2) * ds_o2 - ds_h2
        
        # Calcul de la variation d'enthalpie libre standard
        dg0 = abs(delta_h - temp*delta_s)
        dg = dg0 + self.ideal_gas_constant() * temp * np.log((p_h2)*np.sqrt(p_o2))

        # Calcul des potentiels de Nernst
        e_std = dg0 / (2*self.faraday_constant())      # [V] - potentiel reversible
        e_rev = dg / (2*self.faraday_constant())       # [V] - potentiel reversible + effet de la de la temp곡teure et de la pression
        e_tn = -delta_h / (2*self.faraday_constant())   # [V] - potentiel thermoneutre

        return e_rev, e_std, e_tn, dg0, dg, delta_h



    def fuel_cell_polar(self, jj, e_rev, temp):

        # Pertes d'activation + diffusion + ohmique
        # Valeurs expérimentales issue de la thèse d'Alexandra Pessot pour des conditions opératoire données

        alpha  = 0.50                   # sans unit - coefficient de transfert de charge equivalent
        j0     = 2.60e-7/unit.convert_from("cm2",1) # [A/m2] - densité de courant d'echange
        jn     = 8.08e-4/unit.convert_from("cm2",1) # [A/m2] - densité de courant équivalent de crossover (phénomènes parasites)
        r_diff = 0.081*unit.convert_from("cm2",1)   # [Ohm.m2] - résistance de diffusion surfacique
        r_ohm  = 0.0617*unit.convert_from("cm2",1)  # [Ohm.m2] - résistance ohmique surfacique

        # Calcul pertes Activation
        if (jj+jn)>j0:
            n_act = abs(  self.ideal_gas_constant() \
                        * temp / (alpha*2*self.faraday_constant()) \
                        * np.log((jj+jn)/j0) )
        else:
            n_act = 0

        n_diff = r_diff * jj    # pertes de Diffusion
        n_ohm = r_ohm * jj      # pertes Ohmiques

        # calcul tension de cellule
        voltage = e_rev - n_act - n_diff - n_ohm
        current = jj * self.cell_area

        return voltage, current, n_act, n_diff, n_ohm


    def run_fuel_cell_jj(self, jj, nc=None):
        """Compute working point of a stack of fuell cells or an individuel one if nc=1
        Input quantity jj is current density in A/m2
        """
        if nc==None: nc = self.n_cell

        p_o2 = self.air_o2_ratio * self.cell_entry_pressure
        p_h2 = self.cell_entry_pressure

        temp = self.working_temperature
        e_rev, e_std, e_tn, dg0, dg, delta_h = self.enernst(temp, p_h2, p_o2)
        voltage, current, n_act, n_diff, n_ohm = self.fuel_cell_polar(jj, e_rev, temp)

        pw_elec =  voltage * current            # Puissance electrique d'une cellule
        pw_thermal = (e_tn - voltage) * current # puissance thermique

        gas_molar_flow = current / (2 * self.faraday_constant())
        h2_flow = gas_molar_flow * self.h2_molar_mass   # kg/s, Hydrogen mass flow
        air_flow =  (gas_molar_flow / self.air_o2_ratio) * self.air_molar_mass * self.air_over_feeding  # kg/s, Air mass flow

        pw_chemical = h2_flow * self.h2_heat_value
        efficiency = pw_elec / pw_chemical

        return {"pwe":pw_elec * nc,
                "pwth":pw_thermal * nc,
                "voltage":voltage * nc,
                "current":current,
                "air_flow":air_flow * nc,
                "h2_flow":h2_flow * nc,
                "efficiency":efficiency}


    def run_fuel_cell(self, pw):
        """Compute working point of a stack of fuell cells
        Input quantity is required output electrical power pw
        """
        def fct(jj):
            return pw - self.run_fuel_cell_jj(jj)["pwe"]

        x_ini = 1000
        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        jj = output_dict[0][0]
        return self.run_fuel_cell_jj(jj)


    def design_fuel_cell_stack(self, power_max):
        """Compute the number of cell to ensure the required nominal output power
        A margin of self.power_margin is taken from versus the effective maximum power
        """

        def fct_jj(jj):
            return self.run_fuel_cell_jj(jj, 1)["pwe"]

        xini, dx = 1000, 500
        xres,yres,rc = utils.maximize_1d(xini, dx, [fct_jj])    # Compute the maximum power of the cell

        cell_pw_max = self.run_fuel_cell_jj(xres, 1)["pwe"] * self.power_margin     # Nominal power for one single cell
        self.n_cell = np.ceil( power_max / cell_pw_max )                            # Number of cells

        dict = self.run_fuel_cell(power_max)            # Run the stack for maximum power
        self.power_max = cell_pw_max * self.n_cell      # Get effective max power"
        self.nominal_thermal_power = dict["pwth"]
        self.nominal_voltage = dict["voltage"]
        self.nominal_current = dict["current"]
        self.nominal_h2_flow = dict["h2_flow"]
        self.nominal_air_flow = dict["air_flow"]
        self.nominal_efficiency = dict["efficiency"]


    def print(self):
        print("")
        print("----------------------------------------------------------")
        print("Number of cells = ", self.n_cell)
        print("Maximum output power = ", "%.2f"%unit.kW_W(self.power_max), " kW")
        print("Nominal thermal power = ", "%.2f"%unit.kW_W(self.nominal_thermal_power), " kW")
        print("Nominal current = ", "%.1f"%self.nominal_current, " A")
        print("Nominal voltage = ", "%.1f"%self.nominal_voltage, " V")
        print("Nominal air flow = ", "%.1f"%(self.nominal_air_flow*1000), " g/s")
        print("Nominal hydrogen flow = ", "%.2f"%(self.nominal_h2_flow*1000), " g/s")
        print("Nominal efficiency = ", "%.3f"%self.nominal_efficiency)

        # jj_list = np.linspace(0, self.max_current_density, 100)
        # i_list = []
        # v_list = []
        # pw_list = []
        # for jj in jj_list:
        #     dict = self.run_fuel_cell_jj(jj)
        #     i_list.append(dict["current"])
        #     v_list.append(dict["voltage"])
        #     pw_list.append(dict["pwe"])
        #
        # plt.plot(i_list,pw_list)
        # plt.grid(True)
        # plt.show()
        #
        # plt.plot(i_list,v_list)
        # plt.grid(True)
        # plt.show()



if __name__ == '__main__':

    phd = PhysicalData()

    fc = FuelCell_PEMBT(phd)

    fc.design_fuel_cell_stack(unit.W_kW(50))

    fc.print()


