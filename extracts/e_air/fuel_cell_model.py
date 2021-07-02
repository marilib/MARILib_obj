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






class FuelCellSystem(object):

    def __init__(self, phd):
        self.phd = phd

        self.n_stack = None             # Connected in series
        self.total_max_power = None

        self.air_scoop = PitotScoop(self)
        self.compressor = AirCompressor(self)
        self.precooler = AirPreCooler(self)
        self.h2_heater = HydrogenHeater(self)
        self.stack = FuelCellPEMLT(self)
        self.dissipator = WingSkinDissipator(self)

    def run_fc_system(self, pamb, tamb, vair, total_jj, nc=None):
        jj = total_jj / self.n_stack

        # Fuel cell stack
        fc_dict = self.stack.run_fuel_cell(jj, nc=nc)

        # Heater
        h2_flow = fc_dict["h2_flow"] * self.n_stack
        temp_out = self.stack.working_temperature
        ht_dict = self.h2_heater.operate(h2_flow, temp_out)

        # Air scoop
        air_flow = fc_dict["air_flow"] * self.n_stack
        sc_dict = self.air_scoop.operate(pamb, tamb, vair, air_flow)

        # Compressor
        pt_in = sc_dict["pt_out"]
        tt_in = sc_dict["tt_out"]
        cp_dict = self.compressor.operate(pt_in, tt_in, air_flow)

        # Precooler
        temp_in = cp_dict["tt_out"]
        temp_out = self.stack.working_temperature
        pc_dict = self.precooler.operate(air_flow, temp_in, temp_out)

        # System level data
        total_heat_power = fc_dict["pw_extracted"] * self.n_stack + pc_dict["pw_extracted"] - ht_dict["pw_absorbed"]
        pw_util = fc_dict["pwe"] * self.n_stack - cp_dict["pwe"]

        fc_system = {"pwe": pw_util,
                     "pwe_effective":fc_dict["pwe"] * self.n_stack,
                     "pw_extracted": total_heat_power,
                     "voltage": fc_dict["voltage"] * self.n_stack,
                     "current": fc_dict["current"],
                     "air_flow": fc_dict["air_flow"] * self.n_stack,
                     "h2_flow": fc_dict["h2_flow"] * self.n_stack,
                     "pw_chemical": fc_dict["pw_chemical"] * self.n_stack,
                     "efficiency": pw_util / (fc_dict["pw_chemical"] * self.n_stack)}

        return {"system":fc_system, "stack":fc_dict, "h2_heater":ht_dict, "scoop":sc_dict, "compressor":cp_dict, "precooler":pc_dict}

    def operate(self, pamb, tamb, vair, pw_req):

        def fct(jj):
            dict = self.run_fc_system(pamb, tamb,vair, jj)
            return pw_req - dict["system"]["pwe"]

        jj_ini = 1000
        output_dict = fsolve(fct, x0=jj_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        jj = output_dict[0][0]

        return self.run_fc_system(pamb, tamb, vair, jj)

    def design(self, pamb, tamb, vair, n_stack, req_stack_power):
        """Fuel Cell System design
        WARNING : No optimization of the cell area at that time
        """
        self.n_stack = n_stack
        self.total_max_power = req_stack_power * n_stack

        # Precooler design
        self.precooler.design()

        # Hydrogen heater design
        self.h2_heater.design()

        self.stack.eval_cell_max_power()            # Max power corresponds to self.stack.power_margin of effective max power
        
        jj = self.stack.cell_max_current_density    # Single cell design point

        def fct(n_cell):
            """WARNING : no preliminary design here because both scoop and compressor operation models are independent from design point
            """
            dict = self.run_fc_system(pamb, tamb, vair, jj, nc=n_cell)
            pw_util = dict["stack"]["pwe"]*n_stack - dict["compressor"]["pwe"]
            return req_stack_power*n_stack - pw_util

        nc_ini = req_stack_power / self.stack.cell_max_power
        output_dict = fsolve(fct, x0=nc_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        n_cell = output_dict[0][0]

        dict = self.run_fc_system(pamb, tamb, vair, jj, nc=n_cell)

        # Scoop design
        air_flow = dict["stack"]["air_flow"]
        self.air_scoop.design(pamb, tamb, vair, air_flow)

        # Compressor design
        tt_in = dict["compressor"]["tt_in"]
        p_ratio = dict["compressor"]["p_ratio"]
        self.compressor.design(tt_in, p_ratio, air_flow)

        # Stack design
        fc_req_power = dict["stack"]["pwe"]
        self.stack.design(fc_req_power)

    def print_design(self, graph=False):
        print("")
        print("Fuel cell system design characteristics")
        print("===============================================================================")
        print("Number of stack = ", "%.0f"%self.n_stack)
        print("Design power = ", "%.1f"%unit.convert_to("kW", self.total_max_power), " kW")
        self.air_scoop.print_design()
        self.compressor.print_design()
        self.precooler.print_design()
        self.h2_heater.print_design()
        self.stack.print_design(graph=graph)

    def print_operate(self, dict):
        print("")
        print("Fuel cell system operation characteristics")
        print("===============================================================================")
        print("Total usable power = ", "%.2f"%unit.kW_W(dict["system"]["pwe"]), " kW")
        print("Syetem thermal power = ", "%.2f"%unit.kW_W(dict["system"]["pw_extracted"]), " kW")
        print("Overall efficiency = ", "%.4f"%(dict["system"]["efficiency"]))
        print("")
        print("Hydrogen mass flow = ", "%.2f"%(dict["system"]["h2_flow"]*1000), " g/s")
        print("Total effective power = ", "%.2f"%unit.kW_W(dict["system"]["pwe_effective"]), " kW")
        print("Voltage = ", "%.1f"%(dict["system"]["voltage"]), " V")
        print("Current = ", "%.1f"%(dict["system"]["current"]), " A")

        self.air_scoop.print_operate(dict["scoop"])
        self.compressor.print_operate(dict["compressor"])
        self.precooler.print_operate(dict["precooler"])
        self.h2_heater.print_operate(dict["h2_heater"])
        self.stack.print_operate(dict["stack"])


class PitotScoop(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.diameter = None
        self.design_drag = None

        self.volume = np.nan
        self.mass = np.nan

    def operate(self, pamb, tamb, vair, air_flow):
        """WARNING : pitot scoop operation is independent from design point
        this characteristic is exploited in the design process at system level
        """
        mach = vair / self.fc_system.phd.sound_speed(tamb)
        ptot = self.fc_system.phd.total_pressure(pamb,mach)
        ttot = self.fc_system.phd.total_temperature(tamb,mach)
        captation_drag = air_flow * vair
        return {"p_in":pamb,
                "t_in":tamb,
                "air_flow":air_flow,
                "pt_out":ptot,
                "tt_out":ttot,
                "drag":captation_drag}

    def design(self, pamb, tamb, vair, air_flow):
        """Compute the diameter of the scoop according to flight conditions
        """
        rho = self.fc_system.phd.gas_density(pamb,tamb)
        sec = air_flow / (rho*vair)
        self.diameter = np.sqrt(4*sec/np.pi)
        self.design_drag = air_flow * vair

    def print_design(self):
        print("")
        print("Pitot scoop design characteristics")
        print("----------------------------------------------------------")
        print("Pitot diameter = ", "%.2f"%unit.convert_to("cm",self.diameter), " cm")
        print("Design drag = ", "%.2f"%unit.convert_to("daN",self.design_drag), " daN")
        print("")
        print("Volume allocation = ", "%.0f"%unit.convert_to("dm3", self.volume), " dm3")
        print("Mass = ", "%.1f"%self.mass, " kg")

    def print_operate(self, dict):
        print("")
        print("Pitot scoop operation characteristics")
        print("----------------------------------------------------------")
        print("Output total temperature = ", "%.1f"%dict["tt_out"], " K")
        print("Output total pressure = ", "%.3f"%unit.convert_to("bar", dict["pt_out"]), " bar")
        print("Air mass flow = ", "%.2f"%(dict["air_flow"]*1000), " g/s")
        print("Drag = ", "%.1f"%dict["drag"], " N")



class HydrogenHeater(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.h2_molar_mass = unit.convert_from("g", 2.01588)    # kg/mol
        self.h2_ortho_ratio = 0.75                              # Proportion of ortho allotropic form in gazeous hydrogen
        self.h2_orth_para_heat = unit.convert_from("kJ",527)    # J/kg, Heat extracted to transform ortho into para allotropic form
        self.h2_boiling_temp = 20.28                            # Boiling temperature of liquid hydrogen
        self.h2_vap_latent_heat = 449.36/self.h2_molar_mass     # J/kg, Vaporisation latent heat
        self.h2_specific_heat = unit.convert_from("kJ",14.3)    # J/kg, H2 specific heat, supposed constant above 273 kelvin
        self.h2_integral_heat = unit.convert_from("kJ",3100)    # J/kg, heat to warm 1 kg of gazeous hydrogen from 20.3 K to 273.15 K

        self.h2_heat_liq2zero = None            # J/kg Amount of heat to bring 1kg of liquid H2 to 0°C

        self.volume = None
        self.mass = None

    def design(self):
        self.h2_heat_liq2zero =  self.h2_ortho_ratio * self.h2_orth_para_heat \
                               + self.h2_vap_latent_heat \
                               + self.h2_integral_heat
        self.volume = np.nan
        self.mass = np.nan

    def operate(self, h2_flow, temp_out):
        if temp_out<273.15:
            raise Exception("Hydrogen heater output temperature must be higher or equal to 273.15")
        pw_absorbed = (self.h2_heat_liq2zero + self.h2_specific_heat * (temp_out - 273.15)) * h2_flow
        return {"h2_flow":h2_flow,
                "tt_out":temp_out,
                "pw_absorbed":pw_absorbed}

    def print_design(self):
        print("")
        print("Hydrogen heater design characteristics")
        print("----------------------------------------------------------")
        print("Liquid to 0°C specific heat = ", "%.2f"%unit.convert_to("kJ",self.h2_heat_liq2zero), " kJ/kg")
        print("")
        print("Volume allocation = ", "%.0f"%unit.convert_to("dm3", self.volume), " dm3")
        print("Mass = ", "%.1f"%self.mass, " kg")

    def print_operate(self, dict):
        print("")
        print("Hydrogen heater operation characteristics")
        print("----------------------------------------------------------")
        print("Hydrogen mass flow = ", "%.2f"%(dict["h2_flow"]*1000), " g/s")
        print("Output total temperature = ", "%.2f"%dict["tt_out"], " K")
        print("Absorbed heat power = ", "%.2f"%unit.convert_to("kW",dict["pw_absorbed"]), " kW")



class AirPreCooler(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        r,gam,cp,cv = self.fc_system.phd.gas_data()

        self.air_specific_heat = cp     # J/kg, Air specific heat

        self.volume = None
        self.mass = None

    def design(self):
        self.volume = np.nan
        self.mass = np.nan

    def operate(self, air_flow, temp_in, temp_out):
        pw_extracted = self.air_specific_heat * (temp_in - temp_out) * air_flow
        return {"air_flow":air_flow,
                "tt_in":temp_in,
                "tt_out":temp_out,
                "pw_extracted":pw_extracted}

    def print_design(self):
        print("")
        print("Precooler design characteristics")
        print("----------------------------------------------------------")
        print("Volume allocation = ", "%.0f"%unit.convert_to("dm3", self.volume), " dm3")
        print("Mass = ", "%.1f"%self.mass, " kg")

    def print_operate(self, dict):
        print("")
        print("Precooler operation characteristics")
        print("----------------------------------------------------------")
        print("Air mass flow = ", "%.2f"%(dict["air_flow"]*1000), " g/s")
        print("Input total temperature = ", "%.2f"%dict["tt_in"], " K")
        print("Output total temperature = ", "%.2f"%dict["tt_out"], " K")
        print("Extracted heat power = ", "%.2f"%unit.convert_to("kW",dict["pw_extracted"]), " kW")



class AirCompressor(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.output_pressure   = unit.convert_from("bar", 1.5)

        self.adiabatic_efficiency  = 0.8
        self.mechanical_efficiency = 0.9
        self.electrical_efficiency = 0.85

        self.design_air_flow = None
        self.design_p_ratio = None
        self.design_shaft_power = None
        self.design_elec_power = None

        self.volume = np.nan
        self.mass = np.nan

    def operate(self, pt_in, tt_in, air_flow):
        """WARNING : compressor operation is independent from design point
        this characteristic is exploited in the design process at system level
        """
        r,gam,cp,cv = self.fc_system.phd.gas_data()

        pt_out = self.fc_system.stack.cell_entry_total_pressure
        pressure_ratio = pt_out / pt_in
        adiabatic_power = air_flow * cp * tt_in * (pressure_ratio**((gam-1)/gam)-1)
        real_power = adiabatic_power / self.adiabatic_efficiency

        tt_out = (real_power / (air_flow*cp)) + tt_in

        shaft_power = real_power / self.mechanical_efficiency
        elec_power = shaft_power / self.electrical_efficiency

        return {"air_flow":air_flow,
                "pt_in":pt_in,
                "tt_in":tt_in,
                "pt_out":pt_out,
                "tt_out":tt_out,
                "p_ratio":pt_out/pt_in,
                "pwe":elec_power}

    def design(self, tt_in, p_ratio, air_flow):
        r,gam,cp,cv = self.fc_system.phd.gas_data()

        self.design_air_flow = air_flow
        self.design_p_ratio = p_ratio

        adiabatic_power = air_flow * cp * tt_in * (p_ratio**((gam-1)/gam)-1)
        real_power = adiabatic_power / self.adiabatic_efficiency

        self.design_shaft_power = real_power / self.mechanical_efficiency
        self.design_elec_power = self.design_shaft_power / self.electrical_efficiency

    def print_design(self):
        print("")
        print("Compressor design characteristics")
        print("----------------------------------------------------------")
        print("Adiabatic efficiency = ", "%.3f"%self.adiabatic_efficiency)
        print("Mechanical efficiency = ", "%.3f"%self.mechanical_efficiency)
        print("Electrical efficiency = ", "%.3f"%self.electrical_efficiency)
        print("")
        print("Design air mass flow = ", "%.0f"%unit.convert_to("g",self.design_air_flow), " g/s")
        print("Design pressure ratio = ", "%.3f"%self.design_p_ratio)
        print("Design shaft power = ", "%.2f"%unit.convert_to("kW",self.design_shaft_power), " kW")
        print("Design electric power = ", "%.2f"%unit.convert_to("kW",self.design_elec_power), " kW")
        print("")
        print("Volume allocation = ", "%.0f"%unit.convert_to("dm3", self.volume), " dm3")
        print("Mass = ", "%.1f"%self.mass, " kg")

    def print_operate(self, dict):
        print("")
        print("Compressor operation characteristics")
        print("----------------------------------------------------------")
        print("Air mass flow = ", "%.2f"%(dict["air_flow"]*1000), " g/s")
        print("Output total temperature = ", "%.2f"%dict["tt_out"], " K")
        print("Output total pressure = ", "%.3f"%unit.convert_to("bar", dict["pt_out"]), " bar")
        print("Compression pressure ratio = ", "%.2f"%dict["p_ratio"])
        print("Required electrical power = ", "%.2f"%unit.convert_to("kW", dict["pwe"]), " kW")



class FuelCellPEMLT(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.cell_area = unit.convert_from("cm2", 625)              # m2
        self.max_current_density = 4./unit.convert_from("cm2",1)    # A/m2
        self.cell_entry_total_pressure = unit.convert_from("bar", 1.5)    # Gas pressure at electrode entry
        self.working_temperature = 273.15 + 75                      # Cell working temperature
        self.air_over_feeding = 2                                   # air flow ratio over stoechiometry
        self.power_margin = 0.8                                     # Ratio allowed power over max power
        self.heat_washout_factor = 0.12                             # Fraction of heat washed out by the air flow across the stack

        self.bip_thickness = unit.convert_from("cm",0.65)           # Thickness of one bipolar plate
        self.end_plate_thickness = unit.convert_from("cm",1.15)     # Thickness of one stack end plate

        self.cell_max_power = None          # Max power for one single cell
        self.cell_max_current = None
        self.cell_max_current_density = None
        self.cell_max_current_voltage = None

        self.n_cell = None
        self.power_max = None
        self.nominal_thermal_power = None
        self.nominal_voltage = None
        self.nominal_current = None
        self.nominal_current_density = None
        self.nominal_h2_flow = None
        self.nominal_air_flow = None

        self.volume = None
        self.mass = None

        self.h2_molar_mass = unit.convert_from("g", 2.01588)    # kg/mol
        self.air_molar_mass = unit.convert_from("g", 28.965)    # kg/mol
        self.air_o2_ratio = 0.20946                             # Mole ratio of dioxygen in the air

    def faraday_constant(self):
        return 96485.3329   # Coulomb/mole

    def ideal_gas_constant(self):
        return 8.3144621   # J/K/mol

    def e_nernst(self, temp, p_h2, p_o2):
        """Determination de l'enthalpis de Gibbs : DeltaG_0
        Hypothèse: l'eau produite est liquide
        """
        temp_ref = 298          # Température de référence
        press_ref = 101325      # Pression de référence

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
        dg = dg0 + self.ideal_gas_constant() * temp * (np.log(p_h2/press_ref) + (1/2)*np.log(p_o2/press_ref))

        # Calcul des potentiels de Nernst
        e_std = dg0 / (2*self.faraday_constant())      # [V] - potentiel reversible
        e_rev = dg / (2*self.faraday_constant())       # [V] - potentiel reversible + effet de la de la temp곡teure et de la pression
        e_tn = -delta_h / (2*self.faraday_constant())  # [V] - potentiel thermoneutre

        return e_rev, e_std, e_tn, dg0, dg, delta_h

    def fuel_cell_polar(self, jj, e_rev, temp):

        # Pertes d'activation + diffusion + ohmique
        # Valeurs expérimentales issue de la thèse d'Alexandra Pessot pour des conditions opératoire données

        alpha  = 0.50                               # sans unit - coefficient de transfert de charge equivalent
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

    def run_fuel_cell(self, jj, nc=None):
        """Compute working point of a stack of fuell cells or an individuel one if nc=1
        Input quantity jj is current density in A/m2
        """
        if nc==None: nc = self.n_cell

        p_o2 = self.air_o2_ratio * self.cell_entry_total_pressure
        p_h2 = self.cell_entry_total_pressure

        temp = self.working_temperature
        e_rev, e_std, e_tn, dg0, dg, delta_h = self.e_nernst(temp, p_h2, p_o2)
        voltage, current, n_act, n_diff, n_ohm = self.fuel_cell_polar(jj, e_rev, temp)

        pw_elec =  voltage * current            # Puissance electrique d'une cellule
        pw_th_total = (e_tn - voltage) * current # puissance thermique
        pw_thermal = pw_th_total * (1 - self.heat_washout_factor)

        gas_molar_flow = current / (2 * self.faraday_constant())
        h2_flow = gas_molar_flow * self.h2_molar_mass   # kg/s, Hydrogen mass flow
        air_flow =  (gas_molar_flow / self.air_o2_ratio) * self.air_molar_mass * self.air_over_feeding  # kg/s, Air mass flow

        h2_heat = -delta_h / self.h2_molar_mass     # H2 internal combustion energy
        pw_chemical = h2_flow * h2_heat
        efficiency = pw_elec / pw_chemical

        return {"pwe":pw_elec * nc,
                "pw_extracted":pw_thermal * nc,
                "voltage":voltage * nc,
                "current":current,
                "jj":jj,
                "air_flow":air_flow * nc,
                "h2_flow":h2_flow * nc,
                "pw_chemical":pw_chemical * nc,
                "efficiency":efficiency}

    def operate(self, pw):
        """Compute working point of a stack of fuell cells
        Input quantity is required output electrical power pw
        """
        def fct(jj):
            return pw - self.run_fuel_cell(jj)["pwe"]

        x_ini = 1000
        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        jj = output_dict[0][0]
        return self.run_fuel_cell(jj)

    def eval_cell_max_power(self):
        def fct_jj(jj):
            return self.run_fuel_cell(jj, nc=1)["pwe"]

        xini, dx = 1000, 500
        xres,yres,rc = utils.maximize_1d(xini, dx, [fct_jj])    # Compute the maximum power of the cell

        dict = self.run_fuel_cell(xres, nc=1)

        self.cell_max_power = dict["pwe"] * self.power_margin   # Nominal power for one single cell
        self.cell_max_current = dict["current"]
        self.cell_max_current_density = dict["jj"]
        self.cell_max_current_voltage = dict["voltage"]

    def design(self, power_max):
        """Compute the number of cell to ensure the required nominal output power
        A margin of self.power_margin is taken from versus the effective maximum power
        """
        self.n_cell = np.ceil( power_max / self.cell_max_power )    # Number of cells

        self.power_max = self.cell_max_power * self.n_cell          # Get effective max power"
        dict = self.operate(self.power_max)                         # Run the stack for maximum power

        # tightening fuel cell mass calculation
        end_plate_area = 2 * self.cell_area
        end_plate_volume = 2 * end_plate_area * self.end_plate_thickness
        end_plate_mass = 2.7e-9 * end_plate_volume

        # bipolar fuel cell mass calculation
        n_bip = self.n_cell + 1                             # Number of bipolar plate
        bip_area = 1.7e-4 * self.cell_area
        bip_volume = self.bip_thickness * bip_area * n_bip
        bip_effective_volume = 0.666 * bip_volume
        bip_mass = bip_effective_volume * 1.89e-9
        self.mass = bip_mass + end_plate_mass               # To be confirmed
        self.volume = np.nan

        self.nominal_thermal_power = dict["pw_extracted"]
        self.nominal_voltage = dict["voltage"]
        self.nominal_current = dict["current"]
        self.nominal_current_density = dict["jj"]
        self.nominal_h2_flow = dict["h2_flow"]
        self.nominal_air_flow = dict["air_flow"]
        self.nominal_efficiency = dict["efficiency"]

    def print_design(self, graph=False):
        print("")
        print("Fuel Cell stack design characteristics")
        print("----------------------------------------------------------")
        print("Cell area = ", "%.2f"%unit.convert_to("cm2",self.cell_area), " cm2")
        print("Cell effective maximum power = ", "%.2f"%(self.cell_max_power), " W")
        print("Cell effective maximum current = ", "%.2f"%(self.cell_max_current), " A")
        print("Cell effective maximum current density = ", "%.3f"%(self.cell_max_current_density/1e4), " A/cm2")
        print("Cell effective maximum current voltage = ", "%.3f"%(self.cell_max_current_voltage), " V")
        print("")
        print("Number of cells of the stack = ", self.n_cell)
        print("Stack maximum continuous power = ", "%.2f"%unit.convert_to("kW", self.power_max), " kW")
        print("Stack maximum thermal power = ", "%.2f"%unit.convert_to("kW", self.nominal_thermal_power), " kW")
        print("Stack maximum current density = ", "%.2f"%(self.nominal_current/unit.convert_to("cm2",self.cell_area)), " A/cm2")
        print("Stack maximum current = ", "%.1f"%self.nominal_current, " A")
        print("Stack maximum voltage = ", "%.1f"%self.nominal_voltage, " V")
        print("Stack maximum air mass flow = ", "%.1f"%(self.nominal_air_flow*1000), " g/s")
        print("Stack maximum hydrogen mass flow = ", "%.2f"%(self.nominal_h2_flow*1000), " g/s")
        print("Stack maximum stack efficiency = ", "%.3f"%self.nominal_efficiency)
        print("")
        print("Volume allocation = ", "%.0f"%unit.convert_to("dm3", self.volume), " dm3")
        print("Mass = ", "%.1f"%self.mass, " kg")

        if graph:
            jj_list = np.linspace(0.01, self.max_current_density, 100)
            j_list = []
            v_list = []
            w_list = []
            e_list = []
            for jj in jj_list:
                dict = self.run_fuel_cell(jj, nc=1)
                j_list.append(dict["jj"]/1e4)
                v_list.append(dict["voltage"])
                e_list.append(dict["efficiency"])
                w_list.append(dict["pwe"])

            plt.plot(j_list,v_list)
            plt.grid(True)
            plt.xlabel('Current density (A/cm2)')
            plt.ylabel('Voltage (Volt)')
            plt.show()

            plt.plot(j_list,w_list)
            plt.grid(True)
            plt.xlabel('Current density (A/cm2)')
            plt.ylabel('Power (W)')
            plt.show()

            plt.plot(j_list,e_list)
            plt.grid(True)
            plt.xlabel('Current density (A/cm2)')
            plt.ylabel('Efficiency')
            plt.show()

    def print_operate(self, dict):
        print("")
        print("Fuel cell stack operation characteristics")
        print("----------------------------------------------------------")
        print("Stack effective power = ", "%.2f"%unit.kW_W(dict["pwe"]), " kW")
        print("Extracted heat power = ", "%.2f"%unit.kW_W(dict["pw_extracted"]), " kW")
        print("")
        print("Hydrogen mass flow = ", "%.2f"%(dict["h2_flow"]*1000), " g/s")
        print("Stack current density = ", "%.4f"%(dict["jj"]/1e4), " A/cm2")
        print("Stack effective efficiency = ", "%.4f"%(dict["efficiency"]))
        print("")
        print("Voltage = ", "%.1f"%(dict["voltage"]), " V")
        print("Current = ", "%.1f"%(dict["current"]), " A")



class WingSkinCircuit(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.available_span = None
        self.available_curved_length = None
        self.mean_chord = None

        self.skin_thickness = unit.convert_from("mm",1.5)
        self.skin_conduct = 237.        # W/m/K, Aluminium thermal conductivity
        self.skin_exchange_area = None  # Exchange area with one tube

        self.tube_density = 2700        # Aluminium density
        self.tube_width = unit.convert_from("mm",10)
        self.tube_height = unit.convert_from("mm",6)
        self.tube_thickness = unit.convert_from("mm",0.5)
        self.tube_foot_width = unit.convert_from("mm",0.25)
        self.tube_footprint_width = None

        self.tube_count = None
        self.tube_length = None
        self.tube_section = None
        self.tube_hydro_width = None
        self.tube_exchange_area = None

        self.tube_mass = None
        self.tube_fluid_mass = None

    def pressure_drop(self,temp,speed,hydro_width,tube_length):
        """Pressure drop along a cylindrical tube
        """
        rho, cp, mu, lbd = self.fc_system.phd.fluid_data(temp, fluid="water_mp30")
        rex = (rho * speed / mu) * hydro_width                       # Reynolds number
        cf = 0.5
        for j in range(6):
            cf = (1./(2.*np.log(rex*np.sqrt(cf))/np.log(10)-0.8))**2
        dp = 0.5 * cf * (tube_length/hydro_width) * rho * speed**2
        return dp

    def design(self, span, web_width, mean_chord):
        self.available_span = span
        self.available_curved_length = web_width
        self.mean_chord = mean_chord

        # The fluid makes a loop along the span from and to the nacelle
        self.tube_length = 2 * self.available_span
        self.tube_exchange_area = self.tube_width * self.tube_length
        self.skin_exchange_area = (self.tube_width + 2*self.tube_foot_width) * self.tube_length

        self.tube_footprint_width = self.tube_width + 2*(self.tube_thickness + self.tube_foot_width)
        self.tube_count = np.floor(0.5*self.available_curved_length / self.tube_footprint_width)

        if self.tube_height<0.5*self.tube_width:
            raise Exception("Tube height 'ht' cannot be lower than half tube width 'wt'")
        self.tube_section = 0.125*np.pi*self.tube_width**2 + (self.tube_height-0.5*self.tube_width)*self.tube_width      # Tube section
        rho_f, cp_f, mu_f, lbd_f = self.fc_system.phd.fluid_data(tamb, fluid="water_mp30")

        tube_prm = 0.5*np.pi*self.tube_width + 2*(self.tube_height-0.5*self.tube_width) + self.tube_width    # Tube perimeter
        self.tube_hydro_width = 4 * self.tube_section / tube_prm             # Hydrolic section

        self.tube_fluid_mass = self.tube_section * self.tube_length * self.tube_count * rho_f
        self.tube_mass =  (0.5*np.pi*self.tube_width + 2*(self.tube_height-0.5*self.tube_width) + 2*self.tube_foot_width) \
                        * self.tube_thickness * self.tube_length * (2*self.tube_count) * self.tube_density

    def operate(self, pamb, tamb, air_speed, fluid_speed, fluid_temp_in, fluid_temp_out):
        ha,rhoa,cpa,mua,pra,rea,nua,lbda = self.fc_system.phd.air_thermal_transfer_data(pamb,tamb,air_speed, self.mean_chord)
        temp = 0.5 * (fluid_temp_in + fluid_temp_out)
        hf,rhof,cpf,muf,prf,redf,nudf,lbdf = self.fc_system.phd.fluid_thermal_transfer_data(temp, fluid_speed, self.tube_hydro_width)
        kail = np.sqrt(hf * 2*self.tube_length * self.skin_conduct * 2*self.tube_thickness*self.tube_length)

        # self.skin_exchange_area is taken as reference area
        ks = 1 / (  1/ha
                  + (self.skin_thickness/self.skin_conduct) * (self.skin_exchange_area/self.tube_exchange_area)
                  + 1/((kail + hf*self.tube_exchange_area) / self.skin_exchange_area)
                 )

        fluid_flow = rhof * self.tube_section * self.tube_count * fluid_speed

        def fct(temp_out):
            q_fluid = fluid_flow * cpf * (fluid_temp_in - temp_out)
            q_out = - ks * self.tube_count * self.skin_exchange_area * (fluid_temp_in - temp_out) / np.log((temp_out-tamb)/(fluid_temp_in-tamb))
            return q_fluid-q_out

        temp_ini = fluid_temp_out
        output_dict = fsolve(fct, x0=temp_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        fluid_temp_out = output_dict[0][0]

        q_fluid = fluid_flow * cpf * (fluid_temp_in - fluid_temp_out)
        q_out = - ks * self.tube_count * self.skin_exchange_area * (fluid_temp_in - fluid_temp_out) / np.log((fluid_temp_out-tamb)/(fluid_temp_in-tamb))

        temp = 0.5*(fluid_temp_in + fluid_temp_out)
        pd = self.pressure_drop(temp, fluid_speed, self.tube_hydro_width, self.tube_length)
        pwd = pd * (fluid_flow / rhof)

        return {"temp_in":fluid_temp_in,
                "temp_out":fluid_temp_out,
                "q_fluid":q_fluid,
                "q_out":q_out,
                "flow":fluid_flow,
                "p_drop":pd,
                "pw_drop":pwd,
                "ks":ks,
                "ha":ha,
                "rhoa":rhoa,
                "cpa":cpa,
                "mua":mua,
                "pra":pra,
                "rea":rea,
                "nua":nua,
                "lbda":lbda,
                "rhof":rhof,
                "hf":hf,
                "cpf":cpf,
                "muf":muf,
                "prf":prf,
                "redf":redf,
                "nudf":nudf,
                "lbdf":lbdf}

    def print_design(self):
        print("")
        print("Wing skin dissipator characteristics")
        print("----------------------------------------------------------")
        print("Tube width = ", "%.2f"%unit.convert_to("mm",self.tube_width), " mm")
        print("Tube height = ", "%.2f"%unit.convert_to("mm",self.tube_height), " mm")
        print("Tube foot width = ", "%.2f"%unit.convert_to("mm",self.tube_foot_width), " mm")
        print("Tube footprint width = ", "%.2f"%unit.convert_to("mm",self.tube_footprint_width), " mm")
        print("Tube wall thickness = ", "%.2f"%unit.convert_to("mm",self.tube_thickness), " mm")
        print("Tube internal section = ", "%.2f"%unit.convert_to("cm2",self.tube_section), " cm2")
        print("Tube hydraulic diameter = ", "%.2f"%unit.convert_to("mm",self.tube_hydro_width), " mm")
        print("")
        print("Skin thickness = ", "%.2f"%unit.convert_to("mm",self.skin_thickness), " mm")
        print("Skin thermal conductivity = ", "%.2f"%(self.skin_conduct), "W/m/K")
        print("")
        print("Number of wing tubes = 2 x ", "%.0f"%self.tube_count)
        print("Total tube length = ", "%.2f"%(self.tube_length), " m")
        print("Skin exchange area with air = ", "%.2f"%(self.skin_exchange_area*self.tube_count), " m2")
        print("tube exchange area with skin = ", "%.2f"%(self.tube_exchange_area*self.tube_count), " m2")
        print("")
        print("Tube mass = ", "%.1f"%(self.tube_mass), " kg")
        print("Tube fluid mass = ", "%.1f"%(self.tube_fluid_mass), " kg")

    def print_operate(self, dict):
        print("")
        print("Wing skin dissipator working data")
        print("----------------------------------------------------------")
        print("Fluid input temperature = ", "%.2f"%dict["temp_in"], " °K")
        print("Fluid output temperature = ", "%.2f"%dict["temp_out"], " °K")
        print("Fluid delta temperature = ", "%.2f"%(dict["temp_in"]-dict["temp_out"]), " °K")
        print("q_fluid = ", "%.2f"%unit.convert_to("kW",dict["q_fluid"]), " kW")
        print("q_out = ", "%.2f"%unit.convert_to("kW",dict["q_out"]), " kW")
        print("Global exchange factor = ", "%.2f"%(dict["ks"]), " W/m2/K")
        print("")
        print("Fluid mass flow = ", "%.2f"%(dict["flow"]*1000), " g/s")
        print("Fluid pressure drop = ", "%.4f"%unit.convert_to("bar",dict["p_drop"]), " bar")
        print("Fluid flow power = ", "%.0f"%(dict["pw_drop"]), " W")
        print("")
        print("air heat transfer factor = ", "%.3f"%dict["ha"], " W/m2/K")
        print("rho air = ", "%.3f"%dict["rhoa"], " kg/m3")
        print("Cp air = ", "%.1f"%dict["cpa"], " J/kg")
        print("mu air = ", "%.1f"%(dict["mua"]*1e6), " 10-6Pa.s")
        print("Pr air = ", "%.4f"%dict["pra"])
        print("Re air = ", "%.0f"%dict["rea"])
        print("Nu air = ", "%.0f"%dict["nua"])
        print("Lambda air = ", "%.4f"%dict["lbda"])
        print("")
        print("fluid heat transfer factor = ", "%.3f"%dict["hf"], " W/m2/K")
        print("rho fluide = ", "%.1f"%dict["rhof"], " kg/m3")
        print("Cp fluide = ", "%.1f"%dict["cpf"], " J/kg")
        print("mu fluide = ", "%.1f"%(dict["muf"]*1e6), " 10-6Pa.s")
        print("Pr fluide = ", "%.4f"%dict["prf"])
        print("ReD fluide = ", "%.0f"%dict["redf"])
        print("NuD fluide = ", "%.2f"%dict["nudf"])
        print("Lambda fluide = ", "%.4f"%dict["lbdf"])


class WingSkinDissipator(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.circuit_le = WingSkinCircuit(fc_system)
        self.circuit_te = WingSkinCircuit(fc_system)

        self.pump_efficiency = 0.80
        self.fluid_factor = 1.15        # Factor on fluid mass for piping
        self.tube_factor = 1.20        # Factor on tube mass for piping

        self.web_tube_mass = None
        self.web_fluid_mass = None
        self.mass = None

    def design(self, wing_aspect_ratio, wing_area):
        ref_wing_chord = 2

        wing_span = np.sqrt(wing_area*wing_aspect_ratio)

        available_span = (wing_span - 2) / 2
        wing_chord = wing_area / wing_span

        le_web_width = 1.04 * (wing_chord/ref_wing_chord)
        te_web_width = 1.40 * (wing_chord/ref_wing_chord)

        le_mean_chord = 0.15 * (wing_chord/ref_wing_chord)
        te_mean_chord = 0.95 * (wing_chord/ref_wing_chord)

        self.circuit_le.design(available_span, le_web_width, le_mean_chord)
        self.circuit_te.design(available_span, te_web_width, te_mean_chord)

        self.web_tube_mass = self.circuit_le.tube_mass + self.circuit_te.tube_mass
        self.web_fluid_mass = self.circuit_le.tube_fluid_mass + self.circuit_te.tube_fluid_mass

        self.mass =  self.web_tube_mass * self.tube_factor + self.web_fluid_mass * self.fluid_factor

    def operate(self, pamb, tamb, air_speed, fluid_speed, fluid_temp_in, fluid_temp_out):
        # Both circuits are working in parallel, flows are blended at the end
        dict_le = self.circuit_le.operate(pamb, tamb, air_speed, fluid_speed, fluid_temp_in, fluid_temp_out)
        dict_te = self.circuit_te.operate(pamb, tamb, air_speed, fluid_speed, fluid_temp_in, fluid_temp_out)
        temp_out =   (dict_le["flow"]*dict_le["temp_out"] + dict_te["flow"]*dict_te["temp_out"]) / (dict_le["flow"] + dict_te["flow"])
        pressure_drop = dict_le["p_drop"]
        fluid_flow =  dict_le["flow"] + dict_te["flow"]
        pw_drop = dict_le["pw_drop"] + dict_te["pw_drop"]
        pwe = pw_drop / self.pump_efficiency

        return {"le_data": dict_le,
                "te_data": dict_te,
                "temp_in": fluid_temp_in,
                "temp_out": temp_out,
                "pw_heat": dict_le["q_fluid"] + dict_te["q_fluid"],
                "flow": fluid_flow,
                "p_drop": pressure_drop,
                "pw_drop": pw_drop,
                "pwe":pwe}

    def print_design(self):
        print("")
        print("Wing skin system dissipator design data")
        print("=========================================================================")
        print("Web tube mass = ", "%.1f"%self.web_tube_mass, " kg")
        print("Web fluid mass = ", "%.1f"%self.web_fluid_mass, " kg")
        print("Total mass = ", "%.1f"%self.mass, " kg")
        print("")
        print("Wing skin system dissipator Leading edge design data")
        print("=========================================================================")
        self.circuit_le.print_design()
        print("")
        print("Wing skin system dissipator Trailing edge design data")
        print("=========================================================================")
        self.circuit_te.print_design()

    def print_operate(self, dict):
        print("")
        print("Wing skin system dissipator working data")
        print("=========================================================================")
        print("Fluid output temperature = ", "%.2f"%dict["temp_out"], " °K")
        print("Fluid delta temperature = ", "%.2f"%(dict["temp_in"]-dict["temp_out"]), " °K")
        print("Dissipated heat power = ", "%.2f"%unit.convert_to("kW",dict["pw_heat"]), " kW")
        print("")
        print("Fluid mass flow = ", "%.2f"%(dict["flow"]*1000), " g/s")
        print("Fluid pressure drop = ", "%.4f"%unit.convert_to("bar",dict["p_drop"]), " bar")
        print("Fluid flow power = ", "%.0f"%(dict["pw_drop"]), " W")
        print("Pump electrical power = ", "%.0f"%dict["pwe"], " W")
        print("")
        print("Wing skin system dissipator Leading edge working data")
        print("=========================================================================")
        self.circuit_le.print_operate(dict["le_data"])
        print("")
        print("Wing skin system dissipator Trailing edge working data")
        print("=========================================================================")
        self.circuit_te.print_operate(dict["te_data"])


if __name__ == '__main__':

    phd = PhysicalData()

    fc_syst = FuelCellSystem(phd)

    # Fuel cell test
    #----------------------------------------------------------------------
    altp = unit.m_ft(10000)
    disa = 15
    vair = unit.mps_kmph(200)

    pamb, tamb, g = phd.atmosphere(altp, disa)

    stack_power = unit.convert_from("kW", 50)
    n_stack = 6

    fc_syst.design(pamb, tamb, vair, n_stack, stack_power)
    fc_syst.print_design(graph=True)

    req_power = unit.W_kW(100)

    dict = fc_syst.operate(pamb, tamb, vair, req_power)
    fc_syst.print_operate(dict)


    # Dissipator test
    #----------------------------------------------------------------------
    altp = unit.m_ft(10000)
    disa = 15
    vair = unit.mps_kmph(200)

    pamb, tamb, g = phd.atmosphere(altp, disa)

    fluid_speed = 2. # m/s
    fluid_temp_in = 273.15 + 75
    fluid_temp_out = 273.15 + 70    # Initial value

    wing_aspect_ratio = 13
    wing_area = 42

    fc_syst.dissipator.design(wing_aspect_ratio, wing_area)
    fc_syst.dissipator.print_design()

    dict_rad = fc_syst.dissipator.operate(pamb, tamb, vair, fluid_speed, fluid_temp_in, fluid_temp_out)
    fc_syst.dissipator.print_operate(dict_rad)


