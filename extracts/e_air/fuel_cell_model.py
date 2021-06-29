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





# def dissipate(pamb,tamb,vair,re,ws, wt,ht,et,lt, t_in, fluid_flow):
#     """
#
#     :param pamb: Ambiant pressure
#     :param tamb: Ambiant temperature
#     :param vair: Free air velocity
#     :param re: Local Reynolds number
#     :param ws: Skin thickness
#     :param wt: Tube internal width (m)
#     :param ht: Tube internal height (m)
#     :param et: Tube wall thickness
#     :param lt: Tube length (m)
#     :param t_in: Fluid entry temperature (K)
#     :param fluid_flow: Fuild mass flow in the tube (kg/s)
#     :return:
#     """
#     if ht<0.5*wt:
#         raise Exception("Tube height 'ht' cannot be lower than half tube width 'wt'")
#     tube_sec = 0.125*np.pi*wt + (ht-0.5*wt)*wt      # Tube section
#     tube_prm = 0.5*np.pi*wt + 2*(ht-0.5*wt) + wt    # Tube perimeter
#     hyd_width = 4 * tube_sec / tube_prm             # Hydrolic section
#
#     tube_exa = (wt + 2*et) * lt
#
#
#
#
#     return t_out, pwth_dissipa



class FuelCellSystem(object):

    def __init__(self, phd):
        self.phd = phd

        self.n_stack = None

        self.air_scoop = PitotScoop(self)
        self.compressor = AirCompressor(self)
        self.stack = FuelCellPEMLT(self)
        self.dissipator = LeadingEdgeDissipator(self)


    def run_fc_system(self, pamb, tamb, vair, jj, nc=None):
        fc_dict = self.stack.run_fuel_cell(jj, nc=nc)
        air_flow = fc_dict["air_flow"]
        sc_dict = self.air_scoop.operate(pamb, tamb, vair, air_flow)
        pt_in = sc_dict["pt_out"]
        tt_in = sc_dict["tt_out"]
        cp_dict = self.compressor.operate(pt_in, tt_in, air_flow)
        pw_util = fc_dict["pwe"] - cp_dict["pwe"]
        fc_system = {"efficiency":pw_util/fc_dict["pw_chemical"]}
        return {"system":fc_system, "stack":fc_dict, "scoop":sc_dict, "compressor":cp_dict}


    def operate(self, pamb, tamb, vair, pw_req):
        def fct(jj):
            dict = self.run_fc_system(pamb, tamb,vair, jj)
            return pw_req - (dict["stack"]["pwe"] - dict["compressor"]["pwe"])

        jj_ini = 1000
        output_dict = fsolve(fct, x0=jj_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        fc_power = output_dict[0][0]

        return self.run_fc_system(pamb, tamb, vair, fc_power)


    def design(self, pamb, tamb, vair, req_power):
        """Fuel Cell System design
        WARNING : No optimization of the cell area at that time
        """
        self.stack.eval_cell_max_power()            # Max power corresponds to self.stack.power_margin of effective max power
        
        jj = self.stack.cell_max_current_density    # Single cell design point

        def fct(n_cell):
            """WARNING : no preliminary design here because both scoop and compressor operation models are independent from design point
            """
            dict = self.run_fc_system(pamb, tamb, vair, jj, nc=n_cell)
            pw_util = dict["stack"]["pwe"] - dict["compressor"]["pwe"]
            return req_power - pw_util

        nc_ini = req_power / self.stack.cell_max_power
        output_dict = fsolve(fct, x0=nc_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        n_cell = output_dict[0][0]

        dict = self.run_fc_system(pamb, tamb, vair, jj, nc=n_cell)

        # Fix stack design
        fc_req_power = dict["stack"]["pwe"]
        self.stack.design(fc_req_power)

        # Fix scoop design
        air_flow = dict["stack"]["air_flow"]
        self.air_scoop.design(pamb, tamb, vair, air_flow)

        # Fix compressor design
        tt_in = dict["compressor"]["tt_in"]
        p_ratio = dict["compressor"]["p_ratio"]
        self.compressor.design(tt_in, p_ratio, air_flow)


    def print(self, graph=False):
        print("")
        print("Fuel cell system characteristics")
        print("===============================================================================")
        self.air_scoop.print()
        self.compressor.print()
        self.stack.print(graph=graph)



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


    def print(self):
        print("")
        print("Pitot scoop characteristics")
        print("----------------------------------------------------------")
        print("Pitot diameter = ", "%.2f"%unit.convert_to("cm",self.diameter), " cm")
        print("Design drag = ", "%.2f"%unit.convert_to("daN",self.design_drag), " daN")
        print("Volume allocation = ", "%.3f"%self.volume)
        print("Mass = ", "%.3f"%self.mass)



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


    def print(self):
        print("")
        print("Compressor characteristics")
        print("----------------------------------------------------------")
        print("Adiabatic efficiency = ", "%.3f"%self.adiabatic_efficiency)
        print("Mechanical efficiency = ", "%.3f"%self.mechanical_efficiency)
        print("Electrical efficiency = ", "%.3f"%self.electrical_efficiency)
        print("")
        print("Design air flow = ", "%.0f"%unit.convert_to("g",self.design_air_flow), " g/s")
        print("Design pressure ratio = ", "%.3f"%self.design_p_ratio)
        print("Design shaft power = ", "%.2f"%unit.convert_to("kW",self.design_shaft_power), " kW")
        print("Design electric power = ", "%.2f"%unit.convert_to("kW",self.design_elec_power), " kW")
        print("")
        print("Volume allocation = ", "%.3f"%self.volume, " m3")
        print("Mass = ", "%.3f"%self.mass, " kg")



class FuelCellPEMLT(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.cell_area = unit.convert_from("cm2", 625)              # m2
        self.max_current_density = 5./unit.convert_from("cm2",1)    # A/m2
        self.cell_entry_total_pressure = unit.convert_from("bar", 1.5)    # Gas pressure at electrode entry
        self.working_temperature = 273.15 + 65                      # Cell working temperature
        self.air_over_feeding = 2                                   # air flow ratio over stoechiometry
        self.power_margin = 0.8                                     # Ratio allowed power over max power
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
        dg = dg0 + self.ideal_gas_constant() * temp * np.log((p_h2*1e-5)*np.sqrt(p_o2*1e-5))

        # Calcul des potentiels de Nernst
        e_std = dg0 / (2*self.faraday_constant())      # [V] - potentiel reversible
        e_rev = dg / (2*self.faraday_constant())       # [V] - potentiel reversible + effet de la de la temp곡teure et de la pression
        e_tn = -delta_h / (2*self.faraday_constant())   # [V] - potentiel thermoneutre

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
        pw_thermal = (e_tn - voltage) * current # puissance thermique

        gas_molar_flow = current / (2 * self.faraday_constant())
        h2_flow = gas_molar_flow * self.h2_molar_mass   # kg/s, Hydrogen mass flow
        air_flow =  (gas_molar_flow / self.air_o2_ratio) * self.air_molar_mass * self.air_over_feeding  # kg/s, Air mass flow

        h2_heat = -delta_h / self.h2_molar_mass
        pw_chemical = h2_flow * h2_heat
        efficiency = pw_elec / pw_chemical

        return {"pwe":pw_elec * nc,
                "pwth":pw_thermal * nc,
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

        self.nominal_thermal_power = dict["pwth"]
        self.nominal_voltage = dict["voltage"]
        self.nominal_current = dict["current"]
        self.nominal_current_density = dict["jj"]
        self.nominal_h2_flow = dict["h2_flow"]
        self.nominal_air_flow = dict["air_flow"]
        self.nominal_efficiency = dict["efficiency"]


    def print(self, graph=False):
        print("")
        print("Fuel Cell stack characteristics")
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
        print("Stack maximum air flow = ", "%.1f"%(self.nominal_air_flow*1000), " g/s")
        print("Stack maximum hydrogen flow = ", "%.2f"%(self.nominal_h2_flow*1000), " g/s")
        print("Stack maximum stack efficiency = ", "%.3f"%self.nominal_efficiency)

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



class LeadingEdgeDissipator(object):

    def __init__(self, fc_system):
        self.fc_system = fc_system

        self.available_span = 9                 # One side
        self.available_curved_length = 2.07     # 2 webs
        self.mean_chord = 0.35

        self.skin_thickness = unit.convert_from("mm",1.5)
        self.skin_conduct = 237.        # W/m/K, Aluminium thermal conductivity
        self.skin_exchange_area = None  # Exchange area with one tube

        self.tube_density = 2700        # Aluminium density
        self.tube_width = unit.convert_from("mm",10)
        self.tube_height = unit.convert_from("mm",6)
        self.tube_thickness = unit.convert_from("mm",0.6)
        self.tube_foot_width = unit.convert_from("mm",3)
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
        rho, cp, mu = self.fc_system.phd.fluid_data(temp, fluid="water")
        rex = (rho * speed / mu) * hydro_width                       # Reynolds number
        cf = 0.5
        for j in range(6):
            cf = (1./(2.*np.log(rex*np.sqrt(cf))/np.log(10)-0.8))**2
            print(cf)
        dp = 0.5 * cf * (tube_length/hydro_width) * rho * speed**2
        return dp

    def design_tubes(self):
        # The fluid makes a loop along the span from and to the nacelle
        self.tube_length = 2 * self.available_span
        self.tube_exchange_area = self.tube_width * self.tube_length
        self.skin_exchange_area = (self.tube_width + 2*self.tube_foot_width) * self.tube_length

        self.tube_footprint_width = self.tube_width + 2*(self.tube_thickness + self.tube_foot_width)
        self.tube_count = np.floor(0.5*self.available_curved_length / self.tube_footprint_width)

        if self.tube_height<0.5*self.tube_width:
            raise Exception("Tube height 'ht' cannot be lower than half tube width 'wt'")
        self.tube_section = 0.125*np.pi*self.tube_width**2 + (self.tube_height-0.5*self.tube_width)*self.tube_width      # Tube section
        rho_f, cp_f, mu_f = self.fc_system.phd.fluid_data(tamb, fluid="water")

        tube_prm = 0.5*np.pi*self.tube_width + 2*(self.tube_height-0.5*self.tube_width) + self.tube_width    # Tube perimeter
        self.tube_hydro_width = 4 * self.tube_section / tube_prm             # Hydrolic section

        self.tube_fluid_mass = self.tube_section * self.tube_length * self.tube_count * rho_f
        self.tube_mass =  (0.5*np.pi*self.tube_width + 2*(self.tube_height-0.5*self.tube_width) + 2*self.tube_foot_width) \
                        * self.tube_thickness * self.tube_length * (2*self.tube_count) * self.tube_density

    def operate_tubes(self, pamb, tamb, air_speed, fluid_speed, fluid_temp_in, fluid_temp_out):
        ha,rhoa,cpa,mua,pra,rea,nua,lbda = self.fc_system.phd.air_thermal_transfer_data(pamb,tamb,air_speed, self.mean_chord)
        hf,rhof,cpf,muf,prf,redf,nudf,lbdf = self.fc_system.phd.fluid_thermal_transfer_data(tamb, fluid_speed, self.tube_hydro_width)
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
                "rhoa":rhoa,
                "cpa":cpa,
                "mua":mua,
                "pra":pra,
                "rea":rea,
                "nua":nua,
                "lbda":lbda,
                "rhof":rhof,
                "cpf":cpf,
                "muf":muf,
                "prf":prf,
                "redf":redf,
                "nudf":nudf,
                "lbdf":lbdf}

    def print(self):
        print("")
        print("Dissipator characteristics")
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



if __name__ == '__main__':

    phd = PhysicalData()

    fc_syst = FuelCellSystem(phd)


    # altp = unit.m_ft(5000)
    # disa = 25
    # vair = unit.mps_kmph(250)
    #
    # pamb, tamb, g = phd.atmosphere(altp, disa)
    #
    # design_power = unit.convert_from("kW", 50)
    #
    # fc_syst.design(pamb, tamb, vair, design_power)
    #
    # fc_syst.print(graph=False)
    #
    #
    # req_power = unit.W_kW(30)
    #
    # dict = fc_syst.operate(pamb, tamb, vair, req_power)
    #
    # print("")
    # print("===============================================================================")
    # print("Stack effective power = ", "%.2f"%unit.kW_W(dict["stack"]["pwe"]), " kW")
    # print("Stack thermal power = ", "%.2f"%unit.kW_W(dict["stack"]["pwth"]), " kW")
    # print("Compressor effective power = ", "%.2f"%unit.kW_W(dict["compressor"]["pwe"]), " kW")
    # print("")
    # print("Voltage = ", "%.1f"%(dict["stack"]["voltage"]), " V")
    # print("Current = ", "%.1f"%(dict["stack"]["current"]), " A")
    # print("Stack current density = ", "%.4f"%(dict["stack"]["jj"]/1e4), " A/cm2")
    # print("Stack effective efficiency = ", "%.4f"%(dict["stack"]["efficiency"]))
    # print("Overall efficiency = ", "%.4f"%(dict["system"]["efficiency"]))
    # print("")
    # print("Peripheral power ratio = ", "%.3f"%(dict["compressor"]["pwe"]/dict["stack"]["pwe"]))
    # print("Compression ratio = ", "%.2f"%(dict["compressor"]["pt_out"]/dict["compressor"]["pt_in"]))
    # print("Compressor output temperature = ", "%.2f"%(dict["compressor"]["tt_out"]-273.15), " C°")


    altp = unit.m_ft(10000)
    disa = 15
    vair = unit.mps_kmph(250)

    pamb, tamb, g = phd.atmosphere(altp, disa)

    fluid_speed = 2. # m/s
    fluid_temp_in = 273.15 + 75
    fluid_temp_out = 273.15 + 70    # Initial value

    fc_syst.dissipator.design_tubes()

    dict_rad = fc_syst.dissipator.operate_tubes(pamb, tamb, vair, fluid_speed, fluid_temp_in, fluid_temp_out)

    fc_syst.dissipator.print()

    print("")
    print("===============================================================================")
    print("Fluid input temperature = ", "%.2f"%dict_rad["temp_in"], " °K")
    print("Fluid output temperature = ", "%.2f"%dict_rad["temp_out"], " °K")
    print("Fluid delta temperature = ", "%.2f"%(dict_rad["temp_in"]-dict_rad["temp_out"]), " °K")
    print("q_fluid = ", "%.2f"%unit.convert_to("kW",dict_rad["q_fluid"]), " kW")
    print("q_out = ", "%.2f"%unit.convert_to("kW",dict_rad["q_out"]), " kW")
    print("Global exchange factor = ", "%.2f"%(dict_rad["ks"]), " W/m2/K")
    print("")
    print("Fluid mass flow = ", "%.2f"%(dict_rad["flow"]), " kg/s")
    print("Fluid pressure drop = ", "%.4f"%unit.convert_to("bar",dict_rad["p_drop"]), " bar")
    print("Fluid flow power = ", "%.0f"%(dict_rad["pw_drop"]), " W")
    print("")
    print("rho air = ", "%.3f"%dict_rad["rhoa"], " kg/m3")
    print("Cp air = ", "%.1f"%dict_rad["cpa"], " J/kg")
    print("mu air = ", "%.1f"%(dict_rad["mua"]*1e6), " 10-6Pa.s")
    print("Pr air = ", "%.4f"%dict_rad["pra"])
    print("Re air = ", "%.0f"%dict_rad["rea"])
    print("Nu air = ", "%.0f"%dict_rad["nua"])
    print("Lambda air = ", "%.4f"%dict_rad["lbda"])
    print("")
    print("rho fluide = ", "%.1f"%dict_rad["rhof"], " kg/m3")
    print("Cp fluide = ", "%.1f"%dict_rad["cpf"], " J/kg")
    print("mu fluide = ", "%.1f"%(dict_rad["muf"]*1e6), " 10-6Pa.s")
    print("Pr fluide = ", "%.4f"%dict_rad["prf"])
    print("ReD fluide = ", "%.0f"%dict_rad["redf"])
    print("NuD fluide = ", "%.2f"%dict_rad["nudf"])
    print("Lambda fluide = ", "%.4f"%dict_rad["lbdf"])


