#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.utils import earth, unit
from marilib.aircraft.airframe.component import Component
from marilib.aircraft.airframe.model import init_power


class System(Component):

    def __init__(self, aircraft):
        super(System, self).__init__(aircraft)

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        nacelle_cg = self.aircraft.airframe.nacelle.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg

        self.mass = 0.545*mtow**0.8    # global mass of all systems

        self.cg =   0.50*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg

    def get_reference_offtake(self):
        return 0.


class SystemWithBattery(Component):

    def __init__(self, aircraft):
        super(SystemWithBattery, self).__init__(aircraft)

        self.wiring_efficiency = aircraft.get_init(self,"wiring_efficiency")
        self.wiring_pw_density = aircraft.get_init(self,"wiring_pw_density")

        self.cooling_efficiency = aircraft.get_init(self,"cooling_efficiency")
        self.cooling_pw_density = aircraft.get_init(self,"cooling_pw_density")

        self.battery_density = aircraft.get_init(self,"battery_density")
        self.battery_energy_density = aircraft.get_init(self,"battery_energy_density")

        self.power_chain_efficiency = None

        self.power_chain_mass = None

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        nacelle_cg = self.aircraft.airframe.nacelle.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg
        n_engine = self.aircraft.power_system.n_engine

        self.power_chain_efficiency =   self.wiring_efficiency * self.cooling_efficiency \
                                      * self.aircraft.airframe.nacelle.controller_efficiency \
                                      * self.aircraft.airframe.nacelle.motor_efficiency

        elec_power_max = self.aircraft.power_system.reference_power / self.power_chain_efficiency

        self.power_chain_mass = (1./self.wiring_pw_density + 1./self.cooling_pw_density) * elec_power_max

        power_elec_cg = 0.70*nacelle_cg + 0.30*body_cg

        self.mass = 0.545*mtow**0.8  + self.power_chain_mass  # global mass of all systems

        self.cg =   0.40*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg \
                  + 0.10*power_elec_cg


class SystemWithFuelCell(Component):

    def __init__(self, aircraft):
        super(SystemWithFuelCell, self).__init__(aircraft)

        self.wiring_efficiency = aircraft.get_init(self,"wiring_efficiency")
        self.wiring_pw_density = aircraft.get_init(self,"wiring_pw_density")

        self.compressor_over_pressure = aircraft.get_init(self,"compressor_over_pressure")
        self.compressor_efficiency = aircraft.get_init(self,"compressor_efficiency")
        self.compressor_pw_density = aircraft.get_init(self,"compressor_pw_density")

        self.cooling_power_index = aircraft.get_init(self,"cooling_power_index")
        self.cooling_gravimetric_index = aircraft.get_init(self,"cooling_gravimetric_index")

        self.fuel_cell_pw_density = aircraft.get_init(self,"fuel_cell_pw_density")
        self.fuel_cell_efficiency = aircraft.get_init(self,"fuel_cell_efficiency")

        self.fuel_cell_output_power_ref = None
        self.compressor_power_ref = None
        self.cooler_power_ref = None
        self.heat_power_ref = None

        self.power_chain_efficiency = None
        self.global_energy_density = None

        self.fuel_cell_mass = None
        self.compressor_mass = None
        self.cooling_mass = None
        self.power_chain_mass = None

    def eval_fuel_cell_power(self,required_power,pamb,tamb):
        r,gam,Cp,Cv = earth.gas_data()

        n_engine = self.aircraft.power_system.n_engine
        fuel_type = self.aircraft.arrangement.fuel_type
        fuel_heat = earth.fuel_heat(fuel_type)

        # air_mass_flow = fuel_cell_power * relative_air_mass_flow
        st_mass_ratio = earth.stoichiometry("air","hydrogen")
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

    def eval_geometry(self):
        reference_power = self.aircraft.power_system.reference_power
        n_engine = self.aircraft.power_system.n_engine

        self.power_chain_efficiency =   self.wiring_efficiency \
                                      * self.aircraft.airframe.nacelle.controller_efficiency \
                                      * self.aircraft.airframe.nacelle.motor_efficiency

        # Fuell cell stack is designed for take off
        disa = self.aircraft.requirement.take_off.disa
        altp = self.aircraft.requirement.take_off.altp

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        required_power = n_engine * reference_power / self.power_chain_efficiency

        dict = self.eval_fuel_cell_power(required_power,pamb,tamb)

        self.fuel_cell_output_power_ref = dict["fuel_cell_power"]
        self.compressor_power_ref = dict["compressor_power"]
        self.cooling_power_ref = dict["cooling_power"]

        # Heat dissipated by wiring and nacelles must be added to heat dissipated by fuell cells
        self.heat_power_ref = dict["heat_power"] + n_engine*reference_power*(1. - self.wiring_efficiency +
                                                                             1. - self.aircraft.airframe.nacelle.controller_efficiency +
                                                                             1. - self.aircraft.airframe.nacelle.motor_efficiency)

        self.frame_origin = [0., 0., 0.]

    def eval_mass(self):
        reference_power = self.aircraft.power_system.reference_power

        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        nacelle_cg = self.aircraft.airframe.nacelle.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg
        n_engine = self.aircraft.power_system.n_engine

        self.fuel_cell_mass = self.fuel_cell_output_power_ref / self.fuel_cell_pw_density
        self.compressor_mass = self.compressor_power_ref / self.compressor_pw_density
        self.cooling_mass = self.heat_power_ref / self.cooling_gravimetric_index

        self.power_chain_mass =   self.fuel_cell_mass \
                                + self.compressor_mass \
                                + self.fuel_cell_output_power_ref/self.wiring_pw_density \
                                + self.cooling_mass

        power_elec_cg = 0.30*nacelle_cg + 0.70*body_cg

        self.mass = 0.545*mtow**0.8  + self.power_chain_mass  # global mass of all systems

        self.cg =   0.40*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg \
                  + 0.10*power_elec_cg


class SystemWithLaplaceFuelCell(Component):

    def __init__(self, aircraft):
        super(SystemWithLaplaceFuelCell, self).__init__(aircraft)

        self.wiring_efficiency = aircraft.get_init(self,"wiring_efficiency")
        self.wiring_pw_density = aircraft.get_init(self,"wiring_pw_density")





        self.fuel_cell_output_power_ref = None
        self.compressor_power_ref = None
        self.cooler_power_ref = None
        self.heat_power_ref = None

        self.fuel_cell_system_mass = None
        self.cooling_mass = None

    def eval_fuel_cell_power(self,required_power,pamb,tamb):




        return {"fuel_cell_power":fuel_cell_power,
                "compressor_power":compressor_power,
                "cooling_power":cooling_power,
                "heat_power":heat_power,
                "fuel_flow":fuel_flow}

    def eval_geometry(self):
        reference_power = self.aircraft.power_system.reference_power
        n_engine = self.aircraft.power_system.n_engine

        self.power_chain_efficiency =   self.wiring_efficiency \
                                      * self.aircraft.airframe.nacelle.controller_efficiency \
                                      * self.aircraft.airframe.nacelle.motor_efficiency

        required_power = n_engine * reference_power / self.power_chain_efficiency

        # Fuell cell stack is designed for cruise
        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        mass = self.ktow*self.aircraft.weight_cg.mtow

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        dict = self.eval_fuel_cell_power(required_power,pamb,tamb)

        self.fuel_cell_output_power_ref = dict["fuel_cell_power"]
        self.compressor_power_ref = dict["compressor_power"]
        self.cooling_power_ref = dict["cooling_power"]

        # Heat dissipated by wiring and nacelles must be added to heat dissipated by fuell cells
        self.heat_power_ref = dict["heat_power"] + n_engine*reference_power*(1. - self.wiring_efficiency +
                                                                             1. - self.aircraft.airframe.nacelle.controller_efficiency +
                                                                             1. - self.aircraft.airframe.nacelle.motor_efficiency)

        self.frame_origin = [0., 0., 0.]

    def eval_mass(self):
        reference_power = self.aircraft.power_system.reference_power

        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        nacelle_cg = self.aircraft.airframe.nacelle.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg
        n_engine = self.aircraft.power_system.n_engine

        # self.fuel_cell_mass =
        # self.compressor_mass =
        # self.cooling_mass =
        #
        # self.power_chain_mass =

        power_elec_cg = 0.30*nacelle_cg + 0.70*body_cg

        self.mass = 0.545*mtow**0.8  + self.power_chain_mass  # global mass of all systems

        self.cg =   0.40*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg \
                  + 0.10*power_elec_cg


class SystemPartialTurboElectric(Component):

    def __init__(self, aircraft):
        super(SystemPartialTurboElectric, self).__init__(aircraft)

        class_name = "SystemPartialTurboElectric"

        self.chain_power = aircraft.get_init(class_name,"chain_power", val=0.2*init_power(aircraft))

        self.battery = aircraft.get_init(class_name,"battery")
        self.battery_density = aircraft.get_init(class_name,"battery_density")
        self.battery_energy_density = aircraft.get_init(class_name,"battery_energy_density")
        self.lto_power = aircraft.get_init(class_name,"lto_power")
        self.lto_time = aircraft.get_init(class_name,"lto_time")
        self.cruise_energy = aircraft.get_init(class_name,"cruise_energy")

        self.generator_efficiency = aircraft.get_init(class_name,"generator_efficiency")
        self.generator_pw_density = aircraft.get_init(class_name,"generator_pw_density")

        self.rectifier_efficiency = aircraft.get_init(class_name,"rectifier_efficiency")
        self.rectifier_pw_density = aircraft.get_init(class_name,"rectifier_pw_density")

        self.wiring_efficiency = aircraft.get_init(class_name,"wiring_efficiency")
        self.wiring_pw_density = aircraft.get_init(class_name,"wiring_pw_density")

        self.cooling_efficiency = aircraft.get_init(class_name,"cooling_efficiency")
        self.cooling_pw_density = aircraft.get_init(class_name,"cooling_pw_density")

        self.power_chain_efficiency = None

        self.battery_mass = None
        self.power_chain_mass = None

    def get_power_chain_efficiency(self):
        self.power_chain_efficiency =   self.generator_efficiency * self.rectifier_efficiency \
                                      * self.wiring_efficiency * self.cooling_efficiency \
                                      * self.aircraft.airframe.tail_nacelle.controller_efficiency \
                                      * self.aircraft.airframe.tail_nacelle.motor_efficiency
        return self.power_chain_efficiency

    def get_reference_offtake(self):
        # Total power offtake is split between all engines
        reference_offtake =  self.chain_power \
                            /self.get_power_chain_efficiency() \
                            /self.aircraft.power_system.n_engine
        return reference_offtake

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        nacelle_cg = self.aircraft.airframe.nacelle.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg
        n_engine = self.aircraft.power_system.n_engine

        elec_power_max = self.chain_power / self.get_power_chain_efficiency()

        self.power_chain_mass = (  1./self.generator_pw_density
                                 + 1./self.rectifier_pw_density
                                 + 1./self.wiring_pw_density
                                 + 1./self.cooling_pw_density) * elec_power_max

        if self.battery=="yes":
            self.battery_mass = (self.lto_power * self.lto_time + self.cruise_energy) / self.battery_energy_density
            self.power_chain_mass += self.battery_mass
        else:
            self.battery_mass = 0.

        power_elec_cg = 0.70*nacelle_cg + 0.30*body_cg

        self.mass = 0.545*mtow**0.8  + self.power_chain_mass  # global mass of all systems

        self.cg =   0.40*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg \
                  + 0.10*power_elec_cg


class SystemPartialTurboElectricPods(SystemPartialTurboElectric):

    def __init__(self, aircraft):
        super(SystemPartialTurboElectricPods, self).__init__(aircraft)

        class_name = "SystemPartialTurboElectric"

        self.chain_power_body = aircraft.get_init(class_name,"chain_power_body", val=0.1*init_power(aircraft))
        self.chain_power_pod = aircraft.get_init(class_name,"chain_power_pod", val=0.05*init_power(aircraft))

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]
        self.chain_power = self.chain_power_body + 2.*self.chain_power_pod


class SystemPartialTurboElectricPiggyBack(SystemPartialTurboElectric):

    def __init__(self, aircraft):
        super(SystemPartialTurboElectricPiggyBack, self).__init__(aircraft)

        class_name = "SystemPartialTurboElectric"

        self.chain_power_body = aircraft.get_init(class_name,"chain_power_body", val=0.1*init_power(aircraft))
        self.chain_power_piggyback = aircraft.get_init(class_name,"chain_power_piggyback", val=0.1*init_power(aircraft))

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]
        self.chain_power = self.chain_power_body + self.chain_power_piggyback
