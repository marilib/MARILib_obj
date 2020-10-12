#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.utils import earth, unit
from marilib.aircraft.airframe.component import Component
from marilib.aircraft.airframe.model import init_power
from marilib.aircraft.model_config import get_init


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


class SystemWithBattery(Component):

    def __init__(self, aircraft):
        super(SystemWithBattery, self).__init__(aircraft)

        self.wiring_efficiency = get_init(self,"wiring_efficiency")
        self.wiring_pw_density = get_init(self,"wiring_pw_density")

        self.cooling_efficiency = get_init(self,"cooling_efficiency")
        self.cooling_pw_density = get_init(self,"cooling_pw_density")

        self.battery_density = get_init(self,"battery_density")
        self.battery_energy_density = get_init(self,"battery_energy_density")

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

        self.power_chain_mass = (1./self.wiring_pw_density + 1./self.cooling_pw_density) * (elec_power_max * n_engine)

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

        self.wiring_efficiency = get_init(self,"wiring_efficiency")
        self.wiring_pw_density = get_init(self,"wiring_pw_density")

        self.cooling_efficiency = get_init(self,"cooling_efficiency")
        self.cooling_pw_density = get_init(self,"cooling_pw_density")

        self.fuel_cell_pw_density = get_init(self,"fuel_cell_pw_density")
        self.fuel_cell_efficiency = get_init(self,"fuel_cell_efficiency")

        self.global_energy_density = None
        self.power_chain_efficiency = None

        self.fuel_cell_mass = None
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

        self.fuel_cell_mass = (elec_power_max * n_engine)/self.fuel_cell_pw_density
        self.power_chain_mass =   (1./self.wiring_pw_density + 1./self.cooling_pw_density) * (elec_power_max * n_engine) \
                                + self.fuel_cell_mass

        power_elec_cg = 0.70*nacelle_cg + 0.30*body_cg

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

        self.chain_power = get_init(class_name,"chain_power", val=0.2*init_power(aircraft))

        self.battery = get_init(class_name,"battery")
        self.battery_density = get_init(class_name,"battery_density")
        self.battery_energy_density = get_init(class_name,"battery_energy_density")
        self.lto_power = get_init(class_name,"lto_power")
        self.lto_time = get_init(class_name,"lto_time")
        self.cruise_energy = get_init(class_name,"cruise_energy")

        self.generator_efficiency = get_init(class_name,"generator_efficiency")
        self.generator_pw_density = get_init(class_name,"generator_pw_density")

        self.rectifier_efficiency = get_init(class_name,"rectifier_efficiency")
        self.rectifier_pw_density = get_init(class_name,"rectifier_pw_density")

        self.wiring_efficiency = get_init(class_name,"wiring_efficiency")
        self.wiring_pw_density = get_init(class_name,"wiring_pw_density")

        self.cooling_efficiency = get_init(class_name,"cooling_efficiency")
        self.cooling_pw_density = get_init(class_name,"cooling_pw_density")

        self.power_chain_efficiency = None

        self.battery_mass = None
        self.power_chain_mass = None

    def get_power_chain_efficiency(self):
        self.power_chain_efficiency =   self.generator_efficiency * self.rectifier_efficiency \
                                      * self.wiring_efficiency * self.cooling_efficiency \
                                      * self.aircraft.airframe.tail_nacelle.controller_efficiency \
                                      * self.aircraft.airframe.tail_nacelle.motor_efficiency
        return self.power_chain_efficiency

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

        # Power chain is designed assuming each turbofan can feed the total power if required
        self.power_chain_mass = (  1./self.generator_pw_density
                                 + 1./self.rectifier_pw_density
                                 + 1./self.wiring_pw_density
                                 + 1./self.cooling_pw_density) * (elec_power_max * n_engine)
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

        self.chain_power_body = get_init(class_name,"chain_power_body", val=0.1*init_power(aircraft))
        self.chain_power_pod = get_init(class_name,"chain_power_pod", val=0.05*init_power(aircraft))

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]
        self.chain_power = self.chain_power_body + 2.*self.chain_power_pod


class SystemPartialTurboElectricPiggyBack(SystemPartialTurboElectric):

    def __init__(self, aircraft):
        super(SystemPartialTurboElectricPiggyBack, self).__init__(aircraft)

        class_name = "SystemPartialTurboElectric"

        self.chain_power_body = get_init(class_name,"chain_power_body", val=0.1*init_power(aircraft))
        self.chain_power_piggyback = get_init(class_name,"chain_power_piggyback", val=0.1*init_power(aircraft))

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]
        self.chain_power = self.chain_power_body + self.chain_power_piggyback
