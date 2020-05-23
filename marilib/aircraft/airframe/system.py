#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.utils import earth, unit

from marilib.aircraft.airframe.component import Component

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

        self.cooling_pw_density = get_init(self,"cooling_pw_density")

        self.battery_density = get_init(self,"battery_density")
        self.battery_energy_density = get_init(self,"battery_energy_density")

        self.power_chain_efficiency = None

        self.power_elec_mass = None

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
        n_engine = self.aircraft.airframe.nacelle.n_engine

        self.power_chain_efficiency =   self.wiring_efficiency \
                                      * self.aircraft.airframe.nacelle.controller_efficiency \
                                      * self.aircraft.airframe.nacelle.motor_efficiency

        elec_power_max = self.aircraft.airframe.nacelle.reference_power / self.power_chain_efficiency

        self.power_elec_mass = (1./self.wiring_pw_density + 1./self.cooling_pw_density) * (elec_power_max * n_engine)

        power_elec_cg = 0.70*nacelle_cg + 0.30*body_cg

        self.mass = 0.545*mtow**0.8  + self.power_elec_mass  # global mass of all systems

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

        self.cooling_pw_density = get_init(self,"cooling_pw_density")

        self.fuel_cell_pw_density = get_init(self,"fuel_cell_pw_density")
        self.fuel_cell_efficiency = get_init(self,"fuel_cell_efficiency")

        self.global_energy_density = None
        self.power_chain_efficiency = None

        self.fuel_cell_mass = None
        self.power_elec_mass = None

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
        n_engine = self.aircraft.airframe.nacelle.n_engine

        self.power_chain_efficiency =   self.wiring_efficiency \
                                      * self.aircraft.airframe.nacelle.controller_efficiency \
                                      * self.aircraft.airframe.nacelle.motor_efficiency

        elec_power_max = self.aircraft.airframe.nacelle.reference_power / self.power_chain_efficiency

        self.fuel_cell_mass = (elec_power_max * n_engine)/self.fuel_cell_pw_density
        self.power_elec_mass =   (1./self.wiring_pw_density + 1./self.cooling_pw_density) * (elec_power_max * n_engine) \
                               + self.fuel_cell_mass

        power_elec_cg = 0.70*nacelle_cg + 0.30*body_cg

        self.mass = 0.545*mtow**0.8  + self.power_elec_mass  # global mass of all systems

        self.cg =   0.40*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg \
                  + 0.10*nacelle_cg \
                  + 0.10*power_elec_cg

