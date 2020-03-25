#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import fsolve

import unit
import earth


class Weight_cg(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.mtow = 20500. + 67.e-6*n_pax_ref*design_range
        self.mzfw = 25000. + 41.e-6*n_pax_ref*design_range
        self.mlw = 1.07*self.mzfw
        self.owe = None
        self.mwe = None
        self.mfw = None

    def mass_analysis(self):
        # update all component mass
        self.aircraft.airframe.cabin.eval_mass()
        self.aircraft.airframe.body.eval_mass()
        self.aircraft.airframe.wing.eval_mass()
        self.aircraft.airframe.landing_gear.eval_mass()
        self.aircraft.airframe.cargo.eval_mass()
        self.aircraft.airframe.nacelle.eval_mass()
        self.aircraft.airframe.vertical_stab.eval_mass()
        self.aircraft.airframe.horizontal_stab.eval_mass()
        self.aircraft.airframe.tank.eval_mass()
        self.aircraft.airframe.system.eval_mass()

        # sum all MWE contributions
        self.mwe =   self.aircraft.airframe.cabin.get_mass_mwe() \
                   + self.aircraft.airframe.body.get_mass_mwe() \
                   + self.aircraft.airframe.wing.get_mass_mwe() \
                   + self.aircraft.airframe.landing_gear.get_mass_mwe() \
                   + self.aircraft.airframe.cargo.get_mass_mwe() \
                   + self.aircraft.airframe.nacelle.get_mass_mwe() \
                   + self.aircraft.airframe.vertical_stab.get_mass_mwe() \
                   + self.aircraft.airframe.horizontal_stab.get_mass_mwe() \
                   + self.aircraft.airframe.tank.get_mass_mwe() \
                   + self.aircraft.airframe.system.get_mass_mwe()

        # sum all OWE contributions
        self.owe =   self.aircraft.airframe.cabin.get_mass_owe() \
                   + self.aircraft.airframe.body.get_mass_owe() \
                   + self.aircraft.airframe.wing.get_mass_owe() \
                   + self.aircraft.airframe.landing_gear.get_mass_owe() \
                   + self.aircraft.airframe.cargo.get_mass_owe() \
                   + self.aircraft.airframe.nacelle.get_mass_owe() \
                   + self.aircraft.airframe.vertical_stab.get_mass_owe() \
                   + self.aircraft.airframe.horizontal_stab.get_mass_owe() \
                   + self.aircraft.airframe.tank.get_mass_owe() \
                   + self.aircraft.airframe.system.get_mass_owe()

        if (self.aircraft.arrangement.energy_source=="battery"):
            self.mzfw = self.mtow
        else:
            self.mzfw = self.owe + self.aircraft.airframe.cabin.maximum_payload

        if (self.aircraft.arrangement.energy_source=="battery"):
            self.mlw = self.mtow
        else:
            if (self.aircraft.requirement.n_pax_ref>100):
                self.mlw = min(self.mtow , (1.07*self.mzfw))
            else:
                self.mlw = self.mtow

        # WARNING : for battery powered architecture, MFW corresponds to max battery weight
        self.mfw = min(self.aircraft.airframe.tank.mfw_volume_limited, self.mtow - self.owe)

        # TODO
        # calculer les cg

    def mass_pre_design(self):
        """
        Solve the coupling through MZFW & MLW internally to mass functions
        """
        def fct(x_in):
            self.aircraft.weight_cg.mzfw = x_in[0]
            self.aircraft.weight_cg.mlw = x_in[1]

            self.mass_analysis()

            y_out = np.array([x_in[0] - self.aircraft.weight_cg.mzfw,
                              x_in[1] - self.aircraft.weight_cg.mlw])
            return y_out

        x_ini = np.array([self.aircraft.weight_cg.mzfw,
                          self.aircraft.weight_cg.mlw])

        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.aircraft.weight_cg.mzfw = output_dict[0][0]        # Coupling variable
        self.aircraft.weight_cg.mlw = output_dict[0][1]         # Coupling variable

        self.mass_analysis()


class Aerodynamics(object):

    def __init__(self, aircraft):

        self.hld_conf_to = 0.30
        self.hld_conf_ld = 1.00

    def drag(self,pamb,tamb,mach,cz):
        pass








#--------------------------------------------------------------------------------------------------------------------------------
class Power_system(object):
    """
    Logical aircraft description
    """

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.data = [
                     {"rating": "MTO",
                      "disa": 15.,
                      "altp": 0.,
                      "mach": 0.25,
                      "nei":  0,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "MCN",
                      "disa": 15.,
                      "altp": self.aircraft.requirement.oei[0]["altp"],
                      "mach": self.aircraft.requirement.cruise_mach*0.6,
                      "nei":  1,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "MCL",
                      "disa": 0.,
                      "altp": self.aircraft.requirement.vz_mcl[0]["altp"],
                      "mach": self.aircraft.requirement.cruise_mach,
                      "nei":  0,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "MCR",
                      "disa": 0.,
                      "altp": self.aircraft.requirement.cruise_altp,
                      "mach": self.aircraft.requirement.cruise_mach,
                      "nei":  0,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "FID",
                      "disa": 15.,
                      "altp": self.aircraft.requirement.cruise_altp,
                      "mach": self.aircraft.requirement.cruise_mach,
                      "nei":  0,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None}
                     ]

        self.fuel_heat = self.__fuel_heat__()

    def __fuel_heat__(self):
        energy_source = self.aircraft.arrangement.energy_source
        return earth.fuel_heat(energy_source)

    def thrust_analysis(self):
        for j in range(len(self.data)):
            rating = self.data[j]["rating"]
            disa = self.data[j]["disa"]
            altp = self.data[j]["altp"]
            mach = self.data[j]["mach"]
            nei = self.data[j]["nei"]
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
            fn,ff,sfc = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[j]["thrust"] = fn
            self.data[j]["fuel_flow"] = ff
            self.data[j]["sfc"] = sfc

    def thrust(self,pamb,tamb,mach,rating,throttle,pw_offtake,nei):
        raise NotImplementedError

    def sc(self,pamb,tamb,mach,rating,thrust,pw_offtake,nei):
        raise NotImplementedError

    def oei_drag(self,pamb,tamb):
        raise NotImplementedError


class Turbofan(Power_system):

    def __init__(self, aircraft):
        super(Turbofan, self).__init__(aircraft)

    def thrust(self,pamb,tamb,mach,rating, throttle=1., pw_offtake=0., nei=0):
        """
        Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine

        fn1,ff1 = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle,pw_offtake,nei)

        fn = fn1*(n_engine-nei)
        ff = ff1*(n_engine-nei)
        sfc = ff/fn

        return fn,ff,sfc

    def sc(self,pamb,tamb,mach,rating, thrust, pw_offtake=0., nei=0):
        """
        Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine

        def fct(throttle):
            fn1,ff1 = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle,pw_offtake,nei)
            fn = fn1*(n_engine-nei)
            return thrust-fn

        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        throttle = output_dict[0][0]

        fn1,ff1 = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle,pw_offtake,nei)
        sfc = ff1/fn1

        return sfc,throttle

    def oei_drag(self,pamb,tamb):
        """
        Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx


