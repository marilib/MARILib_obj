#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import fsolve
import earth

#--------------------------------------------------------------------------------------------------------------------------------
class Airframe(object):
    """
    Logical aircraft description
    """

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.vtp_area_constraint = None

    def geometry_analysis(self):
        stab_architecture = self.aircraft.arrangement.stab_architecture

        self.cabin.eval_geometry()
        self.body.eval_geometry()
        self.wing.eval_geometry()
        self.cargo.eval_geometry()
        self.nacelle.eval_geometry()

        if (stab_architecture in ["classic","t_tail"]):
            self.vertical_stab.eval_geometry()
            self.horizontal_stab.eval_geometry()
        elif (stab_architecture=="h_tail"):
            self.horizontal_stab.eval_geometry()
            self.vertical_stab.eval_geometry()

        self.tank.eval_geometry()
        self.landing_gear.eval_geometry()
        self.system.eval_geometry()

    def statistical_tail_area(self):
        """
        Compute tail areas using volume coefficients
        """
        self.vertical_stab.eval_area()
        self.horizontal_stab.eval_area()

    def statistical_pre_design(self):
        """
        Solves strong coupling and compute tail areas using volume coefficients
        """
        stab_architecture = self.aircraft.arrangement.stab_architecture

        self.cabin.eval_geometry()
        self.body.eval_geometry()
        self.wing.eval_geometry()
        self.cargo.eval_geometry()
        self.nacelle.eval_geometry()

        def fct(x_in):
            self.vertical_stab.area = x_in[0]                           # Coupling variable
            self.horizontal_stab.area = x_in[1]                             # Coupling variable

            if (stab_architecture in ["classic","t_tail"]):
                self.vertical_stab.eval_geometry()
                self.horizontal_stab.eval_geometry()
            elif (stab_architecture=="h_tail"):
                self.horizontal_stab.eval_geometry()
                self.vertical_stab.eval_geometry()

            self.vertical_stab.eval_area()
            self.horizontal_stab.eval_area()

            y_out = np.array([x_in[0] - self.vertical_stab.area,
                              x_in[1] - self.horizontal_stab.area])
            return y_out

        x_ini = np.array([self.vertical_stab.area,
                          self.horizontal_stab.area])

        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.vertical_stab.area = output_dict[0][0]                           # Coupling variable
        self.horizontal_stab.area = output_dict[0][1]                             # Coupling variable

        if (stab_architecture in ["classic","t_tail"]):
            self.vertical_stab.eval_geometry()
            self.horizontal_stab.eval_geometry()
        elif (stab_architecture=="h_tail"):
            self.horizontal_stab.eval_geometry()
            self.vertical_stab.eval_geometry()

        self.tank.eval_geometry()
        self.landing_gear.eval_geometry()
        self.system.eval_geometry()

    def mass_analysis(self):
        self.cabin.eval_mass()
        self.body.eval_mass()
        self.wing.eval_mass()
        self.landing_gear.eval_mass()
        self.cargo.eval_mass()
        self.nacelle.eval_mass()
        self.vertical_stab.eval_mass()
        self.horizontal_stab.eval_mass()
        self.tank.eval_mass()
        self.system.eval_mass()


class Weight_cg(object):

    def __init__(self, requirement):

        n_pax_ref = requirement.n_pax_ref
        design_range = requirement.design_range

        self.mtow = 20500. + 67.e-6*n_pax_ref*design_range
        self.mzfw = 25000. + 41.e-6*n_pax_ref*design_range
        self.mlw = 1.07*self.mzfw
        self.owe = None
        self.mwe = None


class Aerodynamics(object):

    def __init__(self, requirement):

        self.hld_conf_to = 0.30
        self.hld_conf_ld = 1.00


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
                      "thrust": None},
                     {"rating": "MCN",
                      "disa": 15.,
                      "altp": self.aircraft.requirement.oei[0]["altp"],
                      "mach": self.aircraft.requirement.cruise_mach*0.6,
                      "nei":  1,
                      "thrust": None},
                     {"rating": "MCL",
                      "disa": 0.,
                      "altp": self.aircraft.requirement.vz_mcl[0]["altp"],
                      "mach": self.aircraft.requirement.cruise_mach,
                      "nei":  0,
                      "thrust": None},
                     {"rating": "MCR",
                      "disa": 0.,
                      "altp": self.aircraft.requirement.cruise_altp,
                      "mach": self.aircraft.requirement.cruise_mach,
                      "nei":  0,
                      "thrust": None},
                     {"rating": "FID",
                      "disa": 15.,
                      "altp": self.aircraft.requirement.cruise_altp,
                      "mach": self.aircraft.requirement.cruise_mach,
                      "nei":  0,
                      "thrust": None}
                     ]

    def update_data(self):
        for j in range(len(self.data)):
            rating = self.data[j]["rating"]
            disa = self.data[j]["disa"]
            altp = self.data[j]["altp"]
            mach = self.data[j]["mach"]
            nei = self.data[j]["nei"]
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
            fn,ff,sfc = self.thrust(self.aircraft,pamb,tamb,mach,rating, throttle=1., pw_offtake=0., nei=nei)
            self.data[j]["thrust"] = fn

    def thrust(self):
        raise NotImplementedError

    def sc(self):
        raise NotImplementedError

    def oei_drag(self):
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




#--------------------------------------------------------------------------------------------------------------------------------
class Aircraft(object):
    """
    Logical aircraft description
    """
    def __init__(self, name, requirement, arrangement):
        """
        Data structure, only one sub-level allowed
        """
        self.name = name
        self.requirement = requirement
        self.arrangement = arrangement

        self.airframe = Airframe(self)

        self.payload = None
        self.power_system = None
        self.aerodynamics = Aerodynamics(requirement)
        self.weight_cg = Weight_cg(requirement)
        self.economics = None
        self.environment = None




