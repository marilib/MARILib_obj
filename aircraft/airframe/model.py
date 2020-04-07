#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import fsolve

import aircraft.tool.unit as unit
import earth

from aircraft.performance import Flight

from aircraft.tool.math import lin_interp_1d, maximize_1d


class Aerodynamics(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.cx_correction = 0.     # drag correction on cx coefficient
        self.cruise_lodmax = None
        self.cz_cruise_lodmax = None

        self.hld_conf_clean = 0.
        self.czmax_conf_clean = None

        self.hld_conf_to = 0.30
        self.czmax_conf_to = None

        self.hld_conf_ld = 1.00
        self.czmax_conf_ld = None

    def aerodynamic_analysis(self):
        mach = self.aircraft.requirement.cruise_mach
        altp = self.aircraft.requirement.cruise_altp
        disa = 0.
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)

        self.cruise_lodmax, self.cz_cruise_lodmax = self.lod_max(pamb,tamb,mach)
        self.czmax_conf_clean,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_clean)
        self.czmax_conf_to,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_to)
        self.czmax_conf_ld,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_ld)

    def drag(self,pamb,tamb,mach,cz):
        """Retrieves airplane drag and L/D in current flying conditions
        """
        # Form & friction drag
        #-----------------------------------------------------------------------------------------------------------
        re = earth.reynolds_number(pamb,tamb,mach)

        fac = ( 1. + 0.126*mach**2 )

        ac_nwa = 0.
        cxf = 0.
        for comp in self.aircraft.airframe:
            nwa = comp.get_net_wet_area()
            ael = comp.get_aero_length()
            frm = comp.get_form_factor()
            cxf += frm*((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) * (nwa/self.aircraft.airframe.wing.area)
            ac_nwa += nwa

        # Parasitic drag (seals, antennas, sensors, ...)
        #-----------------------------------------------------------------------------------------------------------
        knwa = ac_nwa/1000.

        kp = (0.0247*knwa - 0.11)*knwa + 0.166       # Parasitic drag factor

        cx_par = cxf*kp

        # Additional drag
        #-----------------------------------------------------------------------------------------------------------
        X = np.array([1.0, 1.5, 2.4, 3.3, 4.0, 5.0])
        Y = np.array([0.036, 0.020, 0.0075, 0.0025, 0., 0.])

        param = self.aircraft.airframe.body.tail_cone_length/self.aircraft.airframe.body.width

        cx_tap_base = lin_interp_1d(param,X,Y)     # Tapered fuselage drag (tail cone)

        cx_tap = cx_tap_base*self.aircraft.power_system.tail_cone_drag_factor()     # Effect of tail cone fan

        # Total zero lift drag
        #-----------------------------------------------------------------------------------------------------------
        cx0 = cxf + cx_par + cx_tap + self.cx_correction

        # Induced drag
        #-----------------------------------------------------------------------------------------------------------
        cxi = self.aircraft.airframe.wing.induced_drag_factor*cz**2  # Induced drag

        # Compressibility drag
        #-----------------------------------------------------------------------------------------------------------
        # Freely inspired from Korn equation
        cz_design = 0.5
        mach_div = self.aircraft.requirement.cruise_mach + (0.03 + 0.1*(cz_design-cz))

        cxc = 0.0025 * np.exp(40.*(mach - mach_div) )

        # Sum up
        #-----------------------------------------------------------------------------------------------------------
        cx = cx0 + cxi + cxc
        lod = cz/cx

        return cx,lod

    def lod_max(self,pamb,tamb,mach):
        """Maximum lift to drag ratio
        """
        def fct(cz):
            cx,lod = self.drag(pamb,tamb,mach,cz)
            return lod

        cz_ini = 0.5
        dcz = 0.05
        cz_lodmax,lodmax,rc = maximize_1d(cz_ini,dcz,[fct])

        return lodmax,cz_lodmax


class Weight_cg(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.mtow = self.__mtow_init__()
        self.mzfw = self.__mzfw_init__()
        self.mlw = self.__mlw_init__()
        self.owe = None
        self.mwe = None
        self.mfw = None

    def __mtow_init__(self):
        return 20500. + 67.e-6*self.aircraft.requirement.n_pax_ref*self.aircraft.requirement.design_range

    def __mzfw_init__(self):
        return 25000. + 41.e-6*self.aircraft.requirement.n_pax_ref*self.aircraft.requirement.design_range

    def __mlw_init__(self):
        return 1.07*(25000. + 41.e-6*self.aircraft.requirement.n_pax_ref*self.aircraft.requirement.design_range)

    def mass_analysis(self):
        """Update all component mass
        """
        for comp in self.aircraft.airframe.mass_iter():
            comp.eval_mass()

        # sum all MWE & OWE contributions
        mwe = 0.
        owe = 0.
        for comp in self.aircraft.airframe.mass_iter():
            mwe += comp.get_mass_mwe()
            owe += comp.get_mass_owe()
        self.mwe = mwe
        self.owe = owe

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
        """Solve the coupling through MZFW & MLW for a given mtow
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


#--------------------------------------------------------------------------------------------------------------------------------
class Power_system(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.fuel_density = self.__fuel_density__()
        self.fuel_heat = self.__fuel_heat__()
        self.data = [
                     {"rating": "MTO",
                      "disa": None,
                      "altp": None,
                      "mach": None,
                      "nei":  1,
                      "thrttl_opt": None,
                      "thrust_req": None,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "MCN",
                      "disa": None,
                      "altp": None,
                      "mach": None,
                      "nei":  1,
                      "thrttl_opt": None,
                      "thrust_req": None,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "MCL",
                      "disa": None,
                      "altp": None,
                      "mach": None,
                      "nei":  0,
                      "thrttl_opt": None,
                      "thrust_req": None,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "MCR",
                      "disa": None,
                      "altp": None,
                      "mach": None,
                      "nei":  0,
                      "thrttl_opt": None,
                      "thrust_req": None,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None},
                     {"rating": "FID",
                      "disa": None,
                      "altp": None,
                      "mach": None,
                      "nei":  0,
                      "thrttl_opt": 1.,
                      "thrust_req": None,
                      "thrust": None,
                      "fuel_flow": None,
                      "sfc": None}
                     ]

    def __fuel_heat__(self):
        energy_source = self.aircraft.arrangement.energy_source
        return earth.fuel_heat(energy_source)

    def __fuel_density__(self):
        energy_source = self.aircraft.arrangement.energy_source
        return earth.fuel_density(energy_source)

    def thrust_analysis(self):
        self.data[0]["disa"] = self.aircraft.performance.take_off.disa
        self.data[0]["altp"] = self.aircraft.performance.take_off.altp
        self.data[0]["mach"] = self.aircraft.performance.take_off.mach2

        fct = self.aircraft.performance.take_off.thrust_req
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data[0]["thrttl_opt"] = output_dict[0][0]

        self.data[1]["disa"] = self.aircraft.performance.oei_ceiling.disa
        self.data[1]["altp"] = self.aircraft.performance.oei_ceiling.altp
        self.data[1]["mach"] = self.aircraft.performance.oei_ceiling.mach_opt

        fct = self.aircraft.performance.oei_ceiling.thrust_req
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data[1]["thrttl_opt"] = output_dict[0][0]

        self.data[2]["disa"] = self.aircraft.performance.mcl_ceiling.disa
        self.data[2]["altp"] = self.aircraft.performance.mcl_ceiling.altp
        self.data[2]["mach"] = self.aircraft.performance.mcl_ceiling.mach

        fct = self.aircraft.performance.mcl_ceiling.thrust_req
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        throttle_1 = output_dict[0][0]
        fct = self.aircraft.performance.oei_ceiling.thrust_req
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        throttle_2 = output_dict[0][0]
        self.data[2]["thrttl_opt"] = max(throttle_1,throttle_2)

        self.data[3]["disa"] = self.aircraft.performance.mcr_ceiling.disa
        self.data[3]["altp"] = self.aircraft.performance.mcr_ceiling.altp
        self.data[3]["mach"] = self.aircraft.performance.mcr_ceiling.mach

        fct = self.aircraft.performance.mcr_ceiling.thrust_req
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data[3]["thrttl_opt"] = output_dict[0][0]

        self.data[4]["disa"] = self.aircraft.performance.mission.disa
        self.data[4]["altp"] = self.aircraft.performance.mission.altp
        self.data[4]["mach"] = self.aircraft.performance.mission.mach

        for j in range(len(self.data)):
            rating = self.data[j]["rating"]
            disa = self.data[j]["disa"]
            altp = self.data[j]["altp"]
            mach = self.data[j]["mach"]
            nei = self.data[j]["nei"]
            throttle = self.data[j]["thrttl_opt"]
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
            fn,ff,sfc = self.thrust(pamb,tamb,mach,rating, throttle=throttle, nei=nei)
            self.data[j]["thrust_req"] = fn/(self.aircraft.airframe.nacelle.n_engine - nei)
            fn,ff,sfc = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[j]["thrust"] = fn/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[j]["fuel_flow"] = ff
            self.data[j]["sfc"] = sfc

    def thrust(self,pamb,tamb,mach,rating,throttle,nei):
        raise NotImplementedError

    def sc(self,pamb,tamb,mach,rating,thrust,nei):
        raise NotImplementedError

    def oei_drag(self,pamb,tamb):
        raise NotImplementedError

    def tail_cone_drag_factor(self):
        raise NotImplementedError

    def breguet_range(self,range,tow,altp,mach,disa):
        raise NotImplementedError


class Turbofan(Power_system, Flight):

    def __init__(self, aircraft):
        super(Turbofan, self).__init__(aircraft)

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine

        fn1,ff1 = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle,nei=nei)

        fn = fn1*(n_engine-nei)
        ff = ff1*(n_engine-nei)
        sfc = ff/fn

        return fn,ff,sfc

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        rating_factor = self.aircraft.airframe.nacelle.rating_factor

        def fct(throttle):
            fn,ff,sfc = self.thrust(pamb,tamb,mach,rating,throttle,nei=nei)
            return thrust-fn

        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        throttle = output_dict[0][0]
        if (throttle>1.):
            print("Throttle is higher than rating, rating = ",rating,"  throttle = ",throttle)

        fn,ff,sfc = self.thrust(pamb,tamb,mach,rating,throttle,nei=nei)

        return sfc,throttle

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def breguet_range(self,range,tow,altp,mach,disa):
        """Breguet range equation is dependant from power source : fuel or battery
        """
        g = earth.gravity()

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        tas = mach*earth.sound_speed(tamb)

        mass = self.aircraft.performance.mission.ktow*tow

        sar,cz,cx,lod,thrust,throttle,sfc = self.level_flight(pamb,tamb,mach,mass)
        fuel = tow*(1-np.exp(-(sfc*g*range)/(tas*lod)))
        time = 1.09*(range/tas)

        return fuel,time


