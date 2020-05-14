#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import fsolve

from marilib.context import earth

from marilib.aircraft.performance import Flight

from marilib.context.math import lin_interp_1d, maximize_1d


class Aerodynamics(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.cx_correction = 0.     # drag correction on cx coefficient
        self.cruise_lodmax = 16.    # Assumption on L/D max for some initializations
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
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)

        self.cruise_lodmax, self.cz_cruise_lodmax = self.lod_max(pamb,tamb,mach)
        self.czmax_conf_clean,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_clean)
        self.czmax_conf_to,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_to)
        self.czmax_conf_ld,Cz0 = self.aircraft.airframe.wing.high_lift(self.hld_conf_ld)

    def drag(self,pamb,tamb,mach,cz):
        """Retrieves airplane drag and L/D in current flying conditions
        """
        # Form & friction drag
        #-----------------------------------------------------------------------------------------------------------
        re = earth.reynolds_number(pamb, tamb, mach)

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


class WeightCg(object):

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
class ThrustData(object):
    def __init__(self, nei=0):
        self.disa = None
        self.altp = None
        self.mach = None
        self.nei = nei
        self.kfn_opt = None


class PowerSystem(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.fuel_density = self.__fuel_density()
        self.data = {"MTO":ThrustData(nei=1),
                     "MCN":ThrustData(nei=1),
                     "MCL":ThrustData(nei=0),
                     "MCR":ThrustData(nei=0)}

    def __fuel_density(self):
        energy_source = self.aircraft.arrangement.energy_source
        return earth.fuel_density(energy_source)

    def thrust_requirement(self):
        self.data["MTO"].disa = self.aircraft.performance.take_off.disa
        self.data["MTO"].altp = self.aircraft.performance.take_off.altp
        self.data["MTO"].mach = self.aircraft.performance.take_off.mach2

        fct = self.aircraft.performance.take_off.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MTO"].kfn_opt = output_dict[0][0]

        self.data["MCN"].disa = self.aircraft.performance.oei_ceiling.disa
        self.data["MCN"].altp = self.aircraft.performance.oei_ceiling.altp
        self.data["MCN"].mach = self.aircraft.performance.oei_ceiling.mach_opt

        fct = self.aircraft.performance.oei_ceiling.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MCN"].kfn_opt = output_dict[0][0]

        self.data["MCL"].disa = self.aircraft.performance.mcl_ceiling.disa
        self.data["MCL"].altp = self.aircraft.performance.mcl_ceiling.altp
        self.data["MCL"].mach = self.aircraft.performance.mcl_ceiling.mach

        fct = self.aircraft.performance.mcl_ceiling.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MCL"].kfn_opt = output_dict[0][0]

        self.data["MCR"].disa = self.aircraft.performance.mcr_ceiling.disa
        self.data["MCR"].altp = self.aircraft.performance.mcr_ceiling.altp
        self.data["MCR"].mach = self.aircraft.performance.mcr_ceiling.mach

        fct = self.aircraft.performance.mcr_ceiling.thrust_opt
        output_dict = fsolve(fct, x0=1., args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.data["MCR"].kfn_opt = output_dict[0][0]

    def thrust_analysis(self):
        raise NotImplementedError

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


class ThrustDataTf(ThrustData):
    def __init__(self, nei):
        super(ThrustDataTf, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.fuel_flow = None
        self.sfc = None
        self.T41 = None


class Turbofan(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(Turbofan, self).__init__(aircraft)

        self.data = {"MTO":ThrustDataTf(nei=1),
                     "MCN":ThrustDataTf(nei=1),
                     "MCL":ThrustDataTf(nei=0),
                     "MCR":ThrustDataTf(nei=0)}

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].fuel_flow = dict["ff"]
            self.data[rating].sfc = dict["sfc"]
            self.data[rating].T41 = dict["t4"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine

        dict = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle)

        fn = dict["fn"]*(n_engine-nei)
        ff = dict["ff"]*(n_engine-nei)
        sfc = ff/fn
        t41 = dict["t4"]

        return {"fn":fn, "ff":ff, "sfc":sfc, "t4":t41}

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fn = thrust/(n_engine - nei)

        dict = self.aircraft.airframe.nacelle.unitary_sc(pamb,tamb,mach,rating,fn)

        return dict

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        return (tas*dict["lod"])/(mass*g*dict["sfc"])

    def specific_breguet_range(self,tow,range,tas,dict):
        g = earth.gravity()
        return tow*(1-np.exp(-(dict["sfc"]*g*range)/(tas*dict["lod"])))

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        return dict["sfc"]*(mass*g/dict["lod"])*time


class ThrustDataTp(ThrustData):
    def __init__(self, nei):
        super(ThrustDataTp, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.power = None
        self.fuel_flow = None
        self.sfc = None
        self.T41 = None


class Turboprop(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(Turboprop, self).__init__(aircraft)

        self.data = {"MTO":ThrustDataTp(nei=1),
                     "MCN":ThrustDataTp(nei=1),
                     "MCL":ThrustDataTp(nei=0),
                     "MCR":ThrustDataTp(nei=0)}

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].power = dict["pw"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].fuel_flow = dict["ff"]
            self.data[rating].sfc = dict["sfc"]
            self.data[rating].T41 = dict["t4"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine

        dict = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle)

        fn = dict["fn"]*(n_engine-nei)
        ff = dict["ff"]*(n_engine-nei)
        pw = dict["pw"]*(n_engine-nei)
        sfc = ff/fn
        t41 = dict["t4"]

        return {"fn":fn, "ff":ff, "pw":pw, "sfc":sfc, "t4":t41}

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fn = thrust/(n_engine - nei)

        dict = self.aircraft.airframe.nacelle.unitary_sc(pamb,tamb,mach,rating,fn)

        return dict

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
        return (eta_prop*dict["lod"])/(mass*g*dict["sfc"])

    def specific_breguet_range(self,tow,range,tas,dict):
        g = earth.gravity()
        eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
        return tow*(1.-np.exp(-(dict["sfc"]*g*range)/(eta_prop*dict["lod"])))

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        eta_prop = self.aircraft.airframe.nacelle.propeller_efficiency
        return ((mass*g*tas*dict["sfc"])/(eta_prop*dict["lod"]))*time


class ThrustDataEf(ThrustData):
    def __init__(self, nei):
        super(ThrustDataEf, self).__init__(nei)
        self.thrust_opt = None
        self.thrust = None
        self.power = None
        self.sec = None


class Electrofan(PowerSystem, Flight):

    def __init__(self, aircraft):
        super(Electrofan, self).__init__(aircraft)

        self.data = {"MTO":ThrustDataEf(nei=1),
                     "MCN":ThrustDataEf(nei=1),
                     "MCL":ThrustDataEf(nei=0),
                     "MCR":ThrustDataEf(nei=0)}

    def thrust_analysis(self):
        self.thrust_requirement()
        for rating in self.data.keys():
            disa = self.data[rating].disa
            altp = self.data[rating].altp
            mach = self.data[rating].mach
            nei = self.data[rating].nei
            kfn = self.data[rating].kfn_opt
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            dict = self.thrust(pamb,tamb,mach,rating, nei=nei)
            self.data[rating].thrust_opt = kfn*dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].thrust = dict["fn"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].power = dict["pw"]/(self.aircraft.airframe.nacelle.n_engine - nei)
            self.data[rating].sec = dict["sec"]

    def thrust(self,pamb,tamb,mach,rating, throttle=1., nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine

        dict = self.aircraft.airframe.nacelle.unitary_thrust(pamb,tamb,mach,rating,throttle=throttle)

        fn = dict["fn"]*(n_engine-nei)
        pw = dict["pw"]*(n_engine-nei)
        sec = pw/fn

        return {"fn":fn, "pw":pw, "sec":sec}

    def sc(self,pamb,tamb,mach,rating, thrust, nei=0):
        """Total thrust of a pure turbofan engine
        """
        n_engine = self.aircraft.airframe.nacelle.n_engine
        fn = thrust/(n_engine - nei)

        dict = self.aircraft.airframe.nacelle.unitary_sc(pamb,tamb,mach,rating,fn)

        return dict

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.aircraft.airframe.wing.area
        nacelle_width = self.aircraft.airframe.nacelle.width

        dCx = 0.12*nacelle_width**2 / wing_area

        return dCx

    def tail_cone_drag_factor(self):
        return 1.

    def specific_air_range(self,mass,tas,dict):
        g = earth.gravity()
        return (tas*dict["lod"])/(mass*g*dict["sec"])

    def specific_breguet_range(self,tow,range,tas,dict):
        g = earth.gravity()
        return tow*g*range*dict["sec"] / (tas*dict["lod"])

    def specific_holding(self,mass,time,tas,dict):
        g = earth.gravity()
        return dict["sec"]*(mass*g/dict["lod"])*time





