#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from context import earth, unit

import numpy as np
from scipy.optimize import fsolve

from aircraft.tool.math import vander3, trinome, maximize_1d


class Performance(object):
    """
    Master class for all aircraft performances
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.mission = None
        self.take_off = Take_off(aircraft)
        self.approach = Approach(aircraft)
        self.mcr_ceiling = MCR_ceiling(aircraft)
        self.mcl_ceiling = MCL_ceiling(aircraft)
        self.oei_ceiling = OEI_ceiling(aircraft)
        self.time_to_climb = Time_to_Climb(aircraft)

    def analysis(self):
        """Evaluate general performances of the airplane
        """
        #---------------------------------------------------------------------------------------------------
        self.mission.eval_cruise_point()

        #---------------------------------------------------------------------------------------------------
        self.take_off.disa = self.aircraft.requirement.take_off.disa
        self.take_off.altp = self.aircraft.requirement.take_off.altp
        self.take_off.kmtow = self.aircraft.requirement.take_off.kmtow
        self.take_off.kvs1g = self.aircraft.requirement.take_off.kvs1g
        self.take_off.s2_min_path = self.aircraft.requirement.take_off.s2_min_path
        self.take_off.hld_conf = self.aircraft.performance.take_off.hld_conf
        self.take_off.tofl_req = self.aircraft.requirement.take_off.tofl_req

        rating = "MTO"
        kfn = 1.
        mass = self.take_off.kmtow*self.aircraft.weight_cg.mtow
        to_dict = self.take_off.eval(self.take_off.disa, self.take_off.altp, mass, self.take_off.hld_conf,
                                     rating, kfn, self.take_off.kvs1g, self.take_off.s2_min_path)

        self.take_off.tofl_eff = to_dict["tofl"]
        self.take_off.kvs1g_eff = to_dict["kvs1g"]
        self.take_off.s2_path = to_dict["path"]
        self.take_off.v2 = to_dict["v2"]
        self.take_off.mach2 = to_dict["mach2"]
        self.take_off.limit = to_dict["limit"]

        #---------------------------------------------------------------------------------------------------
        self.approach.disa = self.aircraft.requirement.approach.disa
        self.approach.altp = self.aircraft.requirement.approach.altp
        self.approach.kmlw = self.aircraft.requirement.approach.kmlw
        self.approach.kvs1g = self.aircraft.requirement.approach.kvs1g
        self.approach.hld_conf = self.aircraft.performance.approach.hld_conf
        self.approach.app_speed_req = self.aircraft.requirement.approach.app_speed_req

        mass = self.approach.kmlw*self.aircraft.weight_cg.mlw
        ld_dict = self.approach.eval(self.approach.disa,self.approach.altp,mass,self.approach.hld_conf,self.approach.kvs1g)

        self.approach.app_speed_eff = ld_dict["vapp"]

        #---------------------------------------------------------------------------------------------------
        self.mcl_ceiling.disa = self.aircraft.requirement.mcl_ceiling.disa
        self.mcl_ceiling.altp = self.aircraft.requirement.mcl_ceiling.altp
        self.mcl_ceiling.mach = self.aircraft.requirement.mcl_ceiling.mach
        self.mcl_ceiling.kmtow = self.aircraft.requirement.mcl_ceiling.kmtow
        self.mcl_ceiling.rating = self.aircraft.requirement.mcl_ceiling.rating
        self.mcl_ceiling.speed_mode = self.aircraft.requirement.mcl_ceiling.speed_mode
        self.mcl_ceiling.vz_req = self.aircraft.requirement.mcl_ceiling.vz_req

        kfn = 1.
        mass = self.mcl_ceiling.kmtow*self.aircraft.weight_cg.mtow
        cl_dict = self.mcl_ceiling.eval(self.mcl_ceiling.disa,self.mcl_ceiling.altp,self.mcl_ceiling.mach,mass,
                                        self.mcl_ceiling.rating,kfn,self.mcl_ceiling.speed_mode)

        self.mcl_ceiling.vz_eff = cl_dict["vz"]

        #---------------------------------------------------------------------------------------------------
        self.mcr_ceiling.disa = self.aircraft.requirement.mcr_ceiling.disa
        self.mcr_ceiling.altp = self.aircraft.requirement.mcr_ceiling.altp
        self.mcr_ceiling.mach = self.aircraft.requirement.mcr_ceiling.mach
        self.mcr_ceiling.kmtow = self.aircraft.requirement.mcr_ceiling.kmtow
        self.mcr_ceiling.rating = self.aircraft.requirement.mcr_ceiling.rating
        self.mcr_ceiling.speed_mode = self.aircraft.requirement.mcr_ceiling.speed_mode
        self.mcr_ceiling.vz_req = self.aircraft.requirement.mcr_ceiling.vz_req

        kfn = 1.
        mass = self.mcr_ceiling.kmtow*self.aircraft.weight_cg.mtow
        cl_dict = self.mcr_ceiling.eval(self.mcr_ceiling.disa,self.mcr_ceiling.altp,self.mcr_ceiling.mach,mass,
                                        self.mcr_ceiling.rating,kfn,self.mcr_ceiling.speed_mode)

        self.mcr_ceiling.vz_eff = cl_dict["vz"]

        #---------------------------------------------------------------------------------------------------
        self.oei_ceiling.disa = self.aircraft.requirement.oei_ceiling.disa
        self.oei_ceiling.altp = self.aircraft.requirement.oei_ceiling.altp
        self.oei_ceiling.kmtow = self.aircraft.requirement.oei_ceiling.kmtow
        self.oei_ceiling.rating = self.aircraft.requirement.oei_ceiling.rating
        self.oei_ceiling.speed_mode = self.aircraft.requirement.oei_ceiling.speed_mode
        self.oei_ceiling.path_req = self.aircraft.requirement.oei_ceiling.path_req

        kfn = 1.
        mass = self.oei_ceiling.kmtow*self.aircraft.weight_cg.mtow
        ei_dict = self.oei_ceiling.eval(self.oei_ceiling.disa,self.oei_ceiling.altp,mass,self.oei_ceiling.rating,kfn,self.oei_ceiling.speed_mode)

        self.oei_ceiling.path_eff = ei_dict["path"]
        self.oei_ceiling.mach_opt = ei_dict["mach"]

        #---------------------------------------------------------------------------------------------------
        self.time_to_climb.disa = self.aircraft.requirement.time_to_climb.disa
        self.time_to_climb.cas1 = self.aircraft.requirement.time_to_climb.cas1
        self.time_to_climb.altp1 = self.aircraft.requirement.time_to_climb.altp1
        self.time_to_climb.cas2 = self.aircraft.requirement.time_to_climb.cas2
        self.time_to_climb.altp2 = self.aircraft.requirement.time_to_climb.altp2
        self.time_to_climb.mach = self.aircraft.requirement.time_to_climb.mach
        self.time_to_climb.altp = self.aircraft.requirement.time_to_climb.altp
        self.time_to_climb.ttc_req = self.aircraft.requirement.time_to_climb.ttc_req

        rating = "MCL"
        kfn = 1.
        mass = self.aircraft.weight_cg.mtow
        tc_dict = self.time_to_climb.eval(self.time_to_climb.disa,self.time_to_climb.altp,self.time_to_climb.mach,mass,
                                          self.time_to_climb.altp1,self.time_to_climb.cas1,self.time_to_climb.altp2,
                                          self.time_to_climb.cas2,rating,kfn)

        self.time_to_climb.ttc_eff = tc_dict["ttc"]


class Flight(object):
    """Usefull methods for all simulation
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

    def get_speed(self,pamb,speed_mode,mach):
        """retrieve CAS or Mach from mach depending on speed_mode
        """
        speed = {"cas" : earth.vcas_from_mach(pamb, mach),  # CAS required
                 "mach" : mach  # mach required
                 }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return speed

    def get_mach(self,pamb,speed_mode,speed):
        """Retrieve Mach from CAS or mach depending on speed_mode
        """
        mach = {"cas" : earth.mach_from_vcas(pamb, speed),  # Input is CAS
                "mach" : speed  # Input is mach
                }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return mach

    def speed_from_lift(self,pamb,tamb,cz,mass):
        """Retrieve mach from cz using simplified lift equation
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()
        mach = np.sqrt((mass*g)/(0.5*gam*pamb*self.aircraft.airframe.wing.area*cz))
        return mach

    def lift_from_speed(self,pamb,tamb,mach,mass):
        """Retrieve cz from mach using simplified lift equation
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()
        cz = (2.*mass*g)/(gam*pamb*mach**2*self.aircraft.airframe.wing.area)
        return cz

    def level_flight(self,pamb,tamb,mach,mass):
        """Level flight equilibrium
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()
        vsnd = earth.sound_speed(tamb)

        cz = (2.*mass*g)/(gam*pamb*mach**2*self.aircraft.airframe.wing.area)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        thrust = (gam/2.)*pamb*mach**2*self.aircraft.airframe.wing.area*cx
        dict = self.aircraft.power_system.sc(pamb,tamb,mach,"MCR",thrust)
        throttle = dict.get("thtl")
        sfc = dict.get("sfc")
        if sfc is not None:
            sar = (vsnd*mach*lod)/(mass*g*sfc)
            return {"sar":sar, "cz":cz, "cx":cx, "lod":lod, "fn":thrust, "thtl":throttle, "sfc":sfc}
        else:
            sec = dict.get("sec")
            sar = (vsnd*mach*lod)/(mass*g*sec)
            return {"sar":sar, "cz":cz, "cx":cx, "lod":lod, "fn":thrust, "thtl":throttle, "sec":sec}

    def air_path(self,nei,altp,disa,speed_mode,speed,mass,rating,kfn):
        """Retrieve air path in various conditions
        """
        g = earth.gravity()
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        mach = self.get_mach(pamb,speed_mode,speed)

        dict = self.aircraft.power_system.thrust(pamb,tamb,mach,rating)
        fn = dict["fn"]*kfn
        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        if(nei>0):
            dcx = self.aircraft.power_system.oei_drag(pamb,mach)
            cx = cx + dcx*nei
            lod = cz/cx

        acc_factor = earth.climb_mode(speed_mode, mach, dtodz, tstd, disa)
        slope = ( fn/(mass*g) - 1./lod ) / acc_factor
        vz = slope * mach * earth.sound_speed(tamb)
        return slope,vz

    def max_air_path(self,nei,altp,disa,speed_mode,mass,rating,kfn):
        """Optimize the speed of the aircraft to maximize the air path
        """
        def fct(cz):
            pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
            mach = self.speed_from_lift(pamb,tamb,cz,mass)
            speed = self.get_speed(pamb,speed_mode,mach)
            slope,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
            if isformax: return slope
            else: return slope,vz,mach

        cz_ini = 0.5
        dcz = 0.05

        isformax = True
        cz,slope,rc = maximize_1d(cz_ini,dcz,[fct])
        isformax = False

        slope,vz,mach = fct(cz)
        return slope,vz,mach,cz

    def propulsion_ceiling(self,altp_ini,nei,vzreq,disa,speed_mode,speed,mass,rating,throttle):
        """Optimize the speed of the aircraft to maximize the air path
        """
        def fct_prop_ceiling(altp,nei,vzreq,disa,speed_mode,speed,mass,rating):
            [slope,vz] = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,throttle)
            delta_vz = vz - vzreq
            return delta_vz

        fct_arg = (nei,vzreq,disa,speed_mode,speed,mass,rating)

        output_dict = fsolve(fct_prop_ceiling, x0 = altp_ini, args=fct_arg, full_output = True)

        altp = output_dict[0][0]
        rei = output_dict[2]
        if(rei!=1): altp = np.NaN

        return altp, rei

    def eval_sar(self,altp,mass,mach,disa):
        """Evaluate Specific Air Range
        """
        g = earth.gravity()
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        vsnd = earth.sound_speed(tamb)

        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        nei = 0
        thrust = mass*g / lod
        dict = self.aircraft.power_system.sc(pamb,tamb,mach,"MCR",thrust,nei)
        throttle = dict.get("thtl")
        sfc = dict.get("sfc")
        if sfc is not None:
            sar = (vsnd*mach*lod)/(mass*g*sfc)
            return {"sar":sar, "cz":cz, "cx":cx, "lod":lod, "fn":thrust, "thtl":throttle, "sfc":sfc}
        else:
            sec = dict.get("sec")
            sar = (vsnd*mach*lod)/(mass*g*sec)
            return {"sar":sar, "cz":cz, "cx":cx, "lod":lod, "fn":thrust, "thtl":throttle, "sec":sec}

    def eval_max_sar(self,mass,mach,disa):

        def fct(altp,mass,mach,disa):
            dict = self.eval_sar(altp,mass,mach,disa)
            return dict["sar"]

        d_altp = 250.
        altp_ini = self.aircraft.requirement.cruise_altp
        fct = [fct, mass,mach,disa]

        altp_sar_max,sar_max,rc = maximize_1d(altp_ini,d_altp,fct)
        dict = self.eval_sar(altp_sar_max,mass,mach,disa)
        sfc = dict.get("sfc")
        if sfc is not None:
            [sar,cz,cx,lod,fn,thtl,sfc] = dict.values()
            return {"altp":altp_sar_max, "sar":sar, "cz":cz, "cx":cx, "lod":lod, "fn":fn, "thtl":thtl, "sfc":sfc}
        else:
            [sar,cz,cx,lod,fn,thtl,sec] = dict.values()
            return {"altp":altp_sar_max, "sar":sar, "cz":cz, "cx":cx, "lod":lod, "fn":fn, "thtl":thtl, "sec":sec}

    def acceleration(self,nei,altp,disa,speed_mode,speed,mass,rating,throttle):
        """Aircraft acceleration on level flight
        """
        r,gam,Cp,Cv = earth.gas_data()

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        mach = self.get_mach(pamb,speed_mode,speed)

        dict = self.aircraft.power_system.thrust(pamb,tamb,mach,rating,throttle=throttle,nei=nei)

        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        if(nei>0):
            dcx = self.aircraft.power_system.oei_drag(pamb,mach)
            cx = cx + dcx*nei

        acc = (dict["fn"] - 0.5*gam*pamb*mach**2*self.aircraft.airframe.wing.area*cx) / mass
        return acc


class Take_off(Flight):
    """Take Off Field Length
    """
    def __init__(self, aircraft):
        super(Take_off, self).__init__(aircraft)
        self.aircraft = aircraft

        self.disa = None
        self.altp = None
        self.kmtow = None
        self.kvs1g = None
        self.s2_min_path = None
        self.tofl_req = None
        self.tofl_eff = None
        self.hld_conf = self.aircraft.aerodynamics.hld_conf_to
        self.kvs1g_eff = None
        self.v2 = None
        self.mach2 = None
        self.s2_path = None
        self.limit = None

    def thrust_opt(self,kfn):
        mass = self.kmtow*self.aircraft.weight_cg.mtow
        dict = self.eval(self.disa,self.altp,mass,self.hld_conf,"MTO",kfn,self.kvs1g,self.s2_min_path)
        return self.tofl_req/dict["tofl"] - 1.

    def eval(self,disa,altp,mass,hld_conf,rating,kfn,kvs1g,s2_min_path):
        """Take off field length and climb path with eventual kVs1g increase to recover min regulatory slope
        """
        tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf,rating,kfn)

        if(s2_min_path<s2_path):
            limitation = "fl"   # field length
        else:
            dkvs1g = 0.005
            kvs1g_ = np.array([0.,0.])
            kvs1g_[0] = kvs1g
            kvs1g_[1] = kvs1g_[0] + dkvs1g

            s2_path_ = np.array([0.,0.])
            s2_path_[0] = s2_path
            tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf,rating,kfn)

            while(s2_path_[0]<s2_path_[1] and s2_path_[1]<s2_min_path):
                kvs1g_[0] = kvs1g_[1]
                kvs1g_[1] = kvs1g_[1] + dkvs1g
                tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf,rating,kfn)

            if(s2_min_path<s2_path_[1]):
                kvs1g = kvs1g_[0] + ((kvs1g_[1]-kvs1g_[0])/(s2_path_[1]-s2_path_[0]))*(s2_min_path-s2_path_[0])
                tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf,rating,kfn)
                s2_path = s2_min_path
                limitation = "s2"   # second segment
            else:
                tofl = np.nan
                kvs1g = np.nan
                s2_path = 0.
                limitation = None

        to_dict = {"tofl":tofl, "kvs1g":kvs1g, "path":s2_path, "v2":cas, "mach2":mach, "limit":limitation}

        return to_dict

    def take_off(self,kvs1g,altp,disa,mass,hld_conf,rating,kfn):
        """Take off field length and climb path at 35 ft depending on stall margin (kVs1g)
        """
        czmax,cz0 = self.aircraft.airframe.wing.high_lift(hld_conf)

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        rho,sig = earth.air_density(pamb, tamb)

        cz_to = czmax / kvs1g**2
        mach = self.speed_from_lift(pamb,tamb,cz_to,mass)

        nei = 0    # For Magic Line factor computation
        throttle = 1.
        dict = self.aircraft.power_system.thrust(pamb,tamb,mach,rating,throttle,nei)
        fn = kfn*dict["fn"]

        ml_factor = mass**2 / (cz_to*fn*self.aircraft.airframe.wing.area*sig**0.8 )  # Magic Line factor
        tofl = 15.5*ml_factor + 100.    # Magic line

        nei = 1             # For 2nd segment computation
        speed_mode = "cas"  # Constant CAS
        speed = self.get_speed(pamb,speed_mode,mach)

        s2_path,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,"MTO",kfn)

        return tofl,s2_path,speed,mach


class Approach():
    """Approach speed
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.disa = None
        self.altp = None
        self.kmlw = None
        self.kvs1g = None
        self.app_speed_req = None
        self.app_speed_eff = None
        self.hld_conf = self.aircraft.aerodynamics.hld_conf_ld

    def eval(self,disa,altp,mass,hld_conf,kvs1g):
        """Minimum approach speed (VLS)
        """
        g = earth.gravity()
        czmax,cz0 = self.aircraft.airframe.wing.high_lift(hld_conf)
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        rho,sig = earth.air_density(pamb, tamb)
        vapp = np.sqrt((mass*g) / (0.5*rho*self.aircraft.airframe.wing.area*(czmax / kvs1g**2)))
        return {"vapp":vapp}


class MCL_ceiling(Flight):
    """Propulsion ceiling in MCL rating
    """
    def __init__(self, aircraft):
        super(MCL_ceiling, self).__init__(aircraft)
        self.aircraft = aircraft

        self.disa = None
        self.altp = None
        self.mach = None
        self.kmtow = None
        self.rating = None
        self.speed_mode = None
        self.vz_req = None
        self.vz_eff = None

    def thrust_opt(self,kfn):
        mass = self.kmtow*self.aircraft.weight_cg.mtow
        nei = 0
        pamb,tamb,tstd,dtodz = earth.atmosphere(self.altp, self.disa)
        speed = self.get_speed(pamb,self.speed_mode,self.mach)
        slope,vz = self.air_path(nei,self.altp,self.disa,self.speed_mode,speed,mass,self.rating,kfn)
        return vz - self.vz_req

    def eval(self,disa,altp,mach,mass,rating,kfn,speed_mode):
        """Residual climb speed in MCL rating
        """
        nei = 0
        pamb,tamb,tstd,dtodz = earth.atmosphere(self.altp, self.disa)
        speed = self.get_speed(pamb,self.speed_mode,self.mach)
        slope,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
        return {"vz":vz, "slope":slope}


class MCR_ceiling(Flight):
    """Propulsion ceiling in MCR rating
    """
    def __init__(self, aircraft):
        super(MCR_ceiling, self).__init__(aircraft)
        self.aircraft = aircraft

        self.disa = None
        self.altp = None
        self.mach = None
        self.kmtow = None
        self.rating = None
        self.speed_mode = None
        self.vz_req = None
        self.vz_eff = None

    def thrust_opt(self,kfn):
        mass = self.kmtow*self.aircraft.weight_cg.mtow
        nei = 0
        pamb,tamb,tstd,dtodz = earth.atmosphere(self.altp, self.disa)
        speed = self.get_speed(pamb,self.speed_mode,self.mach)
        slope,vz = self.air_path(nei,self.altp,self.disa,self.speed_mode,speed,mass,self.rating,kfn)
        return vz - self.vz_req

    def eval(self,disa,altp,mach,mass,rating,kfn,speed_mode):
        """Residual climb speed in MCR rating
        """
        nei = 0
        pamb,tamb,tstd,dtodz = earth.atmosphere(self.altp, self.disa)
        speed = self.get_speed(pamb,self.speed_mode,self.mach)
        slope,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
        return {"vz":vz, "slope":slope}


class OEI_ceiling(Flight):
    """Definition of all mission types
    """
    def __init__(self, aircraft):
        super(OEI_ceiling, self).__init__(aircraft)
        self.aircraft = aircraft

        self.disa = None
        self.altp = None
        self.kmtow = None
        self.rating = None
        self.speed_mode = None
        self.path_req = None
        self.path_eff = None
        self.mach_opt = None

    def thrust_opt(self,kfn):
        mass = self.kmtow*self.aircraft.weight_cg.mtow
        nei = 1.
        pamb,tamb,tstd,dtodz = earth.atmosphere(self.altp, self.disa)
        speed = self.get_speed(pamb,self.speed_mode,self.mach_opt)
        path,vz = self.air_path(nei,self.altp,self.disa,self.speed_mode,speed,mass,self.rating,kfn)
        return path - self.path_req

    def eval(self,disa,altp,mass,rating,kfn,speed_mode):
        """Compute one engine inoperative maximum path
        """
        nei = 1.
        path,vz,mach,cz = self.max_air_path(nei,altp,disa,speed_mode,mass,rating,kfn)
        return {"path":path, "vz":vz, "mach":mach, "cz":cz}


class Time_to_Climb(Flight):
    """
    Definition of all mission types
    """
    def __init__(self, aircraft):
        super(Time_to_Climb, self).__init__(aircraft)
        self.aircraft = aircraft

        self.disa = None
        self.altp = None
        self.mach = None
        self.kmtow = None
        self.cas1 = None
        self.altp1 = None
        self.cas2 = None
        self.altp2 = None
        self.altp = None
        self.ttc_req = None
        self.ttc_eff = None

    def thrust_opt(self,kfn):
        mass = self.kmtow*self.aircraft.weight_cg.mtow
        dict = self.eval(self.disa,self.altp,self.mach,mass,self.altp1,self.cas1,self.altp2,self.cas2,"MCL",kfn)
        return dict["ttc"]/self.ttc_req - 1.

    def eval(self,disa,toc,mach,mass,altp1,vcas1,altp2,vcas2,rating,kfn):
        """
        Time to climb to initial cruise altitude
        For simplicity reasons, airplane mass is supposed constant
        """
        if(vcas1> unit.mps_kt(250.)):
            print("vcas1 = ", unit.kt_mps(vcas1))
            print("vcas1 must be lower than or equal to 250kt")
        if(vcas1>vcas2):
            print("vcas1 = ", unit.kt_mps(vcas1))
            print("vcas2 = ", unit.kt_mps(vcas2))
            print("vcas1 must be lower than or equal to vcas2")

        cross_over_altp = earth.cross_over_altp(vcas2, mach)

        if(cross_over_altp<altp1):
            print("Cross over altitude is too low")

        if(toc<cross_over_altp):
            cross_over_altp = toc

        # Duration of initial climb
        #-----------------------------------------------------------------------------------------------------------
        altp_0 = altp1
        altp_2 = altp2
        altp_1 = (altp_0+altp_2)/2.
        altp = np.array([altp_0, altp_1, altp_2])

        nei = 0
        speed_mode = "cas"    # Constant CAS

        slope,v_z0 = self.air_path(nei,altp[0],disa,speed_mode,vcas1,mass,rating,kfn)
        slope,v_z1 = self.air_path(nei,altp[1],disa,speed_mode,vcas1,mass,rating,kfn)
        slope,v_z2 = self.air_path(nei,altp[2],disa,speed_mode,vcas1,mass,rating,kfn)
        v_z = np.array([v_z0, v_z1, v_z2])

        if (v_z[0]<0. or v_z[1]<0. or v_z[2]<0.):
            print("Climb to acceleration altitude is not possible")

        A = vander3(altp)
        B = 1./v_z
        C = trinome(A,B)

        time1 = ((C[0]*altp[2]/3. + C[1]/2.)*altp[2] + C[2])*altp[2]
        time1 = time1 - ((C[0]*altp[0]/3. + C[1]/2.)*altp[0] + C[2])*altp[0]

        # Acceleration
        #-----------------------------------------------------------------------------------------------------------
        vc0 = vcas1
        vc2 = vcas2
        vc1 = (vc0+vc2)/2.
        vcas = np.array([vc0, vc1, vc2])

        acc0 = self.acceleration(nei,altp[2],disa,speed_mode,vcas[0],mass,rating,kfn)
        acc1 = self.acceleration(nei,altp[2],disa,speed_mode,vcas[1],mass,rating,kfn)
        acc2 = self.acceleration(nei,altp[2],disa,speed_mode,vcas[2],mass,rating,kfn)
        acc = np.array([acc0, acc1, acc2])

        if(acc[0]<0. or acc[1]<0. or acc[2]<0.):
            print("Acceleration is not possible")

        A = vander3(vcas)
        B = 1./acc
        C = trinome(A,B)

        time2 = ((C[0]*vcas[2]/3. + C[1]/2.)*vcas[2] + C[2])*vcas[2]
        time2 = time2 - ((C[0]*vcas[0]/3. + C[1]/2.)*vcas[0] + C[2])*vcas[0]

        # Duration of climb to cross over
        #-----------------------------------------------------------------------------------------------------------
        altp_0 = altp2
        altp_2 = cross_over_altp
        altp_1 = (altp_0+altp_2)/2.
        altp = np.array([altp_0, altp_1, altp_2])

        slope,v_z0 = self.air_path(nei,altp[0],disa,speed_mode,vcas2,mass,rating,kfn)
        slope,v_z1 = self.air_path(nei,altp[1],disa,speed_mode,vcas2,mass,rating,kfn)
        slope,v_z2 = self.air_path(nei,altp[2],disa,speed_mode,vcas2,mass,rating,kfn)
        v_z = np.array([v_z0, v_z1, v_z2])

        if(v_z[0]<0. or v_z[1]<0. or v_z[2]<0.):
            print("Climb to cross over altitude is not possible")

        A = vander3(altp)
        B = 1./v_z
        C = trinome(A,B)

        time3 = ((C[0]*altp[2]/3. + C[1]/2.)*altp[2] + C[2])*altp[2]
        time3 = time3 - ((C[0]*altp[0]/3. + C[1]/2.)*altp[0] + C[2])*altp[0]

        # Duration of climb to altp
        #-----------------------------------------------------------------------------------------------------------
        if(cross_over_altp<toc):
            altp_0 = cross_over_altp
            altp_2 = toc
            altp_1 = (altp_0+altp_2)/2.
            altp = np.array([altp_0, altp_1, altp_2])

            speed_mode = "mach"    # mach

            slope,v_z0 = self.air_path(nei,altp[0],disa,speed_mode,mach,mass,rating,kfn)
            slope,v_z1 = self.air_path(nei,altp[1],disa,speed_mode,mach,mass,rating,kfn)
            slope,v_z2 = self.air_path(nei,altp[2],disa,speed_mode,mach,mass,rating,kfn)
            v_z = np.array([v_z0, v_z1, v_z2])

            if(v_z[0]<0. or v_z[1]<0. or v_z[2]<0.):
                print("Climb to top of climb is not possible")

            A = vander3(altp)
            B = 1./v_z
            C = trinome(A,B)

            time4 =  ((C[0]*altp[2]/3. + C[1]/2.)*altp[2] + C[2])*altp[2] \
                   - ((C[0]*altp[0]/3. + C[1]/2.)*altp[0] + C[2])*altp[0]
        else:
            time4 = 0.

        #    Total time
        #-----------------------------------------------------------------------------------------------------------
        ttc = time1 + time2 + time3 + time4

        return {"ttc":ttc}


