#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
         Avionic & Systems, Air Transport Departement, ENAC
"""

from marilib.utils import earth, unit

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp2d

from marilib.aircraft.performance import Flight

from marilib.aircraft.model_config import get_init


class StepMission(Flight):
    """Mission simulated step by step
    This module precomputes a table of performances before resolving a time intergration of the trajectory using interpolated performances
    The table is function of pressure altitude and mass in each altitude layer where the speed is fixed :
        low : between 1500ft and 10000ft
        medium : between 10000ft and cross over altitude
        high : from cross over altitude and over
    Precomputed performances are listed in f_key_list :
        vz_mcr : vertical speed in MCR rating
        xacc_lvf : level flight acceleration in MCL rating
        xdot_mcl : horizontal speed when climbing in MCL rating
        vz_mcl : vertical speed in MCL rating
        ff_mcl : fuel flow in MCL rating
        xdot_fid : horizontal speed when descending in FID rating
        vz_fid : vertical speed in FID rating
        ff_fid : fuel flow in FID rating
        tas : current true air speed (whatever the layer)
        sar : current specific air range
        ff : fuel flow in level flight
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.heading_altp = {}
        self.heading_altp["west"] = np.concatenate((np.arange(2000.,41000.,2000.),np.arange(43000.,85000.,4000.)))
        self.heading_altp["east"] = np.concatenate((np.arange(1000.,42000.,2000.),np.arange(45000.,87000.,4000.)))

        self.altpx = unit.m_ft(10000.)
        self.altpz = unit.m_ft(51000.)
        self.altp_list = [ unit.m_ft(1500.), self.altpx,
                                             unit.m_ft(20000.),
                                             unit.m_ft(30000.),
                                             11000.,
                                             unit.m_ft(41000.),
                                             self.altpz]
        self.altpy = None
        self.cas1 = None
        self.cas2 = None
        self.tas2 = None
        self.mach = None

        self.heading = None

        self.f_key_list = ["vz_mcr", "acc_lvf", "dec_lvf", "xdot_mcl", "vz_mcl", "ff_mcl", "xdot_fid", "vz_fid", "ff_fid", "mach", "tas", "sar", "ff"]

        self.data_dict = {}
        self.data_func = {}

        for key in self.f_key_list:
            self.data_dict[key] = {"low":[],"medium":[],"high":[]}

        self.data_dict["altp"] = {"low":[],"medium":[],"high":[]}
        self.data_dict["mass"] = {"low":[],"medium":[],"high":[]}


    def set_flight_domain(self,disa,tow,owe,cas1,cas2,cruise_mach):
        """Precomputation of all relevant quantities into a grid vs mass and altitude
        """
        if cas1>cas2:
            raise Exception("cas1 must be lower than cas2")

        self.cas1 = cas1
        self.cas2 = cas2
        self.mach = cruise_mach

        self.altpx = unit.m_ft(10000.)
        pamb, tamb, tstd, dtodz = earth.atmosphere(self.altpx,disa)
        self.tas2 = self.get_mach(pamb,"cas",cas2) * earth.sound_speed(tamb)

        self.altpy = earth.cross_over_altp(cas2,cruise_mach)
        if self.altpy<unit.m_ft(10000.):
            raise Exception("Cross over altitude must be higher than 10000 ft")

        n_mass = 4
        mass_list = np.linspace(owe,tow,n_mass)
        for layer in self.data_dict["altp"].keys():
            self.data_dict["mass"][layer] = mass_list

        g = earth.gravity()
        nei = 0
        daltg = 10. # Used to perform finite difference to compute dV / dh

        for altp in self.altp_list:
            pamb, tamb, tstd, dtodz = earth.atmosphere(altp,disa)
            altg  = earth.altg_from_altp(altp,disa)
            pamb_,tamb_,dtodz_ = earth.atmosphere_geo(altg+daltg,disa)

            if altp<=self.altpx:
                layer = "low"
                speed_mode = "cas"
                speed = self.cas1
                mach = self.get_mach(pamb,speed_mode,speed)
                tas  = mach * earth.sound_speed(tamb)
                tas_ = self.get_mach(pamb_,speed_mode,speed) * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                self._fill_table(g,nei,pamb,tamb,mach,fac,mass_list,layer)
                self.data_dict["altp"][layer].append(altp)

            if self.altpx<=altp and altp<=self.altpy:
                layer = "medium"
                speed_mode = "cas"
                speed = self.cas2
                mach = self.get_mach(pamb,speed_mode,speed)
                tas  = mach * earth.sound_speed(tamb)
                tas_ = self.get_mach(pamb_,speed_mode,speed) * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                self._fill_table(g,nei,pamb,tamb,mach,fac,mass_list,layer)
                self.data_dict["altp"][layer].append(altp)

            if self.altpy<=altp:
                layer = "high"
                mach = cruise_mach
                tas  = mach * earth.sound_speed(tamb)
                tas_ = mach * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                self._fill_table(g,nei,pamb,tamb,mach,fac,mass_list,layer)
                self.data_dict["altp"][layer].append(altp)

        # Build interpolation functions
        self.data_func = {}
        for f_key in self.f_key_list:
            self.data_func[f_key] = {}
            for layer in self.data_dict["altp"].keys():
                self.data_func[f_key][layer] = interp2d(self.data_dict["mass"][layer],
                                                        self.data_dict["altp"][layer],
                                                        self.data_dict[f_key][layer],
                                                        kind="linear")

    def _fill_table(self,g,nei,pamb,tamb,mach,fac,mass_range,layer):
        """Generic function to build performance tables
        """
        tas = mach * earth.sound_speed(tamb)

        dict_fid = self.aircraft.power_system.thrust(pamb,tamb,mach,"FID")
        dict_mcr = self.aircraft.power_system.thrust(pamb,tamb,mach,"MCR")
        dict_mcl = self.aircraft.power_system.thrust(pamb,tamb,mach,"MCL")

        data = {}
        for key in self.f_key_list:
            data[key] = []

        for mass in mass_range:
            cz = self.lift_from_speed(pamb,tamb,mach,mass)
            cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

            sin_path_mcr = (dict_mcr["fn"]/(mass*g) - 1./lod) / fac
            vz_mcr = tas * sin_path_mcr

            sin_path_mcl = ( dict_mcl["fn"]/(mass*g) - 1./lod ) / fac   # Flight path air path sine
            xdot_mcl = tas * np.sqrt(1.-sin_path_mcl**2)                # Acceleration in climb
            vz_mcl = tas * sin_path_mcl
            ff_mcl = -dict_mcl["ff"]

            sin_path_fid = ( dict_fid["fn"]/(mass*g) - 1./lod ) / fac   # Flight path air path sine
            xdot_fid = tas * np.sqrt(1.-sin_path_fid**2)                # Acceleration in climb
            vz_fid = tas * sin_path_fid
            ff_fid = -dict_fid["ff"]

            acc_lvf = dict_mcl["fn"]/mass - g/lod
            dec_lvf = dict_fid["fn"]/mass - g/lod

            if layer=="high":
                fn_cruise = mass*g / lod
                dict_cruise = self.aircraft.power_system.sc(pamb,tamb,mach,"MCR",fn_cruise,nei)
                dict_cruise["lod"] = lod
                sar = self.aircraft.power_system.specific_air_range(mass,tas,dict_cruise)
                ff = -tas / sar
            else:
                sar = np.nan
                ff = np.nan

            for key in self.f_key_list:
                data[key].append(eval(key))

        for key in self.f_key_list:
            self.data_dict[key][layer].append(data[key])

        return

    def get_val(self,f_key,altp,mass,cas=np.nan):
        """This function interpolates performances in the performance tables
        """
        if altp<=self.altpx-1.:
        # Climbing at constant cas1 inside lower layer
            return self.data_func[f_key]["low"](mass,altp)
        elif abs(self.altpx-altp)<=1.e-4:
        # accelerating at constant altitude altpx
            v1 = self.data_func[f_key]["low"](mass,self.altpx)
            v2 = self.data_func[f_key]["medium"](mass,self.altpx)
            k = (cas-self.cas1) / (self.cas2-self.cas1)
            v = v1*(1.-k) + k*v2
        elif self.altpx+1.<=altp and  altp<=self.altpy:
        # Climbing at constant cas2 inside medium layer
            return self.data_func[f_key]["medium"](mass,altp)
        elif self.altpy<=altp:
        # Climbing or cruising at constant Mach inside upper layer
            return self.data_func[f_key]["high"](mass,altp)

    def transition_path(self,t,state, disa,vz_min_mcr,vz_min_mcl,rating):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,z,mass]
        """
        if rating=="MCL":
            return [self.get_val("xdot_mcl",state[1], state[2]),
                    self.get_val("vz_mcl",state[1], state[2]),
                    self.get_val("ff_mcl",state[1], state[2])]
        elif rating=="FID":
            return [self.get_val("xdot_fid",state[1], state[2]),
                    self.get_val("vz_fid",state[1], state[2]),
                    self.get_val("ff_fid",state[1], state[2])]
        else:
            raise Exception("Rating not allowed, must be MCL or FID")

    def acceleration(self,t,state, disa,rating):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,xdot,mass]
        """
        if rating=="MCL":
            return [state[1],
                    self.get_val("acc_lvf",self.altpx, state[2]),
                    self.get_val("ff_mcl",self.altpx, state[2])]
        elif rating=="FID":
            return [state[1],
                    self.get_val("dec_lvf",self.altpx, state[2]),
                    self.get_val("ff_fid",self.altpx, state[2])]
        else:
            raise Exception("Rating not allowed, must be MCL or FID")

    def cruise(self,t,state, disa):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,z,mass]
        """
        return [self.get_val("tas",state[1], state[2]),
                0.,
                self.get_val("ff",state[1], state[2])]

    def altpx_stop(self,t,state, disa,vz_min_mcr,vz_min_mcl,rating):
        # state = [x,z,mass]
        return self.altpx-state[1]

    def cas2_stop(self,t,state, disa,vz_min_mcr,vz_min_mcl,rating):
        # state = [x,xdot,mass]
        return self.tas2-state[1]

    def altpy_stop(self,t,state, disa,vz_min_mcr,vz_min_mcl,rating):
        # state = [x,z,mass]
        return self.altpy-state[1]

    def mach_stop(self,t,state, disa,vz_min_mcr,vz_min_mcl,rating):
        # state = [x,z,mass]
        return self.mach-self.get_val("mach",state[1], state[2]),

    def best_sar_altp(self,altp,mass,heading="east"):
        altp_list = [zp for zp in self.heading[heading] if altp<=zp<=self.altpz]

        for zp in altp_list:
            if self.vz_mcr<=self.get_val("vz_mcr",zp,mass):
                zp_max_mcr = zp
            else: break

        for zp in altp_list:
            if self.vz_mcl<=self.get_val("vz_mcl",zp,mass):
                zp_max_mcl = zp
            else: break

        sar_list = [self.get_val("sar",zp,mass) for zp in altp_list]

        return min(zp_max_mcr, zp_max_mcl, altp_list[np.argmax(sar_list)])












class StepMissionZero(Flight):
    """Mission simulated step by step
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.westbound_altp = np.concatenate((np.arange(2000.,41000.,2000.),np.arange(43000.,85000.,4000.)))
        self.eastbound_altp = np.concatenate((np.arange(1000.,42000.,2000.),np.arange(45000.,87000.,4000.)))

        self.event_altp.terminal = True
        self.event_mach.terminal = True
        self.event_vcas.terminal = True
        self.event_vz_mcr.terminal = True
        self.event_vz_mcl.terminal = True

        self.event_speed.terminal = True

    def speed_from_mach(self,pamb,mach,speed_mode):
        if speed_mode=="mach":
            return mach
        elif speed_mode=="cas":
            return earth.vcas_from_mach(pamb,mach)

    #-------------------------------------------------------------------------------------------------------------------
    def set_flight_profile(self,t,state, nei,disa):
        """Compute nodes of the flight profile
        state = [x,z,mass,tas]
        """




    #-------------------------------------------------------------------------------------------------------------------
    def climb_path(self,t,state, nei,disa, speed_mode,speed_max, altp_stop,vz_mcr,vz_mcl):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,z,mass,tas]
        """
        kfn = 1.
        altg = state[1]
        mass = state[2]
        vtas = state[3]

        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)

        path,vz,fn,ff,acc,cz,cx,pamb,tamb = Flight.air_path(nei,altp,disa,speed_mode,speed,mass,"MCL",kfn, full_output=True)

        state_d = np.zeros(3)
        state_d[0] = vtas*np.cos(path)
        state_d[1] = vz
        state_d[2] = -ff
        state_d[3] = acc

        return state_d

    def event_climb_altp(self,t,state, nei,disa, speed_mode,speed_max, altp_stop,vz_mcr,vz_mcl):
        """Detect altitude crossing
        state = [x,zp,mass,tas]
        """
        altg = state[1]
        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)

        return altp_stop-altp

    def event_climb_speed(self,t,state, nei,disa, speed_mode,speed_max, altp_stop,vz_mcr,vz_mcl):
        """Detect max Mach crossing
        state = [x,zp,mass]
        """
        altg = state[1]
        vtas = state[3]
        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)
        return speed_max-speed

    def event_vz_mcr(self,t,state, nei,disa, speed_mode,speed_max, altp_stop,vz_mcr,vz_mcl):
        """Detect max cruise ceiling
        state = [x,zp,mass]
        """
        kfn = 1.
        altg = state[1]
        mass = state[2]
        vtas = state[3]
        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)
        path,vz,fn,ff,acc,cz,cx,pamb,tamb = Flight.air_path(nei,altp,disa,speed_mode,speed,mass,"MCR",kfn, full_output=True)
        return vz-vz_mcr

    def event_vz_mcl(self,t,state, nei,disa, speed_mode,speed_max, altp_stop,vz_mcr,vz_mcl):
        """Detect max climb ceiling
        state = [x,zp,mass]
        """
        kfn = 1.
        altg = state[1]
        mass = state[2]
        vtas = state[3]
        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)
        path,vz,fn,ff,acc,cz,cx,pamb,tamb = Flight.air_path(nei,altp,disa,speed_mode,speed,mass,"MCL",kfn, full_output=True)
        return vz-vz_mcl

    #-------------------------------------------------------------------------------------------------------------------
    def change_speed(self,t,state, nei,disa, speed_mode,speed_limit):
        """Perform acceleration or deceleration segment driven by Calibrated Air Speed (speed_mode="cas" or Mach (speed_mode="mach"
        state = [x,z,mass,tas]
        """
        throttle = 1.
        altg = state[1]
        mass = state[2]
        vtas = state[3]

        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)

        if speed>speed_limit:
            rating = "FID"
        else:
            rating = "MCL"

        acc,fn,ff,cz,cx,pamb,tamb = Flight.acceleration(self,nei,altp,disa,"mach",mach,mass,rating,throttle, full_output=True)

        state_d = np.zeros(3)
        state_d[0] = vtas
        state_d[1] = 0.
        state_d[2] = -ff
        state_d[3] = acc

        return state_d

    def event_speed(self,t,state, nei,disa, speed_mode,speed_limit):
        """Detect max Mach crossing
        state = [x,z,mass,tas]
        """
        altg = state[1]
        vtas = state[3]
        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)
        return speed_limit-speed

    #-------------------------------------------------------------------------------------------------------------------
    def cruise_path(self,t,state, nei,disa, speed_mode,speed, x_limit,time_limit,mass_limit):
        """Perform acceleration or deceleration segment driven by Calibrated Air Speed (speed_mode="cas" or Mach (speed_mode="mach"
        state = [x,z,mass,tas]
        """
        throttle = 1.
        altg = state[1]
        mass = state[2]
        vtas = state[3]

        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)

        dict = Flight.level_flight(pamb,tamb,mach,mass)

        state_d = np.zeros(3)
        state_d[0] = vtas
        state_d[1] = 0.
        state_d[2] = -dict["ff"]
        state_d[3] = 0.

        return state_d

    def event_x(self,t,state, nei,disa, speed_mode,speed, x_limit,time_limit,mass_limit):
        """Detect x limit crossing
        state = [x,z,mass,tas]
        """
        x = state[0]
        return x_limit-x

    def event_t(self,t,state, nei,disa, speed_mode,speed, x_limit,time_limit,mass_limit):
        """Detect time limit crossing
        state = [x,z,mass,tas]
        """
        return time_limit-t

    def event_m(self,t,state, nei,disa, speed_mode,speed, x_limit,time_limit,mass_limit):
        """Detect mass limit crossing
        state = [x,z,mass,tas]
        """
        mass = state[2]
        return mass-mass_limit

    #-------------------------------------------------------------------------------------------------------------------
    def descent_path(self,t,state, nei,disa, speed_mode,speed_target, vz_target,altp_target):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,zp,mass]
        """
        altg = state[1]
        mass = state[2]
        vtas = state[3]

        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(self,pamb,mach,speed_mode)

        path,thtl,fn,ff,acc,cz,cx,pamb,tamb = Flight.descent(self,nei,altp,disa,speed_mode,speed,vz_target,mass)

        mach = Flight.get_mach(pamb,speed_mode,speed)
        vtas = earth.vtas_from_mach(altp,disa,mach)

        state_d = np.zeros(3)
        state_d[0] = vtas*np.cos(path)
        state_d[1] = vz_target
        state_d[2] = -ff
        state_d[3] = acc

        return state_d

    def event_descent_altp(self,t,state, nei,disa, speed_mode,speed_target, vz_target,altp_target):
        """Detect altitude crossing
        state = [x,zp,mass]
        """
        altg = state[1]
        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        altp = earth.pressure_altitude(pamb)

        return altp_target-altp

    def event_descent_speed(self,t,state, nei,disa, speed_mode,speed_target, altp_stop,vz_mcr,vz_mcl):
        """Detect max Mach crossing
        state = [x,zp,mass]
        """
        altg = state[1]
        vtas = state[3]
        pamb,tamb,dtodz = earth.atmosphere_geo(altg,disa)
        mach = vtas / earth.sound_speed(tamb)
        speed = self.speed_from_mach(pamb,mach,speed_mode)
        return speed_target-speed






