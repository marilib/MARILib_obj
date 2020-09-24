#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
         Avionic & Systems, Air Transport Departement, ENAC
"""

from marilib.utils import earth, unit

import matplotlib.pyplot as plt

import numpy as np
from scipy.optimize import fsolve
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import RK23, RK45, solve_ivp

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

        altp_west = np.concatenate((np.arange(2000.,41000.,2000.),np.arange(43000.,85000.,4000.)))
        altp_east = np.concatenate((np.arange(1000.,42000.,2000.),np.arange(45000.,87000.,4000.)))

        self.heading_altp = {"west":[], "east":[]}
        self.heading_altp["west"] = [unit.m_ft(zp) for zp in altp_west]
        self.heading_altp["east"] = [unit.m_ft(zp) for zp in altp_east]

        self.zstep = 2200.
        self.altp_list = np.arange(0., 16000., self.zstep)
        self.altpz = self.altp_list[-1]

        self.f_key_list = ["vz_mcr", "acc_lvf", "dec_lvf", "xdot_mcl", "vz_mcl", "fn_mcl", "ff_mcl",
                           "xdot_fid", "vz_fid", "fn_fid", "ff_fid", "pamb", "tamb",
                           "mach", "cas", "tas", "sar", "fn_crz", "ff_crz"]

        self.data_dict = {}
        self.data_func = {}

        for key in self.f_key_list:
            self.data_dict[key] = {"low":[],"medium":[],"high":[]}

        self.data_dict["altp"] = {"low":[],"medium":[],"high":[]}
        self.data_dict["mass"] = {"low":[],"medium":[],"high":[]}

        self.tow = None
        self.owe = None

        self.heading = None
        self.range = None

        self.cruise_x_stop = None
        self.level_blocker = None

        self.altpw = None
        self.altpx = None
        self.altpy = None

        self.cas1 = None
        self.tas1 = None

        self.cas2 = None
        self.tas2 = None

        self.mach = None

        self.mass_list = None

        self.change_altp = None
        self.change_mass = None
        self.change_cstr = None

        self.flight_profile = None

        self.taxi_out_fuel = None
        self.taxi_out_time = None
        self.taxi_out_dist = None

        self.take_off_fuel = None
        self.take_off_time = None
        self.take_off_dist = None

        self.climb_fuel = None
        self.climb_time = None
        self.climb_dist = None

        self.cruise_fuel = None
        self.cruise_time = None
        self.cruise_dist = None

        self.descent_fuel = None
        self.descent_time = None
        self.descent_dist = None

        self.landing_fuel = None
        self.landing_time = None
        self.landing_dist = None

        self.taxi_in_fuel = None
        self.taxi_in_time = None
        self.taxi_in_dist = None


    def set_flight_domain(self,disa,tow,owe,cas1,altp2,cas2,cruise_mach):
        """Precomputation of all relevant quantities into a grid vs mass and altitude
        """
        if cas1>cas2:
            raise Exception("cas1 must be lower than cas2")

        self.tow = tow
        self.owe = owe

        self.altpx = altp2
        self.cas1 = cas1
        self.cas2 = cas2
        self.mach = cruise_mach

        pamb, tamb, tstd, dtodz = earth.atmosphere(self.altpx,disa)
        sound_speed = earth.sound_speed(tamb)
        self.tas1 = self.get_mach(pamb,"cas",cas1) * sound_speed
        self.tas2 = self.get_mach(pamb,"cas",cas2) * sound_speed

        self.altpy = earth.cross_over_altp(cas2,cruise_mach)
        if self.altpy<self.altpx:
            raise Exception("Cross over altitude must be higher than altp2")

        n_mass = 5
        self.mass_list = np.linspace(owe,tow,n_mass)
        for layer in self.data_dict["altp"].keys():
            self.data_dict["mass"][layer] = self.mass_list

        g = earth.gravity()
        nei = 0
        daltg = 10. # Used to perform finite difference to compute dV / dh

        for altp in self.altp_list:
            pamb, tamb, tstd, dtodz = earth.atmosphere(altp,disa)
            altg  = earth.altg_from_altp(altp,disa)
            pamb_,tamb_,dtodz_ = earth.atmosphere_geo(altg+daltg,disa)

            if altp<=self.altpx+self.zstep:
                layer = "low"
                speed_mode = "cas"
                speed = self.cas1
                mach = self.get_mach(pamb,speed_mode,speed)
                tas  = mach * earth.sound_speed(tamb)
                tas_ = self.get_mach(pamb_,speed_mode,speed) * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                self._fill_table(g,nei,pamb,tamb,mach,fac,layer)
                self.data_dict["altp"][layer].append(altp)

            if self.altpx-self.zstep<=altp and altp<=self.altpy+self.zstep:
                layer = "medium"
                speed_mode = "cas"
                speed = self.cas2
                mach = self.get_mach(pamb,speed_mode,speed)
                tas  = mach * earth.sound_speed(tamb)
                tas_ = self.get_mach(pamb_,speed_mode,speed) * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                self._fill_table(g,nei,pamb,tamb,mach,fac,layer)
                self.data_dict["altp"][layer].append(altp)

            if self.altpy-self.zstep<=altp:
                layer = "high"
                mach = cruise_mach
                tas  = mach * earth.sound_speed(tamb)
                tas_ = mach * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                self._fill_table(g,nei,pamb,tamb,mach,fac,layer)
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

    def _fill_table(self,g,nei,pamb,tamb,mach,fac,layer):
        """Generic function to build performance tables
        """
        cas = earth.vcas_from_mach(pamb,mach)
        tas = mach * earth.sound_speed(tamb)
        dict_fid = self.aircraft.power_system.thrust(pamb,tamb,mach,"FID")
        dict_mcr = self.aircraft.power_system.thrust(pamb,tamb,mach,"MCR")
        dict_mcl = self.aircraft.power_system.thrust(pamb,tamb,mach,"MCL")

        data = {}
        for key in self.f_key_list:
            data[key] = []

        for mass in self.mass_list:
            cz = self.lift_from_speed(pamb,tamb,mach,mass)
            cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

            sin_path_mcr = (dict_mcr["fn"]/(mass*g) - 1./lod) / fac
            vz_mcr = tas * sin_path_mcr

            fn_mcl = dict_mcl["fn"]
            sin_path_mcl = (fn_mcl/(mass*g) - 1./lod) / fac   # Flight path air path sine
            xdot_mcl = tas * np.sqrt(1.-sin_path_mcl**2)                # Acceleration in climb
            vz_mcl = tas * sin_path_mcl
            ff_mcl = -dict_mcl["ff"]

            fn_fid = dict_fid["fn"]
            sin_path_fid = (fn_fid/(mass*g) - 1./lod) / fac   # Flight path air path sine
            xdot_fid = tas * np.sqrt(1.-sin_path_fid**2)                # Acceleration in climb
            vz_fid = tas * sin_path_fid
            ff_fid = -dict_fid["ff"]

            acc_lvf = dict_mcl["fn"]/mass - g/lod
            dec_lvf = dict_fid["fn"]/mass - g/lod

            if layer=="high":
                fn_crz = mass*g / lod
                dict_cruise = self.aircraft.power_system.sc(pamb,tamb,mach,"MCR",fn_crz,nei)
                dict_cruise["lod"] = lod
                sar = self.aircraft.power_system.specific_air_range(mass,tas,dict_cruise)
                ff_crz = -tas / sar
            else:
                sar = np.nan
                fn_crz = np.nan
                ff_crz = np.nan

            for key in self.f_key_list:
                data[key].append(eval(key))

        for key in self.f_key_list:
            self.data_dict[key][layer].append(data[key])

        return

    def cruise_profile(self,mass,vz_mcr,vz_mcl,heading):
        rev_mass_list = [mass]
        for m in reversed(self.mass_list):
            if m<mass: rev_mass_list.append(m)
        altp_list = [zp for zp in self.heading_altp[heading] if self.altpy<=zp<=self.altpz]

        rev_mass_list = list(reversed(self.mass_list))
        k_altp = np.argmax([self.get_val("sar",z,rev_mass_list[0])[0] for z in altp_list])

        best_sar_altp = [altp_list[k_altp]]
        best_sar_mass = [mass]
        sar_tab = [[]]

        max_mcr_mass = []
        mcr_tab = [[]]

        max_mcl_mass = []
        mcl_tab = [[]]

        for k,m in enumerate(rev_mass_list):
            mcr_tab[0].append(self.get_val("vz_mcr",altp_list[k_altp],m)[0]-vz_mcr)
            mcl_tab[0].append(self.get_val("vz_mcl",altp_list[k_altp],m)[0]-vz_mcl)
            sar_tab[0].append(self.get_val("sar",altp_list[k_altp],m)[0])

        # Computing maximum mass for first altitude versus MCR
        f_mcr = interp1d(mcr_tab[0],rev_mass_list,kind="linear", fill_value="extrapolate")
        mcr = f_mcr(0.)
        max_mcr_mass.append(mcr.tolist())

        # Computing maximum mass for first altitude versus MCL
        f_mcl = interp1d(mcl_tab[0],rev_mass_list,kind="linear", fill_value="extrapolate")
        mcl = f_mcl(0.)
        max_mcl_mass.append(mcl.tolist())

        dsar = np.zeros(len(rev_mass_list))
        while k_altp<len(altp_list)-1:
            sar_tab.append([])
            mcr_tab.append([])
            mcl_tab.append([])
            for k,m in enumerate(rev_mass_list):
                mcr_tab[-1].append(self.get_val("vz_mcr",altp_list[k_altp+1],m)[0]-vz_mcr)
                mcl_tab[-1].append(self.get_val("vz_mcl",altp_list[k_altp+1],m)[0]-vz_mcl)
                sar_tab[-1].append(self.get_val("sar",altp_list[k_altp+1],m)[0])
                dsar[k] = sar_tab[-1][k] - sar_tab[-2][k]

            # Computing maximum mass for current altitude versus MCR
            f_mcr = interp1d(mcr_tab[-1],rev_mass_list,kind="linear", fill_value="extrapolate")
            mcr = f_mcr(0.)

            # Computing maximum mass for current altitude versus MCL
            f_mcl = interp1d(mcl_tab[-1],rev_mass_list,kind="linear", fill_value="extrapolate")
            mcl = f_mcl(0.)

            # Computing change altitude mass versus SAR
            f_dsar = interp1d(dsar,rev_mass_list,kind="linear", fill_value="extrapolate")
            msr = f_dsar(0.)

            if msr>min(rev_mass_list):
                best_sar_altp.append(altp_list[k_altp+1])
                best_sar_mass.append(msr.tolist())
                max_mcr_mass.append(mcr.tolist())
                max_mcl_mass.append(mcl.tolist())
            else:
                break

            k_altp += 1

        # Computing effetive changing mass according to SAR and climb capabilities
        constraint = ["sar","mcr","mcl"]
        self.change_altp = [best_sar_altp[0]]
        self.change_mass = [min([best_sar_mass[0], max_mcr_mass[0], max_mcl_mass[0]])]
        kc = np.argmin([best_sar_mass[0], max_mcr_mass[0], max_mcl_mass[0]])
        self.change_cstr = [constraint[kc]]
        for k in range(len(best_sar_altp)-1):
            mass = min([best_sar_mass[1+k], max_mcr_mass[1+k], max_mcl_mass[1+k]])
            kc = np.argmin([best_sar_mass[1+k], max_mcr_mass[1+k], max_mcl_mass[1+k]])
            if abs(mass-self.change_mass[-1])<100.:
                self.change_altp[-1] = best_sar_altp[1+k]
                self.change_mass[-1] = mass
                self.change_cstr[-1] = constraint[kc]
            else:
                self.change_altp.append(best_sar_altp[1+k])
                self.change_mass.append(mass)
                self.change_cstr.append(constraint[kc])

        return

    def get_val(self,f_key,altp,mass,cas=None):
        """This function interpolates performances in the performance tables
        """
        if cas==None:
            if altp<=self.altpx:
            # Climbing at constant cas1 inside lower layer
                return self.data_func[f_key]["low"](mass,altp)
            elif self.altpx<altp and  altp<=self.altpy:
            # Climbing at constant cas2 inside medium layer
                return self.data_func[f_key]["medium"](mass,altp)
            elif self.altpy<altp:
            # Climbing or cruising at constant Mach inside upper layer
                return self.data_func[f_key]["high"](mass,altp)
        else:
            # accelerating at constant altitude altpx
            v1 = self.data_func[f_key]["low"](mass,self.altpx)
            v2 = self.data_func[f_key]["medium"](mass,self.altpx)
            print("coucou")
            print(altp,mass,v1,v2)
            k = (cas-self.cas1) / (self.cas2-self.cas1)
            return v1*(1.-k) + k*v2

    def climb_path(self,z,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        The role of the time (t) is taken by altitude (z) in the integration scheme and time takes part of the state vector
        state = [t,x,mass]
        """
        return np.array([1.,
                         self.get_val("xdot_mcl",z,state[2]),
                         self.get_val("ff_mcl",z,state[2])]) / self.get_val("vz_mcl",z,state[2])

    def descent_path(self,z,state):
        """Perform descent trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        The role of the time (t) is taken by altitude (z) in the integration scheme and time takes part of the state vector
        state = [t,x,mass]
        """
        return np.array([1.,
                         self.get_val("xdot_fid",z,state[2]),
                         self.get_val("ff_fid",z,state[2])]) / self.get_val("vz_fid",z,state[2])

    def acceleration(self,v,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        The role of the time (t) is taken by true air speed (v) in the integration scheme and time takes part of the state vector
        state = [t,x,mass]
        """
        return  np.array([1.,v,self.get_val("ff_mcl",self.altpx, state[2])]) / self.get_val("acc_lvf",self.altpx, state[2])

    def deceleration(self,v,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        The role of the time (t) is taken by true air speed (v) in the integration scheme and time takes part of the state vector
        state = [t,x,mass]
        """
        return  np.array([1.,v,self.get_val("ff_fid",self.altpx, state[2])]) / self.get_val("dec_lvf",self.altpx, state[2])

    def cruise(self,m,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        The role of the time (t) is taken by the mass (m) in the integration scheme and time takes part of the state vector
        state = [t,x,z]
        """
        return np.array([1., self.get_val("tas",state[2],m), 0.]) / self.get_val("ff_crz",state[2],m)

    def cruise_stop(self,m,state):
        """Cruise stop event
        state = [t,x,z]
        """
        return self.cruise_x_stop - state[1]
    cruise_stop.terminal = True


    def fly_mission_end(self,start_mass,start_state,x_stop):
        """Perform level cruise sequence(s) and descent
        """
        self.cruise_x_stop = x_stop
        self.level_blocker = self.climb_dist*6.
        go_ahead = True
        level_index = 0
        sc = []
        # First cruise segment, constant mach & altitude, state = [t,x,z]
        #---------------------------------------------------------------------------------------------------------------
        m0 = start_mass
        m1 = self.change_mass[1]
        state0 = start_state
        n = 5
        t_eval = np.linspace(m0,m1,n)

        sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45", events=self.cruise_stop)

        if (self.range-sol.y[1][-1])<self.level_blocker:   # En of cruise is close, it is useless to climb to upper level
            m1 = self.owe
            t_eval = np.linspace(m0,m1,n)
            sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45", events=self.cruise_stop)
            go_ahead = False

        if np.size(sol.t_events)>0:    # Cruise x stop has been reached, recompute segment up to range
            m1 = sol.t_events[0][-1]
            t_eval = np.linspace(m0,m1,n)
            sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45")
            go_ahead = False

        time = sol.y[0]
        dist = sol.y[1]
        altp = sol.y[2]
        mass = sol.t
        fn = [self.get_val("fn_crz",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_crz",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        mach = np.ones(len(sol.t))*self.mach
        tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
        cas = [earth.vcas_from_mach(p,ma) for p,ma in zip(pamb,mach)]
        if go_ahead:
            s5 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))
        else:
            sc = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        while go_ahead and level_index<(np.size(self.change_altp)-2):
            # Climb to next cruise level, constant mach, state = [t,x,m]
            #---------------------------------------------------------------------------------------------------------------
            z0 = self.change_altp[level_index]
            z1 = self.change_altp[level_index+1]
            state1 = np.array([sol.y[0][-1],
                               sol.y[1][-1],
                               sol.t[-1]])
            n = 5
            t_eval = np.linspace(z0,z1,n)

            sol = solve_ivp(self.climb_path,[z0,z1],state1,t_eval=t_eval, method="RK45")

            time = sol.y[0]
            dist = sol.y[1]
            mass = sol.y[2]
            altp = sol.t
            fn = [self.get_val("fn_mcl",z,m)[0] for z,m in zip(altp,mass)]
            ff = [self.get_val("ff_mcl",z,m)[0] for z,m in zip(altp,mass)]
            pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
            tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
            cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
            mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
            tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
            s6 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

            if sol.y[1][-1]>self.cruise_x_stop:    # Cruise x stop has been reached during climb, extend previous segment
                m1 = sol.y[2][-1]
                t_eval = np.linspace(m0,m1,n)
                sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45", events=self.cruise_stop)
                m1 = sol.t_events[0]
                t_eval = np.linspace(m0,m1,n)
                sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45")
                time = sol.y[0]
                dist = sol.y[1]
                altp = sol.y[2]
                mass = sol.t
                fn = [self.get_val("fn_crz",z,m)[0] for z,m in zip(altp,mass)]
                ff = [self.get_val("ff_crz",z,m)[0] for z,m in zip(altp,mass)]
                pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
                tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
                mach = np.ones(len(sol.t))*self.mach
                tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
                cas = [earth.vcas_from_mach(p,ma) for p,ma in zip(pamb,mach)]
                s5 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

                if len(sc)==0:
                    sc = s5
                else:
                    sc = np.hstack((sc[:,0:-1],s5))

                go_ahead = False

            else:                               # Cruise x stop has not been reached
                # Fly the next cruise segment, constant mach & altitude, state = [t,x,z]
                #---------------------------------------------------------------------------------------------------------------
                m0 = sol.y[2][-1]
                m1 = self.change_mass[level_index+2]
                state0 = np.array([sol.y[0][-1],
                                   sol.y[1][-1],
                                   sol.t[-1]])
                n = 5
                t_eval = np.linspace(m0,m1,n)

                sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45", events=self.cruise_stop)

                if np.size(sol.t_events)==0 and (self.range-sol.y[1][-1])<self.level_blocker:   # En of cruise is close, it is useless to climb to upper level, extend
                    m1 = self.owe
                    t_eval = np.linspace(m0,m1,n)
                    sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45", events=self.cruise_stop)
                    go_ahead = False

                if np.size(sol.t_events)>0:    # Cruise x stop has been reached, recompute segment up to range
                    m1 = sol.t_events[0][0]
                    t_eval = np.linspace(m0,m1,n)
                    sol = solve_ivp(self.cruise, [m0,m1], state0, t_eval=t_eval, method="RK45")
                    go_ahead = False

                time = sol.y[0]
                dist = sol.y[1]
                mass = sol.t
                altp = sol.y[2]
                fn = [self.get_val("fn_crz",z,m)[0] for z,m in zip(altp,mass)]
                ff = [self.get_val("ff_crz",z,m)[0] for z,m in zip(altp,mass)]
                pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
                tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
                cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
                mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
                tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
                s7 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

                if len(sc)==0:
                    sc = np.hstack((s5[:,0:-1],s6[:,0:-1],s7))
                else:
                    sc = np.hstack((sc[:,0:-1],s6[:,0:-1],s7))

                level_index += 1

        self.cruise_time = sol.y[0][-1] - self.climb_time
        self.cruise_fuel = self.tow - sol.t[-1] - self.climb_fuel
        self.cruise_dist = sol.y[1][-1] - self.climb_dist

        # First descent segment, constant mach, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        z0 = sc[2][-1]
        z1 = self.altpy
        state0 = np.array([sc[0][-1],
                           sc[1][-1],
                           sc[3][-1]])
        n = 5
        t_eval = np.linspace(z0,z1,n)

        sol = solve_ivp(self.descent_path,[z0,z1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = sol.t
        fn = [self.get_val("fn_fid",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_fid",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
        mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
        tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
        s1 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        # Second descent segment, constant cas2, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        z0 = self.altpy
        z1 = self.altpx
        state0 = np.array([sol.y[0][-1],
                           sol.y[1][-1],
                           sol.y[2][-1]])
        n = 5
        t_eval = np.linspace(z0,z1,n)

        sol = solve_ivp(self.descent_path,[z0,z1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = sol.t
        fn = [self.get_val("fn_fid",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_fid",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
        mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
        tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
        s2 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        # Desceleration fron cas2 to cas1, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        v0 = self.tas2
        v1 = self.tas1
        state0 = np.array([sol.y[0][-1],
                           sol.y[1][-1],
                           sol.y[2][-1]])
        n = 5
        t_eval = np.linspace(v0,v1,n)

        sol = solve_ivp(self.deceleration,[v0,v1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = np.ones(len(sol.t))*self.altpx
        fn = [self.get_val("fn_fid",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_fid",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        tas = sol.t
        mach = [v/earth.sound_speed(tamb[0]) for v in tas]
        cas = [earth.vcas_from_mach(pamb[0],ma) for ma in mach]
        s3 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        # Last descent segment, constant cas1, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        z0 = self.altpx
        z1 = self.altpw
        state0 = np.array([sol.y[0][-1],
                           sol.y[1][-1],
                           sol.y[2][-1]])
        n = 5
        t_eval = np.linspace(z0,z1,n)

        sol = solve_ivp(self.descent_path,[z0,z1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = sol.t
        fn = [self.get_val("fn_fid",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_fid",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
        mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
        tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
        s4 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        sd = np.hstack((s1[:,0:-1],s2[:,0:-1],s3[:,0:-1],s4))

        self.descent_time = sol.y[0][-1] - self.climb_time - self.cruise_time
        self.descent_fuel = self.tow - sol.y[2][-1] - self.climb_fuel - self.cruise_fuel
        self.descent_dist = sol.y[1][-1] - self.climb_dist - self.cruise_dist

        x_end = sd[1][-1]

        return sc,sd,x_end


    def fly_mission(self,disa,range,tow,owe,altp1,cas1,altp2,cas2,cruise_mach,vz_min_mcr,vz_min_mcl,heading):

        self.range = range
        self.altpw = altp1

        # Precompute airplane performances
        #---------------------------------------------------------------------------------------------------------------
        self.set_flight_domain(disa,tow,owe,cas1,altp2,cas2,cruise_mach)

        # First climb segment, constant cas1, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        z0 = self.altpw
        z1 = self.altpx
        state0 = np.array([0., 0., tow])

        n = 5
        t_eval = np.linspace(z0,z1,n)

        sol = solve_ivp(self.climb_path,[z0,z1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = sol.t
        fn = [self.get_val("fn_mcl",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_mcl",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
        mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
        tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
        s1 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        # Acceleration fron cas1 to cas2, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        v0 = self.tas1
        v1 = self.tas2
        state0 = np.array([sol.y[0][-1],
                           sol.y[1][-1],
                           sol.y[2][-1]])
        n = 5
        t_eval = np.linspace(v0,v1,n)

        sol = solve_ivp(self.acceleration,[v0,v1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = np.ones(len(sol.t))*self.altpx
        fn = [self.get_val("fn_mcl",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_mcl",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        tas = sol.t
        mach = [v/earth.sound_speed(tamb[0]) for v in tas]
        cas = [earth.vcas_from_mach(pamb[0],ma) for ma in mach]
        s2 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        # Second climb segment, constant cas2, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        z0 = self.altpx
        z1 = self.altpy
        state0 = np.array([sol.y[0][-1],
                           sol.y[1][-1],
                           sol.y[2][-1]])
        n = 5
        t_eval = np.linspace(z0,z1,n)

        sol = solve_ivp(self.climb_path,[z0,z1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = sol.t
        fn = [self.get_val("fn_mcl",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_mcl",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
        mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
        tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
        s3 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        # Precomputecruise profile
        #---------------------------------------------------------------------------------------------------------------
        self.cruise_profile(mass[-1],vz_min_mcr,vz_min_mcl,heading)

        # print(self.mass_list)
        # print("")
        # print(self.change_altp)
        # print(self.change_mass)
        # print(self.change_cstr)

        # Third climb segment, constant mach, state = [t,x,mass]
        #---------------------------------------------------------------------------------------------------------------
        z0 = self.altpy
        z1 = self.change_altp[0]
        state0 = np.array([sol.y[0][-1],
                           sol.y[1][-1],
                           sol.y[2][-1]])
        n = 5
        t_eval = np.linspace(z0,z1,n)

        sol = solve_ivp(self.climb_path,[z0,z1],state0,t_eval=t_eval, method="RK45")

        time = sol.y[0]
        dist = sol.y[1]
        mass = sol.y[2]
        altp = sol.t
        fn = [self.get_val("fn_mcl",z,m)[0] for z,m in zip(altp,mass)]
        ff = [self.get_val("ff_mcl",z,m)[0] for z,m in zip(altp,mass)]
        pamb = [self.get_val("pamb",z,m)[0] for z,m in zip(altp,mass)]
        tamb = [self.get_val("tamb",z,m)[0] for z,m in zip(altp,mass)]
        cas = [self.get_val("cas",z,m)[0] for z,m in zip(altp,mass)]
        mach = [earth.mach_from_vcas(p,v) for p,v in zip(pamb,cas)]
        tas = [ma*earth.sound_speed(t) for ma,t in zip(mach,tamb)]
        s4 = np.vstack((time,dist,altp,mass,pamb,tamb,mach,tas,cas,fn,ff))

        s = np.hstack((s1[:,0:-1],s2[:,0:-1],s3[:,0:-1],s4))

        self.climb_time = sol.y[0][-1]
        self.climb_fuel = self.tow - sol.y[2][-1]
        self.climb_dist = sol.y[1][-1]

        # First test on range
        #---------------------------------------------------------------------------------------------------------------
        if self.climb_dist>self.range:
            raise Exception("Range is shorter than climb distance")

        start_mass = sol.y[2][-1]

        start_state = np.array([sol.y[0][-1],
                                sol.y[1][-1],
                                self.change_altp[0]])

        x_stop1 = self.range

        # Fly the rest of the mission looking for target range
        #---------------------------------------------------------------------------------------------------------------
        sc,sd,x_end1 = self.fly_mission_end(start_mass,start_state,x_stop1)

        if x_end1<self.range:
            raise Exception("OWE does not allow to complete the mission, lower OWE to embark more fuel")

        x_stop2 = x_stop1 + (self.range - x_end1)

        sc,sd,x_end2 = self.fly_mission_end(start_mass,start_state,x_stop2)

        x_stop = x_stop1 +(self.range-x_end1)*(x_stop2-x_stop1)/(x_end2-x_end1)

        sc,sd,x_end = self.fly_mission_end(start_mass,start_state,x_stop)

        flight_profile = np.hstack((s[:,0:-1],sc[:,0:-1],sd)).transpose()

        self.flight_profile = {"label":["time","dist","altp","mass","pamb","tamb","mach","tas","cas","fn","ff"],
                               "unit":["h","NM","ft","kg","Pa","K","mach","kt","kt","kN","kg/h"],
                               "data":flight_profile}

        # Departure ground legs
        #---------------------------------------------------------------------------------------------------------------
        dict = self.departure_ground_legs(self.tow)

        self.taxi_out_fuel = dict["fuel"]["taxi_out"]
        self.taxi_out_time = dict["time"]["taxi_out"]
        self.taxi_out_dist = 0.

        self.take_off_fuel = dict["fuel"]["take_off"]
        self.take_off_time = dict["time"]["take_off"]
        self.take_off_dist = 0.

        # Arrival ground legs
        #-----------------------------------------------------------------------------------------------------------
        dict = self.arrival_ground_legs(sd[3][-1])

        self.landing_fuel = dict["fuel"]["landing"]
        self.landing_time = dict["time"]["landing"]
        self.landing_dist = 0.

        self.taxi_in_fuel = dict["fuel"]["taxi_in"]
        self.taxi_in_time = dict["time"]["taxi_in"]
        self.taxi_in_dist = 0.

        return

    def fly_this_profile(self,disa,tow,profile):
        """Compute fuel consumption along a given flight path
        state = [x,z,m]
        """
        g = earth.gravity()

        dt1 = profile[1:-1,0] - profile[0:-2,0]     # Time intervals
        dx = profile[1:-1,1] - profile[0:-2,1]      # Track intervals
        dz = profile[1:-1,2] - profile[0:-2,2]      # Altitude intervals

        t1 = 0.5*(profile[1:-1,0] + profile[0:-2,0])    # Mean time stamps
        vx = dx / dt1                                   # Mean speed
        vz = dz / dt1                                   # Mean speed

        dvx = vx[1:-1] - vx[0:-2]
        dvz = vz[1:-1] - vz[0:-2]

        dt2 = t1[1:-1] - t1[0:-2]
        t2 = 0.5*(t1[1:-1] + t1[0:-2])

        x_dot = interp1d(t1,vx, kind="linear", fill_value="extrapolate")
        z_dot = interp1d(t1,vz, kind="linear", fill_value="extrapolate")
        vx_dot = interp1d(t2,dvx/dt2, kind="linear", fill_value="extrapolate")
        vz_dot = interp1d(t2,dvz/dt2, kind="linear", fill_value="extrapolate")

        def state_dot(t,state):
            altp = state[1]
            mass = state[2]

            vx = x_dot(t)
            vz = z_dot(t)
            tas = np.sqrt(vx**2 + vz**2)

            sin_path = vz / tas

            vxd = vx_dot(t)
            vzd = vz_dot(t)
            acc = np.sqrt(vxd**2 + vzd**2)

            pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
            mach = tas*earth.sound_speed(tamb)

            cz = self.lift_from_speed(pamb,tamb,mach,mass)
            cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

            nei = 0.
            fn = mass*g*(acc/g + sin_path + 1./lod)
            dict = self.aircraft.power_system.sc(pamb,tamb,mach,"MCL",fn,nei)
            m_dot = -dict["ff"]

            return np.array([vx, vz, m_dot])












    def draw_flight_profile(self):

        plot_title = self.aircraft.name
        window_title = "Mission profile"

        abs = [unit.NM_m(x) for x in self.flight_profile["data"][:,1]]
        ord = [unit.ft_m(z) for z in self.flight_profile["data"][:,2]]

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(window_title)
        fig.suptitle(plot_title, fontsize=14)

        plt.plot(abs,ord,linewidth=2,color="blue")

        plt.grid(True)

        plt.ylabel('Pressure altitude (ft)')
        plt.xlabel('Distance (NM)')

        plt.show()



