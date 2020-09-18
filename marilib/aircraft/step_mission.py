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
from scipy.interpolate import interp1d, interp2d
from scipy.integrate import solve_ivp

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

        self.altpx = unit.m_ft(10000.)
        self.altpz = unit.m_ft(51000.)
        self.altp_list = [ unit.m_ft(1500.), self.altpx,
                                             unit.m_ft(20000.),
                                             unit.m_ft(30000.),
                                             11000.,
                                             unit.m_ft(41000.),
                                             self.altpz]

        self.f_key_list = ["vz_mcr", "acc_lvf", "dec_lvf", "xdot_mcl", "vz_mcl", "ff_mcl", "xdot_fid", "vz_fid", "ff_fid", "mach", "tas", "sar", "ff"]

        self.data_dict = {}
        self.data_func = {}

        for key in self.f_key_list:
            self.data_dict[key] = {"low":[],"medium":[],"high":[]}

        self.data_dict["altp"] = {"low":[],"medium":[],"high":[]}
        self.data_dict["mass"] = {"low":[],"medium":[],"high":[]}

        self.heading = None

        self.altpy = None
        self.cas1 = None
        self.cas2 = None
        self.tas2 = None
        self.mach = None

        self.mass_list = None

        self.change_altp = None
        self.change_mass = None
        self.change_cstr = None

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

            if altp<=self.altpx:
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

            if self.altpx<=altp and altp<=self.altpy:
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

            if self.altpy<=altp:
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
        kf = 0.56
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

            sin_path_mcr = (kf*dict_mcr["fn"]/(mass*g) - 1./lod) / fac
            vz_mcr = tas * sin_path_mcr

            sin_path_mcl = (kf*dict_mcl["fn"]/(mass*g) - 1./lod) / fac   # Flight path air path sine
            xdot_mcl = tas * np.sqrt(1.-sin_path_mcl**2)                # Acceleration in climb
            vz_mcl = tas * sin_path_mcl
            ff_mcl = -dict_mcl["ff"]

            sin_path_fid = (dict_fid["fn"]/(mass*g) - 1./lod) / fac   # Flight path air path sine
            xdot_fid = tas * np.sqrt(1.-sin_path_fid**2)                # Acceleration in climb
            vz_fid = tas * sin_path_fid
            ff_fid = -dict_fid["ff"]

            acc_lvf = kf*dict_mcl["fn"]/mass - g/lod
            dec_lvf = kf*dict_fid["fn"]/mass - g/lod

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

    def cruise_profile(self,mass,vz_mcr,vz_mcl,heading="east"):
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

    def climb_path(self,t,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,z,mass]
        """
        altp = state[1]
        mass = state[2]
        n = np.size(altp)
        if n==1:
            return [self.get_val("xdot_mcl",altp,mass),
                    self.get_val("vz_mcl",altp,mass),
                    self.get_val("ff_mcl",altp,mass)]
        else:
            y = np.zeros((n,3))
            for i in range(n):
                y[i,:] = [self.get_val("xdot_mcl",altp[i],mass[i]),
                          self.get_val("vz_mcl",altp[i],mass[i]),
                          self.get_val("ff_mcl",altp[i],mass[i])]
            return y

    def descent_path(self,t,state):
        """Perform descent trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,z,mass]
        """
        return [self.get_val("xdot_fid",state[1], state[2]),
                self.get_val("vz_fid",state[1], state[2]),
                self.get_val("ff_fid",state[1], state[2])]

    def acceleration(self,t,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,xdot,mass]
        """
        return [state[1],
                self.get_val("acc_lvf",self.altpx, state[2]),
                self.get_val("ff_mcl",self.altpx, state[2])]

    def deceleration(self,t,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,xdot,mass]
        """
        return [state[1],
                self.get_val("dec_lvf",self.altpx, state[2]),
                self.get_val("ff_fid",self.altpx, state[2])]

    def cruise(self,t,state):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,z,mass]
        """
        return [self.get_val("tas",state[1], state[2]),
                0.,
                self.get_val("ff",state[1], state[2])]

    def altpx_stop(self,t,state):
        # state = [x,z,mass]
        return self.altpx-state[1]
    altpx_stop.terminal = True

    def cas2_stop(self,t,state):
        # state = [x,xdot,mass]
        return self.tas2-state[1]

    def altpy_stop(self,t,state):
        # state = [x,z,mass]
        return self.altpy-state[1]

    def mach_stop(self,t,state):
        # state = [x,z,mass]
        return self.mach-self.get_val("mach",state[1], state[2]),

    def fly_mission(self,disa,tow,zfw,cas1,cas2,cruise_mach,vz_min_mcr,vz_min_mcl):

        self.set_flight_domain(disa,tow,zfw,cas1,cas2,cruise_mach)

        self.cruise_profile(tow,vz_min_mcr,vz_min_mcl)

        # state = [x,z,mass]
        state0 = [0.,
                  unit.m_ft(1500.),
                  tow]

        time_span = [0., 120.]

        sol = solve_ivp(self.climb_path, time_span, state0, vectorized=False,  method="RK45")#, events=self.altpx_stop)

        # print(sol.t)
        # print(sol.y)



