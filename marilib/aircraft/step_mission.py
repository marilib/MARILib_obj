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

from marilib.aircraft.performance import Flight

from marilib.aircraft.model_config import get_init


class StepMission(Flight):
    """Mission simulated step by step
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.westbound_altp = np.concatenate((np.arange(2000.,41000.,2000.),np.arange(43000.,85000.,4000.)))
        self.eastbound_altp = np.concatenate((np.arange(1000.,42000.,2000.),np.arange(45000.,87000.,4000.)))

    def set_flight_domain(self,disa,tow,zfw,cas1,cas2,cruise_mach):
        """Precomputation of all relevant quantities into a grid vs mass and altitude
        """
        if cas1>cas2:
            raise Exception("cas1 must be lower than cas2")

        altpx = unit.m_ft(10000.)
        altpy = earth.cross_over_altp(cas2,cruise_mach)
        if altpy<unit.m_ft(10000.):
            raise Exception("Cross over altitude must be higher than 10000 ft")

        altp0 = unit.m_ft(1500.)
        altp1 = unit.m_ft(50000.)
        n_altp = 8
        altp_list = np.linspace(altp0, altp1, n_altp)

        n_mass = 4
        mass_list = np.linspace(zfw,tow,n_mass)

        g = earth.gravity()
        nei = 0
        daltg = 10.

        data_dict1 = {}
        data_dict2 = {}
        data_dict3 = {}

        for altp in altp_list:
            pamb, tamb, tstd, dtodz = earth.atmosphere(altp,disa)
            altg  = earth.altg_from_altp(altp,disa)
            pamb_,tamb_,tstd_,dtodz_ = earth.atmosphere_geo(altg+daltg,disa)

            if altp<=altpx:

                speed_mode = "cas"
                speed = cas1
                mach = Flight.get_mach(pamb,speed_mode,speed)
                tas  = mach * earth.sound_speed(tamb)
                tas_ = Flight.get_mach(pamb_,speed_mode,speed) * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                data_dict1[altp] = self._fill_table(g,nei,pamb,tamb,mach,fac,mass_list)

            if altpx<=altp and altp<=altpy:

                speed_mode = "cas"
                speed = cas2
                mach = Flight.get_mach(pamb,speed_mode,speed)
                tas  = mach * earth.sound_speed(tamb)
                tas_ = Flight.get_mach(pamb_,speed_mode,speed) * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                data_dict2[altp] = self._fill_table(g,nei,pamb,tamb,mach,fac,mass_list)

            if altpy<=altp:

                speed_mode = "mach"
                mach = cruise_mach
                tas  = mach * earth.sound_speed(tamb)
                tas_ = mach * earth.sound_speed(tamb_)
                dtas_o_dh = (tas_-tas) / daltg
                fac = (1. + (tas/g)*dtas_o_dh)
                data_dict3[altp] = self._fill_table(g,nei,pamb,tamb,mach,fac,mass_list)



    def _fill_table(self,g,nei,pamb,tamb,mach,fac,mass_list):

        tas = mach * earth.sound_speed(tamb)

        dict_mcr = self.aircraft.power_system.thrust(pamb,tamb,mach,"MCR")
        dict_mcl = self.aircraft.power_system.thrust(pamb,tamb,mach,"MCL")

        for mass in mass_list:

            cz = self.lift_from_speed(pamb,tamb,mach,mass)
            cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

            sin_path_mcr = (dict_mcr["fn"]/(mass*g) - 1./lod) / fac
            vz_mcr = tas * sin_path_mcr

            sin_path_mcl = ( dict_mcl["fn"]/(mass*g) - 1./lod ) / fac   # Flight path air path sine
            xdot_mcl = tas * np.sqrt(1.-sin_path_mcl**2)    # Acceleration in climb
            vz_mcl = tas * sin_path_mcl
            mdot_mcl = dict_mcl["ff"]

            xacc_lvf = dict_mcl["fn"]/mass - g/lod

            fn_cruise = mass*g / lod
            dict_cruise = self.aircraft.power_system.sc(pamb,tamb,mach,"MCR",fn_cruise,nei)
            dict_cruise["lod"] = lod
            sar = self.aircraft.power_system.specific_air_range(mass,tas,dict_cruise)
            ff = tas / sar

            dict = {}

            dict["vz_mcr"].append(vz_mcr)
            dict["xacc_lvf"].append(xacc_lvf)
            dict["sar"].append(sar)

            dict["xdot_mcl"].append(xdot_mcl)
            dict["vz_mcl"].append(vz_mcl)
            dict["mdot_mcl"].append(mdot_mcl)

            dict["mass"].append(mass)
            dict["tas"].append(tas)
            dict["ff"].append(ff)

        return dict









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
