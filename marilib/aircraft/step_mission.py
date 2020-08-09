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

        self.event_altp.terminal = True
        self.event_mach.terminal = True
        self.event_vz_mcr.terminal = True
        self.event_vz_mcl.terminal = True

        self.event_speed.terminal = True

    def climb_path(self,t,state, nei,disa,speed_mode,speed, mach_max,altp_max,vz_mcr,vz_mcl):
        """Perform climb trajectory segment at constant Calibrated Air Speed (speed_mode="cas" or constant Mach (speed_mode="mach"
        state = [x,zp,mass]
        """
        kfn = 1.
        altp = state[1]
        mass = state[2]

        path,vz,fn,ff,cz,cx,pamb,tamb = Flight.air_path(nei,altp,disa,speed_mode,speed,mass,"MCL",kfn, full_output=True)

        mach = Flight.get_mach(pamb,speed_mode,speed)
        vtas = earth.vtas_from_mach(altp,disa,mach)

        state_d = np.zeros(3)
        state_d[0] = vtas*np.cos(path)
        state_d[1] = vz
        state_d[2] = -ff

        return state_d

    def event_altp(self,t,state, nei,disa,speed_mode,speed, mach_max,altp_max,vz_mcr,vz_mcl):
        """Detect altitude crossing
        state = [x,zp,mass]
        """
        altp = state[1]
        return altp_max-altp

    def event_mach(self,t,state, nei,disa,speed_mode,speed, mach_max,altp_max,vz_mcr,vz_mcl):
        """Detect max Mach crossing
        state = [x,zp,mass]
        """
        altp = state[1]
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        mach = Flight.get_mach(pamb,speed_mode,speed)
        return mach_max-mach

    def event_vz_mcr(self,t,state, nei,disa,speed_mode,speed, mach_max,altp_max,vz_mcr,vz_mcl):
        """Detect max cruise ceiling
        state = [x,zp,mass]
        """
        kfn = 1.
        altp = state[1]
        mass = state[2]
        path,vz,fn,ff,cz,cx,pamb,tamb = Flight.air_path(nei,altp,disa,speed_mode,speed,mass,"MCR",kfn, full_output=True)
        return vz-vz_mcr

    def event_vz_mcl(self,t,state, nei,disa,speed_mode,speed, mach_max,altp_max,vz_mcr,vz_mcl):
        """Detect max climb ceiling
        state = [x,zp,mass]
        """
        kfn = 1.
        altp = state[1]
        mass = state[2]
        path,vz,fn,ff,cz,cx,pamb,tamb = Flight.air_path(nei,altp,disa,speed_mode,speed,mass,"MCL",kfn, full_output=True)
        return vz-vz_mcl

    def change_speed(self,t,state, rating,nei,altp,disa,speed_mode,speed, speed_limit):
        """Perform acceleration or deceleration segment driven by Calibrated Air Speed (speed_mode="cas" or Mach (speed_mode="mach"
        state = [vtas,mass]
        """
        throttle = 1.
        vtas = state[0]
        mass = state[1]

        acc,fn,ff,cz,cx,pamb,tamb = Flight.acceleration(self,nei,altp,disa,speed_mode,speed,mass,rating,throttle, full_output=True)

        state_d = np.zeros(2)
        state_d[0] = acc
        state_d[1] = -ff

        return state_d

    def event_speed(self,t,state, rating,nei,altp,disa,speed_mode,speed, speed_limit):
        """Detect speed limit
        state = [vtas,mass]
        """
        vtas = state[0]
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        mach_limit = self.get_mach(pamb,speed_mode,speed_limit)
        mach = vtas / earth.sound_speed(tamb)
        return mach_limit-mach

