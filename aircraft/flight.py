#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from aircraft.tool import unit
import earth

import numpy as np
from scipy.optimize import fsolve


def get_speed(pamb,speed_mode,mach):
    """
    retrieves CAS or mach from mach depending on speed_mode
    """
    speed = {1 : earth.vcas_from_mach(pamb,mach),   # CAS required
             2 : mach                               # mach required
             }.get(speed_mode, "Erreur: select speed_mode equal to 1 or 2")
    return speed


def get_mach(pamb,speed_mode,speed):
    """
    Retrieves mach from CAS or mach depending on speed_mode
    """
    mach = {1 : earth.mach_from_vcas(pamb,speed),   # Input is CAS
            2 : speed                               # Input is mach
            }.get(speed_mode, "Erreur: select speed_mode equal to 1 or 2")
    return mach


def speed_from_lift(aircraft,pamb,tamb,cz,mass):
    """
    Retrieves mach from cz using simpliffied lift equation
    """
    g = earth.gravity()
    r,gam,Cp,Cv = earth.gas_data()
    mach = np.sqrt((mass*g)/(0.5*gam*pamb*aircraft.airframe.wing.area*cz))
    return mach


def lift_from_speed(aircraft,pamb,tamb,mach,mass):
    """
    Retrieves cz from mach using simpliffied lift equation
    """
    g = earth.gravity()
    r,gam,Cp,Cv = earth.gas_data()
    cz = (2.*mass*g)/(gam*pamb*mach**2*aircraft.airframe.wing.area)
    return cz


def level_flight(aircraft,pamb,tamb,mach,mass):
    """
    Level flight equilibrium
    """
    g = earth.gravity()
    r,gam,Cp,Cv = earth.gas_data()

    cz = (2.*mass*g)/(gam*pamb*mach**2*aircraft.airframe.wing.area)
    cx,lod = aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

    fn = (gam/2.)*pamb*mach**2*aircraft.airframe.wing.area*cx
    sfc,throttle = aircraft.power_system.sc(pamb,tamb,mach,"MCR",fn)
    if (throttle>1.): print("level_flight, throttle is higher than 1, throttle = ",throttle)

    return cz,cx,lod,fn,sfc,throttle


def air_path(aircraft,nei,altp,disa,speed_mode,speed,mass,rating):
    """
    Retrieves air path in various conditions
    """
    g = earth.gravity()
    pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
    mach = get_mach(pamb,speed_mode,speed)

    fn,sfc,sec,data = aircraft.power_system.thrust(aircraft,pamb,tamb,mach,rating)
    cz = lift_from_speed(aircraft,pamb,tamb,mach,mass)
    cx,lod = aircraft.aerodynamics.drag(aircraft,pamb,tamb,mach,cz)

    if(nei>0):
        dcx = aircraft.power_system.oei_drag(aircraft,pamb,mach)
        cx = cx + dcx*nei
        lod = cz/cx

    acc_factor = earth.climb_mode(speed_mode,dtodz,tstd,disa,mach)
    slope = ( fn/(mass*g) - 1/lod ) / acc_factor
    vz = mach*slope*earth.sound_speed(tamb)

    return slope,vz

