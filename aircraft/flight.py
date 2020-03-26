#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import earth


def level_flight(aircraft,pamb,tamb,mach,mass):
    g = earth.gravity()
    r,gam,Cp,Cv = earth.gas_data()

    cz = (2.*mass*g)/(gam*pamb*mach**2*aircraft.airframe.wing.area)
    cx,lod = aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

    fn = (gam/2.)*pamb*mach**2*aircraft.airframe.wing.area*cx
    sfc,throttle = aircraft.power_system.sc(pamb,tamb,mach,"MCR",fn)
    if (throttle>1.): print("level_flight, throttle is higher than 1, throttle = ",throttle)

    return cz,cx,lod,fn,sfc,throttle



