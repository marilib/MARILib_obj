#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019
@author: DRUOT Thierry
============================================================================================
Very small  MDO test case

Optimize a 150 passengers aircraft that flies at Mach 0.78

The optimization problem to be solved :
Minimiser : mtow
wrt : area, ar, slst    100 < area < 200, 5 < ar < 20, 100e5 < slst < 200e5
st : tofl < 2000 m
     vapp < 135 kt
     300 ft/min < vz

Remarks :

the mtow and owe variables create a strong coupling through
the mass and mission, and mass_coupling
functions

The unit.py file allows to convert units when needed


"area":["m2","Wing geometrical reference area"],
"ar":["no_dim","Wing aspect ratio"],
"lod":["no_dim","Cruise lift to drag ratio"],
"cl_max_to":["no_dim","Maximum lift coefficient at take off"],
"cl_max_ld":["no_dim","Maximum lift coefficient at landing"],
"brp":["no_dim","Engine by pass ratio"],
"slst":["daN","Engine reference static thrust"],
"sfc":["kg/daN/h","Engine specific fuel consumption"],
"fn_ton":["daN","Engine reference thrust in take off conditions"],
"fn_mcl":["daN","Engine reference thrust in climb conditions"],
"mtow":["kg","Maximum take off weight"],
"mlw":["kg","Maximum landing weight"],
"mzfw":["kg","Maximum zero fuel weight"],
"owe":["kg","Operating empty weight"],
"payload":["kg","Nominal payload"],
"range":["NM","Nominal mission range"],
"fuel":["kg","Nominal mission fuel"],
"tofl":["m","Take off field length at mtow"],
"vapp":["kt","Approach speed at mlw"],
"vz":["ft/min","Vertical speed top of climb"],
"""

import math


def aerodynamic(area, ar):
    lod = 0.5 / math.sqrt((0.334 / ar) * (0.0137 + 1.15 / area))
    hld_to = 0.3
    cl_max_ld = 2.3
    cl_max_to = (1 - hld_to) * 2.00 + hld_to * cl_max_ld
    return lod, cl_max_to, cl_max_ld


def engine(bpr, slst):
    sfc = (0.4 + 1 / bpr**0.895) / 36000
    fn_ton = 0.80 * slst  # MTO rating, SL ISA
    fn_mcl = 0.19 * slst  # MCL rating, 35000 ft; ISA
    return sfc, fn_ton, fn_mcl


def mass(mtow, area, ar, slst):
    ne = 2      # number of engines
    npax = 150  # number of passengers
    payload = 100 * npax
    payload_max = 120 * npax
    owe = 19000 + 0.0893 * mtow + 77 * area + 95 * ar**1.5 + 0.0272 * slst * ne
    mzfw = owe + payload_max
    mlw = 1.07 * mzfw
    return payload, owe, mzfw, mlw


def mission(mtow, range, sfc, lod):
    g = 9.81
    tas = 230     # (m/s, mach 0.78, 35000ft, ISA)
    fuel = mtow * (1 - math.exp(-(sfc * g * range) / (tas * lod * 0.95)))
    return fuel


def mtow_coupling(owe, payload, fuel):
    mtow = owe + payload + fuel
    return mtow


def take_off(mtow, fn_ton, area, cl_max_to):
    ne = 2      # number of engines
    sig = 1     # Sea level
    kvs1g_to = 1.13
    cl_to = cl_max_to / kvs1g_to**2
    kml = mtow**2 / (cl_to * ne * fn_ton * area * sig**0.8)
    tofl = 14.23 * kml + 97.58
    return tofl


def approach(mlw, area, cl_max_ld):
    g = 9.81
    rho = 1.225
    kvs1g_ld = 1.23
    cl_ld = cl_max_ld / kvs1g_ld**2
    vapp = math.sqrt((mlw * g) / (0.5 * rho * area * cl_ld))
    return vapp


def climb(mtow, fn_mcl, lod):
    g = 9.81
    ne = 2      # number of engines
    mach = 0.78
    tas = mach * 297      # (m/s, mach 0.78, 35000ft, ISA)
    mass = 0.97 * mtow    # Top of climb mass
    fac = 1 + 20.49029021 * mach**2 * (-0.0065)  # Constant Mach climb
    vz = tas * (ne * fn_mcl / (mass * g) - 1 / lod) / fac
    return vz
