#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 20 20:20:20 2020
@author: DRUOT Thierry
"""

import numpy
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from marilib.utils import unit

#===========================================================================================================
def atmosphere(altp,disa=0.):
    """
    Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
    """
    g = 9.80665
    r = 287.053
    gam = 1.4

    Z = numpy.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = numpy.array([-0.0065, 0., 0.0010, 0.0028, 0.])
    P = numpy.array([101325., 0., 0., 0., 0., 0.])
    T = numpy.array([288.15, 0., 0., 0., 0., 0.])

    if (Z[-1]<altp):
        raise Exception("atmosphere, altitude cannot exceed 50km")

    j = 0
    while (Z[1+j]<=altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<numpy.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
        else:
            P[j+1] = P[j]*numpy.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0.<numpy.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
    else:
        pamb = P[j]*numpy.exp(-(g/r)*((altp-Z[j])/T[j]))
    tamb = T[j] + dtodz[j]*(altp-Z[j]) + disa
    vsnd = numpy.sqrt(gam*r*tamb)
    return pamb,tamb,vsnd,g


# Plane object
#--------------------------------------------------------------------------------
class plane(object):
    """
    Drone data
    """
    def __init__(self,
                 npax = None,
                 range = None,
                 cruise_mach = None,
                 mtow = None,
                 payload = None,
                 fuel = None,
                 ldw = None,
                 owe = None):
        self.npax = npax
        self.range = range
        self.cruise_mach = cruise_mach
        self.cruise_altp = unit.m_ft(35000.)
        self.mtow = mtow
        self.payload = payload
        self.fuel = fuel
        self.ldw = ldw
        self.owe = owe
        self.lod = 17.                                  # Techno assumption
        self.sfc = unit.convert_from("kg/daN/h",0.54)   # Techno assumption
        self.owe_a = 0.     #-1.29e-07                  # Techno assumption
        self.owe_b = 0.47  #0.47   #5.38e-01                   # Techno assumption
        self.owe_c = 9100.  #9100.  #2.192e+03                  # Techno assumption
        self.mpax = 105.    # weight per passenger
        self.kr = 0.05      # fraction of mission fuel for reserve

    # Evaluation method
    #--------------------------------------------------------------------------------
    def eval_plane(self,X):
        self.owe = X[0]
        self.fuel = X[1]
        self.payload = self.npax*self.mpax

        self.ldw = self.owe + self.payload + self.kr*self.fuel                  # Definition of LDW
        self.mtow = self.owe + self.payload + self.fuel                         # Definition of MTOW
        owe_eff = (self.owe_a*self.mtow + self.owe_b)*self.mtow + self.owe_c    # Structure design rule
        pamb,tamb,vsnd,g = atmosphere(self.cruise_altp)
        range_eff = 0.95*((self.cruise_mach*vsnd*self.lod)/(g*self.sfc))*numpy.log(self.mtow/self.ldw)   # Breguet

        return numpy.array([self.owe-owe_eff, self.range-range_eff])

    # Design method
    #--------------------------------------------------------------------------------
    def design_plane(self):
        Xini = self.npax*self.mpax*numpy.array([1.,1.])
        dict = fsolve(self.eval_plane, x0=Xini, full_output=True)

        if (dict[2]!=1):
            raise Exception("Convergence problem")

        self.eval_plane(numpy.array([dict[0][0],dict[0][1]]))

    # Print method
    #--------------------------------------------------------------------------------
    def print_plane(self):
        print("Plane data")
        print("------------------------------------------------")
        print("Number of passenger = ","%.0f"%(self.npax))
        print("Nominal range = ","%.0f"%unit.NM_m(self.range)," NM")
        print("Cruise mach = ","%.2f"%(self.cruise_mach))
        print("------------------------------------------------")
        print("MTOW = ","%.0f"%(self.mtow)," kg")
        print("OWE = ","%.0f"%(self.owe)," kg")
        print("Payload = ","%.0f"%(self.payload)," kg")
        print("Mission fuel = ","%.0f"%(self.fuel)," kg")
        print("Landing weight = ","%.0f"%(self.ldw)," kg")


# Create a plane object
#===============================================================================
a = plane()

# Load requirements
#-------------------------------------------------------------------------------
a.npax = 150.
a.range = unit.m_NM(3000.)
a.cruise_mach = 0.78

# Perform design
#-------------------------------------------------------------------------------
a.design_plane()

# Print result
#-------------------------------------------------------------------------------
a.print_plane()






