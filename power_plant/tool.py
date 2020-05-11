#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np
from scipy import interpolate

from context import unit



class Material(object):

    def __init__(self):
        self.concrete = 0.
        self.iron = 0.
        self.steel = 0.
        self.aluminium = 0.
        self.copper = 0.
        self.lead = 0.
        self.silicon = 0.
        self.plastic = 0.
        self.fiber_glass = 0.
        self.glass = 0.
        self.oil = 0.
        self.Na2CO3 = 0.
        self.KNO3 = 0.
        self.quinone = 0.

    def grey_energy(self):
        """Embodied energy of materials  (in J/kg)
        Source:
        G.P.Hammond and C.I.Jones (2006) Embodied energy and carbon footprint database, Department of Mechanical Engineering, University of Bath, United Kingdom
        Embodied energy in thermal energy storage (TES) systems for high temperature applications
        Given data are in standard units (J/kg)
        """
        data = {"concrete":1.11e6,
                "iron":25.0e6,
                "steel":20.1e6,
                "aluminium":155.0e6,
                "copper":42.0e6,
                "lead":25.2e6,
                "silicon":15.0e6,
                "plastic":75.0e6,
                "fiber_glass":28.0e6,
                "glass":15.0e6,
                "oil":50.0e6,
                "Na2CO3":16.0e6,
                "KNO3":16.0e6,
                "quinone":10.0e6
                }
        grey_energy = 0.
        for m in data.keys():
            grey_energy += data[m] * getattr(self, m, 0.)
        return grey_energy

    def Scarcity(self, type):
        """World wide stocks of some minerals (in kg)
        Source : Wikipedia
        """
        data = {"sand": 192.e18,
                "iron": 87.e12,
                "aluminium": 28.e12,
                "copper": 630.e9
                }
        return data.get(type,None)



def max_solar_power(latt,long,pamb,day,gmt):
    """Compute max solar radiative power from location and time on Earth

    :param latt: Lattitude in radians
    :param long: Longitude in radians
    :param day: Day of the year, from 1 to 365
    :param gmt: GMT time in the day, from 0. to 24.
    :return:
    """
    delta = unit.rad_deg(23.45 * np.sin(unit.rad_deg((284.+day)*(360./365.))))
    equ = 0. # output of time equation, neglected here
    solar_time = gmt + (unit.deg_rad(long)*(4./60.)) - equ  # Solar time
    eta = unit.rad_deg((360./24.)*(solar_time - 12.))       # Time angle
    sin_a = np.sin(latt) * np.sin(delta) + np.cos(latt)*np.cos(delta)*np.cos(eta)
    alpha = np.arcsin(sin_a)                                # Sun elevation
    ref_solar_pw = 1367.                                    # Reference solar power
    pw_out = ref_solar_pw * (1. + 0.034*np.cos(unit.rad_deg(day*(360./365.))))
    m0 = np.sqrt(1229. + (614.*sin_a)**2) - 614.*sin_a      # Absorbtion coefficient
    p0 = 101325.                                            # Sea level reference pressure
    m = m0*(pamb/p0)                                        # Influence of altitude on the absorbtion coefficient
    tau = 0.6                                               # Transmission coefficient
    pw_direct = pw_out * tau**m * sin_a                     # Direct solar radiative power
    pw_diffus = pw_out * (0.271 - 0.294*tau**m) * sin_a     # Diffused  solar radiative power
    if (alpha>unit.rad_deg(3.)):
        pw_total = pw_direct + pw_diffus
    else:
        pw_total = 0.
    return pw_total


def mean_yearly_sun_power(latt):
    """Compute max mean yearly solar radiative power from latitude on Earth
    """
    lat = [  0.0,  5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0, 80.0, 85.0, 90.0]
    pwr = [256.0,254.6,250.7,244.5,236.1,225.7,213.4,199.5,184.2,168.0,151.3,134.7,118.7,103.8, 91.6, 82.2, 75.2, 71.0, 69.6]
    sun_pwr = interpolate.interp1d(lat,pwr,kind="cubic")
    return sun_pwr(latt)



latt = unit.rad_deg(43.668731)
long = unit.rad_deg(1.497691)
pamb = 101325.
#     JA FE MA AV MA JU
day = 31+29+31+30+31+30
gmt = 12.

pw = max_solar_power(latt,long,pamb,day,gmt)
print("")
print("Solar power = ","%8.1f" % pw)


#
# md_pw = 0.
# period = 24*60
# for t in range(period):
#     md_pw += max_solar_power(latt,long,pamb,day,float(t)/60.)/period
# print("")
# print("Mean dayly solar power = ","%8.1f" % md_pw)
#
#
# year = 365
# period = 24*60
# for l in range(91):
#     my_pw = 0.
#     for y in range(year):
#         for t in range(period):
#             latt = unit.rad_deg(l)
#             my_pw += max_solar_power(latt,long,pamb,y+1,float(t)/60.)/period/year
#     print("Lat = ", "%8.1f"%l, " , Pw = ", "%8.1f"%my_pw)
#

