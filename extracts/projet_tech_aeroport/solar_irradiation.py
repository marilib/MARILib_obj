#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np
from scipy import interpolate

import unit



def max_solar_power(latt,long,pamb,day,gmt):
    """Compute max solar radiative power from location and time on Earth

    :param latt: Lattitude in radians
    :param long: Longitude in radians
    :param day: Day of the year, from 1 to 365
    :param gmt: GMT time in the day, from 0. to 24.
    :return:
    """
    delta = unit.rad_deg(23.45 * np.sin(unit.rad_deg((284. + day) * (360. / 365.))))
    equ = 0. # output of time equation, neglected here
    solar_time = gmt + (unit.deg_rad(long) * (4. / 60.)) - equ  # Solar time
    eta = unit.rad_deg((360. / 24.) * (solar_time - 12.))       # Time angle
    sin_a = np.sin(latt) * np.sin(delta) + np.cos(latt)*np.cos(delta)*np.cos(eta)
    alpha = np.arcsin(sin_a)                                # Sun elevation
    ref_solar_pw = 1367.                                    # Reference solar power
    pw_out = ref_solar_pw * (1. + 0.034 * np.cos(unit.rad_deg(day * (360. / 365.))))
    m0 = np.sqrt(1229. + (614.*sin_a)**2) - 614.*sin_a      # Absorbtion coefficient
    p0 = 101325.                                            # Sea level reference pressure
    m = m0*(pamb/p0)                                        # Influence of altitude on the absorbtion coefficient
    tau = 0.6                                               # Transmission coefficient
    pw_direct = pw_out * tau**m * sin_a                     # Direct solar radiative power
    pw_diffus = pw_out * (0.271 - 0.294*tau**m) * sin_a     # Diffused  solar radiative power
    if (alpha> unit.rad_deg(3.)):
        pw_total = pw_direct + pw_diffus
    else:
        pw_total = 0.
    return pw_total



if __name__ == "__main__":

    latt = unit.rad_deg(43.668731)
    long = unit.rad_deg(1.497691)
    pamb = 101325.

    #     JA FE MA AV MA JU
    day = 31+29+31+30+31+30
    gmt = 12.

    pw = max_solar_power(latt,long,pamb,day,gmt)

    print("")
    print("Solar power = ","%8.1f"%pw, " W")

