#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Thomas Ligeois, Thierry DRUOT
"""

import numpy as np
from scipy.optimize import fsolve

import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

import unit, utils
from physical_data import PhysicalData



class DragPolar(object):
    """Provide a drag polar from very few data from the airplane and one polar point
    Drag polar includes a simple Reynolds effect
    Drag polar does not include compressibility effect
    """
    def __init__(self):

        # Airplane geometrical data
        self.wing_area = 42         # Wing reference area
        self.aspect_ratio = 13      # Wing aspect ratio
        self.body_width = 2         # Fuselage width

        # cruise point definition
        self.disa = 0
        self.altp = unit.m_ft(10000)
        self.vtas = unit.mps_kmph(210)

        self.cz_crz = 0.72127       # Cruise lift coefficient
        self.lod_crz = 17.8044       # Cruise lift to drag ratio

        # Additional parameters
        self.wing_span = np.sqrt(self.wing_area*self.aspect_ratio)
        self.wing_mac = self.wing_area / self.wing_span

        pamb,tamb,g = self.atmosphere(self.altp, self.disa)
        rho = self.gas_density(pamb,tamb)
        mu = self.air_viscosity(tamb)
        re = rho*self.vtas/mu
        cx_crz = self.cz_crz / self.lod_crz

        # Drag polar characteristics
        self.kre = 1e3*(1/np.log(re*self.wing_mac))**2.58
        self.kind = (1.05 + (self.body_width / self.wing_span)**2) / (np.pi * self.aspect_ratio)
        self.cx0 = cx_crz - self.kind*self.cz_crz**2

    def get_cx(self, pamb, tamb, vtas, cz):
        rho = self.gas_density(pamb,tamb)
        mu = self.air_viscosity(tamb)
        re = rho*vtas/mu
        kr = 1e3*(1/np.log(re*self.wing_mac))**2.58
        cx = self.cx0*(kr/self.kre) + self.kind*cz**2
        q = 0.5*rho*vtas**2
        drag = q*self.wing_area*cx
        return cx,drag

    def atmosphere(self, altp, disa=0.):
        """Ambient data from pressure altitude from ground to 50 km according to Standard Atmosphere
        """
        g, r, gam = 9.80665, 287.053, 1.4
        Z = [0., 11000., 20000., 32000., 47000., 50000.]
        dtodz = [-0.0065, 0., 0.0010, 0.0028, 0.]
        P = [101325., 0., 0., 0., 0., 0.]
        T = [288.15, 0., 0., 0., 0., 0.]
        if (Z[-1] < altp):
            raise Exception("atmosphere, altitude cannot exceed "+str(Z[-1]+" m"))
        j = 0
        while (Z[1+j] <= altp):
            T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
            if (0. < np.abs(dtodz[j])):
                P[j+1] = P[j]*(1. + (dtodz[j]/T[j]) * (Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
            else:
                P[j+1] = P[j]*np.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
            j = j + 1
        if (0. < np.abs(dtodz[j])):
            pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
        else:
            pamb = P[j]*np.exp(-(g/r)*((altp-Z[j])/T[j]))
        tamb = T[j] + dtodz[j]*(altp-Z[j]) + disa
        return pamb, tamb, g

    def gas_density(self, pamb,tamb):
        """Ideal gas density
        """
        r = 287.053
        rho = pamb / ( r * tamb )
        return rho

    def sound_speed(self, tamb):
        """Sound speed for ideal gas
        """
        r, gam = 287.053, 1.4
        vsnd = np.sqrt( gam * r * tamb )
        return vsnd

    def air_viscosity(self, tamb):
        """Mixed gas dynamic viscosity, Sutherland's formula
        WARNING : result will not be accurate if gas is mixing components of too different molecular weights
        """
        mu0,T0,S = 1.715e-5, 273.15, 110.4
        mu = (mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
        return mu



if __name__ == '__main__':

    dp = DragPolar()

    disa = 0
    altp = unit.m_ft(17700)
    vtas = unit.mps_kmph(210)

    pamb,tamb,_ = dp.atmosphere(altp, disa)

    cz = 0.984202
    cx,_ = dp.get_cx(pamb, tamb, vtas, cz)

    print("")
    print(cz/cx)








