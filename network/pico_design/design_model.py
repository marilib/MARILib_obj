#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot, Weichang LYU
"""

import numpy as np
from scipy.optimize import fsolve, least_squares

import matplotlib.pyplot as plt

from context import unit, math



# ======================================================================================================
# Airplane object definition
# ------------------------------------------------------------------------------------------------------
class Aircraft(object):
    """Plane object
    """
    def __init__(self,
                 npax=150.,
                 range=unit.m_NM(3000.),
                 mach=0.78):
        self.cruise_altp = unit.m_ft(35000.)    # Reference cruise altitude
        self.cruise_mach = mach     # Speed
        self.range = range          # Range
        self.npax = npax            # Npax
        self.mpax = 130.            # Weight per passenger
        self.payload = None         # Design mission payload
        self.mtow = None            # Design mission Maximum Take Off Weight
        self.owe = None             # Design mission Operating Empty Weight
        self.ldw = None             # Design mission Landing Weight
        self.fuel_mission = None    # Design mission fuel
        self.fuel_reserve = None    # Design mission reserve fuel
        self.kr = 0.03              # fraction of mission fuel for reserve

        self.payload_max = None     # Maximum payload
        self.range_pl_max = None    # Range for maximum payload mission

        self.payload_fuel_max = None    # Payload for max fuel mission
        self.range_fuel_max = None      # Range for max fuel mission

        self.range_no_pl = None     # Range for zero payload mission

        self.lod = 20.                                  # Techno assumption
        self.sfc = unit.convert_from("kg/daN/h", 0.54)  # Techno assumption

        self.eff_ratio = self.lod / self.sfc            # Efficiency ratio for specific air range
        self.owe_coef = [-1.478e-07, 5.459e-01, 8.40e+02]   # "Structural model"

        self.design_aircraft()

    def structure(self, mtow, coef=None):
        """Structural relation
        """
        if coef is not None: self.owe_coef = coef
        owe = (self.owe_coef[0]*mtow + self.owe_coef[1]) * mtow + self.owe_coef[2]    # Structure design rule
        return owe

    def mission(self, mtow, fuel_mission, effr=None):
        """Mission relation
        Warning : if given effr must be expressed in daN.h/kg
        """
        if effr is not None: self.eff_ratio = effr / unit.convert_from("kg/daN/h", 1.)
        pamb, tamb, vsnd, g = self.atmosphere(self.cruise_altp)
        range_factor = (self.cruise_mach*vsnd*self.eff_ratio)/g
        range = range_factor*np.log(mtow/(mtow-fuel_mission))       # Breguet equation
        return range

    def operation(self, range, n_pax):
        """Operational mission

        :param range: Distance to fly
        :param n_pax: Number of passengers
        :return: TOW, Mission fuel
        """
        def fct(x_in):
            tow = x_in[0]
            fuel_mission = x_in[1]
            range_eff = self.mission(tow, fuel_mission)
            owe_eff = tow - (self.owe + self.mpax*n_pax + (1.+self.kr)*fuel_mission)
            return np.array([self.owe-owe_eff, range-range_eff])

        x_ini = self.npax*self.mpax*np.array([4., 1.])
        dict = fsolve(self.eval_design, x0=x_ini, full_output=True)
        if (dict[2] != 1): raise Exception("Convergence problem")
        tow = dict[0][0]
        fuel_mission = dict[0][1]
        return fuel_mission,tow

    def eval_design(self, X):
        """Evaluation function for design_plane
        """
        self.mtow = X[0]
        self.fuel_mission = X[1]

        owe_eff = self.structure(self.mtow)  # 1
        range_eff = self.mission(self.mtow, self.fuel_mission)  # 2

        self.fuel_reserve = self.kr*self.fuel_mission  # 3
        self.ldw = self.mtow - self.fuel_mission
        self.payload = self.npax * self.mpax  # 4
        self.owe = self.mtow - self.payload - self.fuel_mission - self.fuel_reserve  # 5
        return np.array([self.owe-owe_eff,self.range-range_eff])

    def design_aircraft(self, coef=None, kr=None, mpax=None, effr=None):
        """Design method (mass-mission adaptation only
        Warning : if given effr must be expressed in daN.h/kg
        """
        if coef is not None: self.owe_coef = coef
        if kr is not None: self.kr = kr
        if mpax is not None: self.mpax = mpax
        if effr is not None: self.eff_ratio = effr /unit.convert_from("kg/daN/h", 1.)

        Xini = self.npax*self.mpax*np.array([4., 1.])
        dict = fsolve(self.eval_design, x0=Xini, full_output=True)
        if (dict[2] != 1): raise Exception("Convergence problem")

        self.eval_design(np.array([dict[0][0], dict[0][1]]))

        self.payload_max = self.payload * 1.20
        fuel = (self.mtow - self.owe - self.payload_max) / (1.+self.kr)
        self.range_pl_max = self.mission(self.mtow, fuel)

        self.payload_fuel_max = self.payload * 0.40
        fuel_max = (self.mtow - self.owe - self.payload_fuel_max) / (1.+self.kr)
        self.range_fuel_max = self.mission(self.mtow, fuel_max)

        tow = self.owe + fuel_max * (1.+self.kr)
        self.range_no_pl = self.mission(tow, fuel_max)

    def is_in_plr(self, npax, range):
        """Assess if a mission is possible
        """
        payload = npax * self.mpax
        c1 = self.payload_max - payload
        c2 =  (payload-self.payload_fuel_max)*(self.range_pl_max-self.range_fuel_max) \
            - (self.payload_max-self.payload_fuel_max)*(range-self.range_fuel_max)
        c3 = payload*(self.range_fuel_max-self.range_no_pl) - self.payload_max*(range-self.range_no_pl)
        if (c1>=0. and c2>=0. and c3>=0.): return True
        else: return False

    def atmosphere(self, altp, disa=0.):
        """Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
        """
        g = 9.80665
        r = 287.053
        gam = 1.4

        Z = np.array([0., 11000., 20000., 32000., 47000., 50000.])
        dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])
        P = np.array([101325., 0., 0., 0., 0., 0.])
        T = np.array([288.15, 0., 0., 0., 0., 0.])

        if (Z[-1] < altp):
            raise Exception("atmosphere, altitude cannot exceed 50km")

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
        vsnd = np.sqrt(gam*r*tamb)
        return pamb, tamb, vsnd, g

    def payload_range(self):
        """Print the payload - range diagram
        """
        payload = [self.payload_max,
                   self.payload_max,
                   self.payload_fuel_max,
                   0.]

        range = [0.,
                 unit.NM_m(self.range_pl_max),
                 unit.NM_m(self.range_fuel_max),
                 unit.NM_m(self.range_no_pl)]

        nominal = [self.payload,
                   unit.NM_m(self.range)]

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title("Pico Design")
        fig.suptitle("Payload - Range", fontsize=14)

        plt.plot(range,payload,linewidth=2,color="blue")
        plt.scatter(range[1:],payload[1:],marker="+",c="orange",s=100)
        plt.scatter(nominal[1],nominal[0],marker="o",c="green",s=50)

        plt.grid(True)

        plt.ylabel('Payload (kg)')
        plt.xlabel('Range (NM)')

        plt.show()






