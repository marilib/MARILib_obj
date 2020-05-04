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

import pandas

import matplotlib.pyplot as plt

from context import unit, math




def atmosphere(altp, disa=0.):
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
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j]) *
                           (Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
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


# ======================================================================================================
# Packaging data
# ------------------------------------------------------------------------------------------------------
file = "../input_data/Aircraft_general_data.csv"

# Loading csv file
# ------------------------------------------------------------------------------------------------------
data_frame = pandas.read_csv(file, delimiter = ";", skipinitialspace=True, header=None)

# Extracting info
# ------------------------------------------------------------------------------------------------------
label = [el.strip() for el in data_frame.iloc[0,1:].values]     # Variable names
unit_ = [el.strip() for el in data_frame.iloc[1,1:].values]     # Figure units
name = [el.strip() for el in data_frame.iloc[2:,0].values]      # Aircraft names
data = data_frame.iloc[2:,1:].values                            # Data matrix

# Packaging figures into a dictionary
# ------------------------------------------------------------------------------------------------------
data_dict = {}

n = len(name)   # the number of aircraft

# Data matrix is spread over a dictionary column by column
# Dictionary allows to address each column by the name of the corresponding variable
# for instance dict["Npax"] retrieves an array with all the realizations of Npax
for j in range(len(label)):
    data_dict[label[j]] = data[:,j].astype(np.float)

# ======================================================================================================
# Identifing structure model
# ------------------------------------------------------------------------------------------------------
param = data_dict["MTOW"]

A = np.vstack([param**2,param, np.ones(n)]).T                # Need to transpose the stacked matrix

B = data_dict["OWE"]

(C, res, rnk, s) = np.linalg.lstsq(A, B, rcond=None)

# Main results
#------------------------------------------------------------------------------------------------------
print(C)

AC = np.dot(A,C)

res = np.sqrt(np.sum((AC-B)**2))

print("%.0f"%res)

# Graph
#------------------------------------------------------------------------------------------------------
plt.plot(B, AC, 'o', color="red", markersize=2)
plt.plot(B, B, 'green')
plt.grid(True)
plt.suptitle('Structure : residual = '+"%.0f"%res, fontsize=14)
plt.ylabel('Approximations (kg)')
plt.xlabel('Experiments (kg)')
plt.savefig("Structure",dpi=500,bbox_inches = 'tight')
plt.show()

# Result
#------------------------------------------------------------------------------------------------------
def operating_empty_weight(mtow):
    owe = (-1.478e-07*mtow + 5.459e-01)*mtow + 8.40e+02
    return owe



# ======================================================================================================
# Airplane object definition
# ------------------------------------------------------------------------------------------------------
class Aircraft(object):
    """Plane object
    """
    def __init__(self,
                 npax=150.,
                 range=unit.m_NM(3000.),
                 cruise_mach=0.78):
        self.npax = npax  # Npax
        self.range = range  # Range
        self.cruise_mach = cruise_mach  # Speed
        self.cruise_altp = unit.m_ft(35000.)
        self.mpax = 135.    # weight per passenger
        self.payload = None
        self.mtow = None  # MTOW
        self.fuel_mission = None
        self.fuel_reserve = None
        self.ldw = None
        self.owe = None  # OWE
        self.lod = 20.                                   # Techno assumption
        self.sfc = unit.convert_from("kg/daN/h", 0.53)   # Techno assumption
        self.eff_ratio = self.lod / self.sfc
        self.owe_coef = [8.40e+02, 5.459e-01, -1.478e-07]               # Techno assumption
        self.kr = 0.03      # fraction of mission fuel for reserve

    def structure(self, mtow, coef=None):
        """Structural relation
        """
        if coef is not None: self.owe_coef = coef
        owe = (self.owe_coef[2]*mtow + self.owe_coef[1]) * mtow + self.owe_coef[0]    # Structure design rule
        return owe

    def mission(self, mtow, fuel_mission, effr=None):
        """Mission relation
        Warning : if given effr must be expressed in daN.h/kg
        """
        if effr is not None: self.eff_ratio = effr / unit.convert_from("kg/daN/h", 1.)
        pamb, tamb, vsnd, g = atmosphere(self.cruise_altp)
        range_factor = (self.cruise_mach*vsnd*self.eff_ratio)/g
        # Breguet equation
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
        return tow,fuel_mission

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



# ======================================================================================================
# Packaging data
# ------------------------------------------------------------------------------------------------------
file = "../input_data/Aircraft_general_data.csv"

# Loading csv file
# ------------------------------------------------------------------------------------------------------
data_frame = pandas.read_csv(file, delimiter = ";", skipinitialspace=True, header=None)

# Extracting info
# ------------------------------------------------------------------------------------------------------
label = [el.strip() for el in data_frame.iloc[0,1:].values]     # Variable names
unit_ = [el.strip() for el in data_frame.iloc[1,1:].values]     # Figure units
name = [el.strip() for el in data_frame.iloc[2:,0].values]      # Aircraft names
data = data_frame.iloc[2:,1:].values                            # Data matrix

# Packaging figures into a dictionary
# ------------------------------------------------------------------------------------------------------
data_dict = {}

n = len(name)   # the number of aircraft

# Data matrix is spread over a dictionary column by column
# Dictionary allows to address each column by the name of the corresponding variable
# for instance dict["Npax"] retrieves an array with all the realizations of Npax
for j in range(len(label)):
    data_dict[label[j]] = data[:,j].astype(np.float)

# ======================================================================================================
# Identifing mission model
# ------------------------------------------------------------------------------------------------------
ac = Aircraft()

B = data_dict["MTOW"]
n = len(B)

def model(x_in):
    R = []
    for j in range(n):
        ac.npax = data_dict["Npax"][j]
        ac.range = data_dict["Range"][j]*1000.
        ac.cruise_mach = data_dict["Speed"][j]
        #ac.design_aircraft(effr=x_in[0], mpax=x_in[1], kr=x_in[2])
        ac.design_aircraft(effr=x_in)
        R.append(ac.mtow)
    return R

def residual(x_in):
    return model(x_in)-B

def objective(x_in):
    return 1.e4/np.sqrt(np.sum(model(x_in)-B)**2)

x0 = ac.lod / unit.convert_to("kg/daN/h", ac.sfc)

x_out = x0

# x_out, y_out, rc = math.maximize_1d(x0, 0.1, [objective])

# Main results
#------------------------------------------------------------------------------------------------------
print(x_out)

R = model(x_out)

res = np.sqrt(np.sum((R-B)**2))

print("%.0f"%res)

# Graph
#------------------------------------------------------------------------------------------------------
plt.plot(B, R, 'o', color="red", markersize=2)
plt.plot(B, B, 'green')
plt.grid(True)
plt.suptitle('Design : residual = '+"%.0f"%res, fontsize=14)
plt.ylabel('Approximations (kg)')
plt.xlabel('Experiments (kg)')
plt.savefig("Design",dpi=500,bbox_inches = 'tight')
plt.show()






