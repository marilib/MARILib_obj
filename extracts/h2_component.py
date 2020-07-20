#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np


def Pam3pkg_barLpkg(barLpkg): return barLpkg*1.e2  # Translate bar.L/kg into Pa.m3/kg
def barLpkg_Pam3pkg(Pam3pkg): return Pam3pkg/1.e2  # Translate Pa.m3/kg into bar.L/kg

def Pa_bar(bar): return bar*1.e5   # Translate bar into Pascal
def bar_Pa(Pa): return Pa/1.e5   # Translate Pascal into bar

# Data
#-----------------------------------------------------------------------------------------------
sp = Pam3pkg_barLpkg(250.)      # bar.L/kg,  pressurized tank efficiency

sm = 1400.                      # kg/m3, pressure material density

sim = 5.                        # kg/m2, insulation material surface mass
sit = 0.06                      # m, insulation thickness

dlh2 = 71.                      # kg/m3, liquid H2 density

dp = Pa_bar(11.)                # bar, fuel delta pressure

length = 12.    # m, tank length
width = 2.5     # m, tank diameter

# Model
#-----------------------------------------------------------------------------------------------
vext = 0.80 * length * (0.25*np.pi*width**2)    # tank external volume
sext = 0.80 * length * (np.pi*width)            # tank external area

vp = vext / (1. + dp/(sp*sm))   # pressurized volume

mp = dp * vp / sp               # pressure structural mass

pt = (mp / sm) / sext           # pressure structural thickness

sp = sext * (1.-2.*pt/length) * (1.-2.*pt/width)    # pressurized external area

vlh2 = vp - sp*sit      # LH2 volume
mlh2 = vlh2 * dlh2      # LH2 mass

mi = sp * sim           # insulation structural mass
mpi = mp + mi           # total mass

mr = mpi / mlh2         # tank mass per lh2 mass (without carrying structure)

# Print
#-----------------------------------------------------------------------------------------------
print("tank external volume = ", "%0.1f"%vext, " m3")
print("tank external area = ", "%0.1f"%sext, " m2")
print("pressurized volume = ", "%0.1f"%vp, " m3")
print("liquid H2 volume = ", "%0.1f"%vlh2, " m3")
print("liquid H2 mass = ", "%0.1f"%mlh2, " kg")
print("pressure structural mass = ", "%0.1f"%mp, " kg")
print("pressure structural thickness = ", "%0.3f"%pt, " m")
print("insulation structural mass = ", "%0.1f"%mi, " kg")
print("tank mass = ", "%0.1f"%mpi, " kg")
print("tank mass over LH2 mass = ", "%0.3f"%mr, " kg/kg")


