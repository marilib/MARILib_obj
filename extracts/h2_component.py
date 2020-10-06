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
tank_efficiency = Pam3pkg_barLpkg(250.)      # bar.L/kg,  pressurized tank efficiency

shield_density = 1400.          # kg/m3, pressure material density

insul_density = 200.            # kg/m3, insulation material density
insul_thick = 0.06              # m, insulation thickness

lh2_density = 71.               # kg/m3, liquid H2 density

over_pressure = Pa_bar(11.)     # bar, fuel delta pressure

length = 12.    # m, tank length
width = 2.5     # m, tank diameter

# Model
#-----------------------------------------------------------------------------------------------
ext_volume = 0.80 * length * (0.25*np.pi*width**2)  # tank external volume
ext_area = 0.80 * length * (np.pi*width)            # tank external area

# pressurized volume, assuming insulation is inside the volume
press_volume = ext_volume / (1. + over_pressure/(tank_efficiency*shield_density))

psm = over_pressure * press_volume / tank_efficiency    # pressure structural mass

shell_volume = psm / shield_density                     # pressure shell volume

pst = shell_volume / ext_area                           # pressure structural thickness

pia = ext_area * (1.-2.*pst/length) * (1.-2.*pst/width) # pressurized insulated area

insul_volume = pia*insul_thick                          # insulation volume

lh2_volume = press_volume - insul_volume    # LH2 volume
lh2_mass = lh2_volume * lh2_density         # LH2 mass

ism = insul_volume * insul_density          # insulation structural mass
total_mass = psm + ism                      # total tank mass
total_volume = shell_volume + insul_volume  # total tank volume

tds = total_mass / total_volume     # global tank density

rtm = total_mass / lh2_mass         # relative tank mass, tank mass per lh2 mass (without carrying structure)

# Print
#-----------------------------------------------------------------------------------------------
print("tank external volume = ", "%0.1f"%ext_volume, " m3")
print("tank external area = ", "%0.1f"%ext_area, " m2")
print("pressurized volume = ", "%0.1f"%press_volume, " m3")
print("liquid H2 volume = ", "%0.1f"%lh2_volume, " m3")
print("liquid H2 mass = ", "%0.1f"%lh2_mass, " kg")
print("pressure structural mass = ", "%0.1f"%psm, " kg")
print("pressure structural thickness = ", "%0.3f"%pst, " m")
print("insulation structural mass = ", "%0.1f"%ism, " kg")
print("tank mass = ", "%0.1f"%total_mass, " kg")
print("tank material density = ", "%0.1f"%tds, " kg/m3")
print("tank mass over LH2 mass = ", "%0.3f"%rtm, " kg/kg")


