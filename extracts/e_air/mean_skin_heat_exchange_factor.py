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



def air_thermal_transfer_data(phd, pamb,tamb,air_speed, x):
    """Thermal transfert factor for turbulent air flow
    """
    r,gam,cp,cv = phd.gas_data()
    rho = phd.gas_density(pamb,tamb)
    mu = phd.air_viscosity(tamb)
    alpha = phd.air_thermal_diffusivity()
    pr = mu / (alpha * rho)                         # Prandtl number
    re = rho * air_speed / mu                       # Reynolds number
    if (re*x)<1.e8 and 0.6<pr and pr<60.:
        nu = 0.0296 * (re*x)**(4/5) * pr**(1/3)     # Nusselt number
    else:
        print("Re = ", re*x, "  Pr = ", pr)
        raise Exception("Re or Pr are not in the valid domain")
    lbd = alpha * rho * cp      # Thermal conductivity
    h = lbd * nu / x
    return h, rho, cp, mu, pr, re*x, nu, lbd


phd = PhysicalData()

altp = unit.m_ft(10000)
disa = 15
vair = unit.mps_kmph(200)

pamb, tamb, g = phd.atmosphere(altp, disa)

x0 = 0.8
dx = 0.7
n = 20

def integral_heat_transfer(pamb,tamb,vair, x0, dx, n):
    x_list = np.linspace(0, dx, n)
    x_int = x0
    h_int = 0
    x = 0
    for j in range(n-1):
        d = x_list[j+1] - x_list[j]
        x = x_int + 0.5 * d
        h, rho, cp, mu, pr, re, nu, lbd = air_thermal_transfer_data(phd, pamb,tamb,vair, x)
        x_int += d
        h_int += h * d/dx
    return h_int

h_int = integral_heat_transfer(pamb,tamb,vair, x0, dx, n)
print("")
print("Integral heat transfer factor = ", "%.1f"%h_int, " W/m2/K")

x = x0 + 0.5*dx
h, rho, cp, mu, pr, re, nu, lbd = air_thermal_transfer_data(phd, pamb,tamb,vair, x)

print("")
print("Heat transfer factor = ", "%.1f"%h, " W/m2/K")
print("rho air = ", "%.3f"%rho, " kg/m3")
print("Cp air = ", "%.1f"%cp, " J/kg")
print("mu air = ", "%.1f"%(mu*1e6), " 10-6Pa.s")
print("Pr air = ", "%.4f"%pr)
print("Re air = ", "%.0f"%re)
print("Nu air = ", "%.0f"%nu)
print("Lambda air = ", "%.4f"%lbd)
