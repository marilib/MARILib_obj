#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
         Aircraft & Systems, Air Transport Departement, ENAC
"""

from marilib.utils import earth, unit

import numpy as np
from scipy.optimize import fsolve

from marilib.aircraft.performance import Flight

r,gam,Cp,Cv = earth.gas_data()
g = earth.gravity()


disa = 0.

daltp = 1.
altp1 = 5000.
altp2 = altp1 + daltp

pamb1,tamb1,tstd1,dtodz1 = earth.atmosphere(altp1,disa)
pamb2,tamb2,tstd2,dtodz2 = earth.atmosphere(altp2,disa)


mach = 0.65

tas1 = mach*earth.sound_speed(tamb1)
tas2 = mach*earth.sound_speed(tamb2)

print((tas2-tas1)/daltp)
acc_fac1 = earth.climb_mode("mach",mach,dtodz1,tstd1,disa)
print((acc_fac1-1.)*g/tas1)


vcas = unit.mps_kt(250.)

mach1 = earth.mach_from_vcas(pamb1,vcas)
mach2 = earth.mach_from_vcas(pamb2,vcas)
tas1 = mach1*earth.sound_speed(tamb1)
tas2 = mach2*earth.sound_speed(tamb2)

print((tas2-tas1)/daltp)
acc_fac = earth.climb_mode("cas",mach1,dtodz1,tstd1,disa)
print((acc_fac-1.)*g/tas1)

