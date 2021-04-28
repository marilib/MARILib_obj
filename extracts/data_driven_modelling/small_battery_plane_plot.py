#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
import unit

from small_battery_plane import SmallPlane

import matplotlib.pyplot as plt
from matplotlib import rc
font = {'size':12}
rc('font',**font)






#-------------------------------------------------------------------------------------------------------------------
# MAP display : minimal working example

sp = SmallPlane(alt=unit.m_ft(3000),
                tas=unit.mps_kmph(130),
                mode="electric")

sp.battery_enrg_density = unit.J_Wh(200)


distances = np.linspace(50e3, 1000e3, 30)
npaxs = np.arange(1, 19)
X, Y = np.meshgrid(distances, npaxs)

pkm = []
for x,y in zip(X.flatten(),Y.flatten()):
    sp.distance = x
    sp.n_pax = y
    sp.design_solver()
    pkm.append(sp.design["pk_o_m"]/sp.design["pk_o_m_min"])

# convert to numpy array with good shape
pkm = np.array(pkm)
pkm = pkm.reshape(np.shape(X))

print("")
# Plot contour
cs = plt.contourf(X / 1000, Y, pkm, levels=20)

c2c = plt.contour(X / 1000, Y, Y/X*1e3, levels=[0.0146], colors =['lightgrey'], linewidths=2)
c2h = plt.contourf(X / 1000, Y, Y/X*1e3, levels=[0.0146,1], linewidths=2, colors='none', hatches=['\\'])
for c in c2h.collections:
    c.set_edgecolor('lightgrey')

# # plt.clabel(c1, inline=True, fmt="%d",fontsize=15)

c1c = plt.contour(X / 1000, Y, pkm, levels=[1], colors =['yellow'], linewidths=2)
c1h = plt.contourf(X / 1000, Y, pkm, levels=[0,1], colors='none', linewidths=2, hatches=['\\'])
for c in c1h.collections:
    c.set_edgecolor('yellow')

# plt.plot(X / 1000, Y, '+k')
# plt.plot([0, distances[-1]*1e-3], [0, (19./1300.)*distances[-1]*1e-3], linestyle='solid', color='blue')

plt.colorbar(cs, label=r"P.K/M")
plt.grid(True)

plt.suptitle("PK/M Field")
plt.xlabel("Distance (km)")
plt.ylabel("N passenger")

plt.show()





