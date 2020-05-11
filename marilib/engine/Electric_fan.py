# -*- coding: utf-8 -*-
"""

"""
import math
import numpy as np
from marilib.engine.ExergeticEngine import ElectricFan
from matplotlib import pyplot as plt


TF = ElectricFan()

# Set the flight conditions to 35000ft, ISA, Mn 0.78 as static temperature, static pressure and Mach number
TF.set_flight(218.808, 23842.272, 0.78)

# The engine has an inlet "inlet", a fan "Fan", and an exhaust "PE"
# set the efficiency of each of each component
# Note: Fan efficiency has to exist, but its value will be overwritten by some semi-empirical estimation
TF.ex_loss["inlet"] = TF.from_PR_loss_to_Ex_loss(0.997)
tau_f, TF.ex_loss["Fan"] = TF.from_PR_to_tau_pol(1.32, 0.94)
TF.ex_loss["PE"] = TF.from_PR_loss_to_Ex_loss(0.985)

# Design and size the engine, for 21000. N thrust, with fan pressure ratio 1.46
s, c, p = TF.design(21000., 1.46, d_bli=0.)
# Print the results
TF.print_stations(s)
TF.print_perfos(p)
TF.print_perfos(s)
# Station nomenclature:
# 0: free stream
# 1: inlet face
# 2: fan face
# 3: Fan exit
# 9: Exhaust, fully expended

# In each station, the following items are defined:
# Ht: specific total enthalpy
# Ex: specific total exergy
# Pt: total pressure
# Tt: total temperature
# w: mass flow
# Wr: corrected mass flow

# Get the fan size: Mach 0.6 is typical for the fan face
V1, A1, Ps1, Ts1 = TF.get_statics(s["2"]['Ht'], s["2"]['Ex'], 0.6)

fan_area = A1 * s["2"]["w"]  # in m2
# get the fan diameter, using a typical value for the hub to tip ratio: 0.28
fan_diameter = math.sqrt(4. * fan_area / np.pi / (1 - 0.28**2))  # in m
exhaust_area = s['9']['A']  # in m2
print("Fan area: {:5.3f} m2\nFan diameter: {:5.3f} m = {:5.2f} in\nExhaust area: {:5.3f} m2\n".
      format(fan_area, fan_diameter, fan_diameter/0.0254, exhaust_area))

# draw a sankey diagram of the powers flowing in the engine
fig = plt.figure(facecolor="w", tight_layout=True)
ax1 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
TF.draw_sankey(s, c, p, ax1)
plt.show()

# ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])

# TF.draw_pie(s, ax2)
# plt.show()


# recompute the design point in off-design mode
# throttle is the rotation speed, as a ratio from design point.
s0, c, p = TF.off_design(throttle=1., d_bli=0.)
guess = p["sol"]
# now for a cruise loop
Tm = [1.1, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76,
      0.74, 0.72, 0.70]

Fn = []
sfc = []

for T in Tm:
    s1, c, p = TF.off_design(guess=guess, throttle=T, d_bli=21000./4)
    if p['success']:
        guess = p['sol']
        Fn.append(p['Fnet'])
        sfc.append(p['Pth']/p['Fnet'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Fn, sfc, '-ok')
ax.grid()
plt.show()

# now for a take-off point
TF.set_flight(288.15, 101325., 0.25)
s, c1, p = TF.off_design(throttle=1., d_bli=0.)
TF.print_stations(s)
TF.print_perfos(p)
