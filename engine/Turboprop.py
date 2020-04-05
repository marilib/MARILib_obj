# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from matplotlib import pyplot as plt

from engine.ExergeticEngine import Turboprop


# instantiate the 3-shafts turboprop object
TP = Turboprop()

# Set the flight conditions to 12000ft, ISA, Mn 0.316 (TAS=200kt) as static temperature, static pressure and Mach number
TP.set_flight(264.3756, 64440.86044, 0.316)

# Maximum temperature (T41), in degK
Ttm = 1526.5
# Mass flow in kg.s-1
w = 5.7897
# pressure ratios
lpr = 5.389
hpr = 2.69
# shaft power (watt)
shp = 1843.9 * 745.7

# set the efficiencies
TP.ex_loss = {"inlet": 0., "prop": 0.16, "PGB": 0.005369055}
tau_l, TP.ex_loss["LPC"] = TP.from_PR_to_tau_is(lpr, 0.78)
tau_h, TP.ex_loss["HPC"] = TP.from_PR_to_tau_is(hpr, 0.82)
TP.ex_loss["Burner"] = TP.from_PR_loss_to_Ex_loss(0.94)
aut_h, TP.ex_loss["HPT"] = TP.from_PR_to_tau_is(0.5253, 0.9)
aut_l, TP.ex_loss["LPT"] = TP.from_PR_to_tau_is(0.4671, 0.9)
aut_p, TP.ex_loss["PWT"] = TP.from_PR_to_tau_is(0.303, 0.7987)
TP.ex_loss["PE"] = TP.from_PR_loss_to_Ex_loss(0.98)

# initial cycle parameters: mass flow, specific shaft power, temperature ratios
x0 = np.array([w, shp/w, tau_l, tau_h])
# we may compute the cycle directly
# s is a dictionary with the stations state (Tt, Pt, Ht, Ex, etc...)
# p is a dictionary with the engine performances (net thrust, propulsive and thermal efficiencies, fuel flow)
# c is a dictionary with the component state (pressure ratio, shaft work)
# s, c, p = TP.cycle(w, shp/w, tau_l, tau_h, Ttmax=Ttm)
# or find the cycle parameters that exactly match the thrust (in Newton) and the pressure ratios
s, c, p = TP.design(13000., lpr, hpr, 1.1, Ttmax=Ttm, guess=x0)
# print a table with the state at each stations
TP.print_stations(s)
# print the global performances
TP.print_perfos(p)

# create a nice sankey diagram
fig = plt.figure(facecolor="w", tight_layout=True)
ax1 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
TP.draw_sankey(s, c, p, ax1)

# create a pie chart of all the losses
#ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
#TP.draw_pie(s, ax2)

# show the two graphs in a window
plt.show()

# recompute the design point in off-design mode
# You may either set the fuel flow or the T41 (Ttmax)
s, c, p = TP.off_design(wfe=p['wfe'])
guess = p["sol"]
# now for a cruise loop
# first we go up, step by step
Tm1 = [1.01, 1.02, 1.03, 1.04, 1.05]
# then down, step by step
Tm = [1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83, 0.82, 0.81,
      0.80, 0.79, 0.78, 0.77, 0.76, 0.75, 0.74, 0.73, 0.72, 0.71, 0.70, 0.69, 0.68, 0.67, 0.66, 0.65, 0.64, 0.63, 0.62,
      0.61]
# going up
for T in Tm1:
    s, c, p = TP.off_design(guess=guess, Ttmax=T * Ttm)
    if p['success']:
        guess = p['sol']
# going down, memorizing the thrust and sfc
Fn = []
sfc = []
Tms = []
for T in Tm:
    s, c, p = TP.off_design(guess=guess, Ttmax=T * Ttm)
    print("done")
    if p['success']:
        guess = p['sol']
        Fn.append(p['Fnet'])
        sfc.append(p['Pth']/p['Fnet'])
        Tms.append(T*Ttm)
        print(p['Fnet'])
# print the T41 that were computed
print(Tms)
# draw the curve
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Fn, sfc, '-ok')
ax.grid()
plt.show()


# now for a take-off point
TP.set_flight(288.15, 101325., 0.25)
TP.propeller_speed = 1.2
x0 = np.ones_like(TP.dp)
x0[0] = 1.34
x0[1] = 0.91
s, c, p = TP.off_design(Ttmax=Ttm, guess=x0)
if p['success']:
    TP.print_stations(s)
