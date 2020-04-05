# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from matplotlib import pyplot as plt

from engine.ExergeticEngine import ElectricFan


TF = ElectricFan()

# Set the flight conditions to 35000ft, ISA, Mn 0.78 as static temperature, static pressure and Mach number
TF.set_flight(218.808, 23842.272, 0.78)

TF.ex_loss["inlet"] = TF.from_PR_loss_to_Ex_loss(0.997)
tau_f, TF.ex_loss["Fan"] = TF.from_PR_to_tau_pol(1.46, 0.95)
TF.ex_loss["PE"] = TF.from_PR_loss_to_Ex_loss(0.99)

s, c, p = TF.design(21000., 1.46, d_bli=21000./4)
TF.print_stations(s)
TF.print_perfos(p)
TF.print_perfos(s)

fig = plt.figure(facecolor="w", tight_layout=True)
ax1 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
TF.draw_sankey(s, c, p, ax1)
plt.show()

#ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])

#TF.draw_pie(s, ax2)
#plt.show()


# recompute the design point in off-design mode
s, c, p = TF.off_design(throttle=1., d_bli=21000./4)
guess = p["sol"]
# now for a cruise loop
Tm = [1.1, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76,
      0.74, 0.72, 0.70]

Fn = []
sfc = []

for T in Tm:
    s, c, p = TF.off_design(guess=guess, throttle=T, d_bli=21000./4)
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
x0 = np.array([2., 1.])  # 2x the mass flow, same temperature rise
s, c, p = TF.off_design(throttle=1., guess=x0)
TF.print_stations(s)
TF.print_perfos(p)
