# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from matplotlib import pyplot as plt

from engine.ExergeticEngine import Turbofan


TF = Turbofan()
# Set the flight conditions to 35000ft, ISA, Mn 0.78 as static temperature, static pressure and Mach number
TF.set_flight(218.808, 23842.272, 0.78)
TF.ex_loss = {"inlet": 0., "LPC": 0.132764781, "HPC": 0.100735895, "Burner": 0.010989737,
              "HPT": 0.078125215, "LPT": 0.104386722, "Fan": 0.074168491, "SE": 0.0, "PE": 0.}
# Design for a given thrust (Newton), BPR, FPR, LPC PR, HPC PR, T41 (Kelvin)
Ttm = 1750.
s, c, p = TF.design(21000., 9., 1.66, 2.4, 14., Ttmax=Ttm, HPX=90*745.7, wBleed=0.5)
TF.print_stations(s)
TF.print_perfos(p)
TF.print_components(c)
TF.print_csv(s)

fig = plt.figure(facecolor="w", tight_layout=True)
ax1 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
TF.draw_sankey(s, c, p, ax1)
plt.show()
#ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
#TF.draw_pie(s, ax2)
#plt.show()


# recompute the design point in off-design mode
s, c, p = TF.off_design(wfe=p['wfe'], HPX=90*745.7)
#guess = p["sol"]
guess = TF.magic_guess()
TF.print_stations(s)
TF.print_perfos(p)

# now for a cruise loop
Tm = [1.1, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76,
      0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.6, 0.58, 0.56, 0.54, 0.52, 0.5]

Fn = []
sfc = []

for T in Tm:
    s, c, p = TF.off_design(guess=guess, Ttmax=T*Ttm, HPX=90*745.7)
    if p['success']:
#        guess = p['sol']
#        print(guess)
        guess = TF.magic_guess()
        Fn.append(p['Fnet'] / 10.)
        sfc.append(p['wfe'] * 36000. / p['Fnet'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Fn, sfc, '-ok')
ax.grid()
plt.show()


# test with reduced speed
TF.set_flight(218.808, 23842.272, 0.68)
#x0 = np.ones_like(TF.dp)
#x0 = p['sol']
x0 = TF.magic_guess()
s, c, p = TF.off_design(Ttmax=Ttm, guess=x0)
print("---------------------------------------------------------")
TF.print_stations(s)
TF.print_perfos(p)
print(p['wfe'] * 36000. / p['Fnet'])
print("---------------------------------------------------------")


# test with reduced speed
TF.set_flight(218.808, 23842.272, 0.58)
#x0 = np.ones_like(TF.dp)
#x0 = p['sol']
x0 = TF.magic_guess()
s, c, p = TF.off_design(Ttmax=Ttm, guess=x0)
print("---------------------------------------------------------")
TF.print_stations(s)
TF.print_perfos(p)
print(p['wfe'] * 36000. / p['Fnet'])
print("---------------------------------------------------------")


# now for a take-off point
TF.set_flight(288.15, 101325., 0.15)
#x0 = np.ones_like(TF.dp)
#x0[0] = x0[0] * 2.
#x0 = p['sol']
x0 = TF.magic_guess()
s, c, p = TF.off_design(Ttmax=Ttm, guess=x0)
TF.print_stations(s)
TF.print_perfos(p)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print(x0)


# other point
TF.set_flight(285.1782, 95951.78827609782, 0.38)
#x0 = np.ones_like(TF.dp)
#x0[0] = x0[0] * 2.
#x0 = p['sol']
x0 = TF.magic_guess()
s, c, p = TF.off_design(Ttmax=1200., guess=x0)
TF.print_stations(s)
TF.print_perfos(p)
print("+++++++++++++++++++++++++++++++++++++++++++++++++++")
print(x0)
