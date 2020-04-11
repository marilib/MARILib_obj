# -*- coding: utf-8 -*-
"""

"""
import numpy as np
from engine.ExergeticEngine import Turbofan
# from scipy.optimize import root
from matplotlib import pyplot as plt


TF = Turbofan()
TF.cooling_flow = 0.1
# Set the flight conditions to 35000ft, ISA, Mn 0.78 as static temperature, static pressure and Mach number
TF.set_flight(218.808, 23842.3, 0.76)

TF.ex_loss = {"inlet": 0., "LPC": 0.132764781, "HPC": 0.100735895, "Burner": 0.010989737,
              "HPT": 0.08, "LPT": 0.11, "Fan": 0.074168491, "SE": 0.0, "PE": 0.}
TF.ex_loss["inlet"] = TF.from_PR_loss_to_Ex_loss(0.9985)
tau_f, TF.ex_loss["Fan"] = TF.from_PR_to_tau_pol(1.166, 0.93689)
tau_l, TF.ex_loss["LPC"] = TF.from_PR_to_tau_pol(3.12, 0.89)
tau_h, TF.ex_loss["HPC"] = TF.from_PR_to_tau_pol(14.22820513, 0.89)
TF.ex_loss["SE"] = TF.from_PR_loss_to_Ex_loss(0.992419452)

# Design for a given thrust (Newton), BPR, FPR, LPC PR, HPC PR, T41 (Kelvin)
Ttm = 1700.
# s, c, p = TF.cycle(15.05, 28.47, tau_f, tau_l, tau_h, Ttmax=Ttm, HPX=90*745.7, wBleed=0.50983516)
s, c, p = TF.design(20795., 15.05, 1.166, 3.12, 14.2282, Ttmax=Ttm, HPX=90*745.7, wBleed=0.50983516)
TF.print_stations(s)
TF.print_perfos(p)

# def toto(x):
#     tau_l, TF.ex_loss["LPC"] = TF.from_PR_to_tau_pol(3.12, x[0])
#     tau_h, TF.ex_loss["HPC"] = TF.from_PR_to_tau_pol(14.22820513, x[0])
#     TF.ex_loss["HPT"] = x[1]
#     TF.ex_loss["LPT"] = x[1]
#     s, c, p = TF.cycle(15.05, 28.47, tau_f, tau_l, tau_h, Ttmax=Ttm, HPX=90 * 745.7, wBleed=0.50983516)
#     return [s['5']['Tt']/616.47-1., s['5']['Pt']/49064.54-1]
#
# x0 = [0.9, 0.05]
# res = root(toto, x0)
# tau_l, TF.ex_loss["LPC"] = TF.from_PR_to_tau_pol(3.12, res.x[0])
# tau_h, TF.ex_loss["HPC"] = TF.from_PR_to_tau_pol(14.22820513, res.x[0])
# TF.ex_loss["HPT"] = res.x[1]
# TF.ex_loss["LPT"] = res.x[1]
# s, c, p = TF.cycle(15.05, 28.47, tau_f, tau_l, tau_h, Ttmax=Ttm, HPX=90 * 745.7, wBleed=0.50983516)
# TF.print_stations(s)
# TF.print_perfos(p)
# print(res.x)
# fig = plt.figure(facecolor="w", tight_layout=True)
# ax1 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
# TF.draw_sankey(s, c, p, ax1)
# plt.show()
# fig = plt.figure(facecolor="w", tight_layout=True)
# ax1 = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
# ax2 = fig.add_subplot(2, 1, 2, xticks=[], yticks=[])
# TF.draw_pie(s, ax1)
# plt.show()


# # recompute the design point in off-design mode
# print("\nConvergence on Net thrust:")
# TF.set_flight(285.1782, 95951.78827609782, 0.38)
# s, c, p = TF.off_design(Fnet=17786., HPX=0., wBleed=0.)
# print(p["Fnet"])
# print(s["4"]["Tt"])
# #s, c, p = TF.off_design(Fnet=17786., HPX=90*745.7, wBleed=0.50983516)
# TF.print_stations(s)
# TF.print_perfos(p)

# recompute the design point in off-design mode
print("\nConvergence on Ttmax:")
TF.set_flight(285.1782, 95952., 0.38)
s, c, p = TF.off_design(Ttmax=1530., HPX=0., wBleed=0.)
print(p["Fnet"])
print(s["4"]["Tt"])
TF.print_stations(s)
TF.print_perfos(p)

# recompute the design point in off-design mode
print("\nConvergence on N1:")
s, c, p = TF.off_design(N1=0.85, HPX=0., wBleed=0.)
print(p["Fnet"])
print(s["4"]["Tt"])
TF.print_stations(s)
TF.print_perfos(p)


# now for a cruise loop
Tm = [1.1, 1.08, 1.06, 1.04, 1.02, 1.00, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.86, 0.84, 0.82, 0.80, 0.78, 0.76,
      0.74, 0.72, 0.70, 0.68, 0.66, 0.64, 0.62, 0.6, 0.58, 0.56, 0.54, 0.52, 0.5, 0.45, 0.4, 0.35, 0.25, 0.20, 0.15, 0.10]
FnRef = p['Fnet']
Fn = []
sfc = []
guess = p['sol']
for T in Tm:
    s1, c, p = TF.off_design(guess=guess, Fnet=T*FnRef, HPX=90*745.7)
    if p['success']:
        guess = p['sol']
        Fn.append(p['Fnet'] / 10.)
        sfc.append(p['wfe'] * 36000. / p['Fnet'])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Fn, sfc, '-ok')
ax.grid()
plt.show()
print(Fn)
print(sfc)

# now for a take-off point
TF.set_flight(288.15, 101325., 0.25)
s, c0, p = TF.off_design(Ttmax=Ttm)
if s is not None:
    TF.print_stations(s)
    TF.print_perfos(p)

# now for a maxi cont point
TF.set_flight(228.15, 81325., 0.40)
s, c0, p = TF.off_design(Ttmax=Ttm*0.95)
if s is not None:
    TF.print_stations(s)
    TF.print_perfos(p)
