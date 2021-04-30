#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

import numpy

import data as geo

from tools import rad_deg
from geometry import geometry
from contour import contour
from force import force
from drawing import init_plot_view, \
                    plot_top_view, \
                    plot_side_view, \
                    plot_back_view

import matplotlib.pyplot as plt

g = 9.80665

# xcg = 0.2555
# xcg = 0.3
# mass = 70000
#
# vair = 150
#
# psi = rad_deg(0)
# theta = rad_deg(0)
# phi = rad_deg(0)
#
# alpha = rad_deg(0)
# betha = rad_deg(0)
#
# dl = rad_deg(0)
# dm = rad_deg(0)
# dn = rad_deg(0)
# dx = 0
#
# trim = rad_deg(0)
#
# a0 = rad_deg(3.031)
# f0 = mass*g/5
#
# p = rad_deg(0)
# q = rad_deg(0)
# r = rad_deg(0)


a0 = 0.0
f0 = 137293.1
mass = 80000.0
xcg = 0.2
vair = 150.0
psi = 0.0
theta = 0.0
phi = 0.0
alpha = 0.0
betha = 0.0
trim = 0.0
dl = 0.0
dm = 0.0
dn = 0.0
dx = 0
p = 0.0
q = 0.0
r = 0.0


geometry()

contour(a0,trim,dl,dm,dn)

force(a0,f0,mass,xcg,vair,psi,theta,phi,alpha,betha,trim,dl,dm,dn,dx,p,q,r)



print(geo.Ftotal)
print(geo.MtotalXg)



# Drawing_ box
#-----------------------------------------------------------------------------------------------------------
xTop,yTop,xSide,ySide,xBack,yBack = init_plot_view("Demo","Aerodynamic forces")

kScale = 1.50e-5

plot_top_view(plt,xTop,yTop,kScale)
plot_side_view(plt,xSide,ySide,dl,dm,kScale)
plot_back_view(plt,xBack,yBack,kScale)


plt.show()


