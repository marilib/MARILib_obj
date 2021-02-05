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

xcg = 0.20
mass = 70000

vair = 150

psi = 0
theta = 0
phi = 0

alpha = rad_deg(10)
betha = rad_deg(0)

dl = rad_deg(15)
dm = rad_deg(-15)
dn = rad_deg(0)

trim = rad_deg(-10)

a0 = rad_deg(-5)





geometry(a0,trim)

contour(a0,trim,dl,dm,dn)

force(mass,xcg,vair,psi,theta,phi,alpha,betha,dl,dm,dn)


#
# print(geo.Ftotal)
# print(geo.MtotalXg)
# print("")
# print(geo.RafVec)
#



# Drawing_ box
#-----------------------------------------------------------------------------------------------------------
xTop,yTop,xSide,ySide,xBack,yBack = init_plot_view("Demo","Aerodynamic forces")

kScale = 1.50e-5

plot_top_view(plt,xTop,yTop,kScale)
plot_side_view(plt,xSide,ySide,dl,dm,kScale)
plot_back_view(plt,xBack,yBack,kScale)


plt.show()


