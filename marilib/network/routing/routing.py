#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot, Weichang Lyu
"""

import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

from marilib.context import unit
from marilib.context.math import lin_interp_1d


def distance_to_center(w,n,x,h,m,y):
    """Compute the total distance between a focal point and the center of all cells

    :param w: Cell width
    :param n: Number of cells in width
    :param x: Relative position of the focal point in width
    :param h: Cell height
    :param m: Number of cells in height
    :param y: Relative position of the focal point in height
    :return: The total distance
    """
    dist = 0.
    for i in range(n):
        for j in range(m):
            dist += np.sqrt((i+0.5-x/n)*w**2*+(j+0.5-y/m)*h**2)
    return dist


# Cell size
w = 1000.
h = 1000.

# Area size (number of cell)
n = 1000
m = 1000

# Center position
x = 0.5
y = 0.5


print("Varying aspect ratio")
print("--------------------------------------")
print("x = ",x,"  y = ",y)
print("--------------------------------------")
dist = distance_to_center(w,n,x,h,m,y)
print("n = ",n,"  m = ",m,"  n.m = ",n*m)
print("dist = ",dist*1.e-12," Tm (1e12 m)")
print("--------------------------------------")
n = 500
m = 2000
dist = distance_to_center(w,n,x,h,m,y)
print("n = ",n,"  m = ",m,"  n.m = ",n*m)
print("dist = ",dist*1.e-12," Tm (1e12 m)")
print("--------------------------------------")
n = 250
m = 4000
dist = distance_to_center(w,n,x,h,m,y)
print("n = ",n,"  m = ",m,"  n.m = ",n*m)
print("dist = ",dist*1.e-12," Tm (1e12 m)")
print("--------------------------------------")
n = 125
m = 8000
dist = distance_to_center(w,n,x,h,m,y)
print("n = ",n,"  m = ",m,"  n.m = ",n*m)
print("dist = ",dist*1.e-12," Tm (1e12 m)")


print("")
print("Varying the position of the focal point")
print("--------------------------------------")
n = 1000
m = 1000
print("n = ",n,"  m = ",m,"  n.m = ",n*m)
print("--------------------------------------")
x = 0.25
y = 0.50
dist = distance_to_center(w,n,x,h,m,y)
print("x = ",x,"  y = ",y)
print("dist = ",dist*1.e-12," Tm (1e12 m)")
print("--------------------------------------")
x = 0.25
y = 0.25
dist = distance_to_center(w,n,x,h,m,y)
print("x = ",x,"  y = ",y)
print("dist = ",dist*1.e-12," Tm (1e12 m)")
print("--------------------------------------")
x = 0.
y = 0.
dist = distance_to_center(w,n,x,h,m,y)
print("x = ",x,"  y = ",y)
print("dist = ",dist*1.e-12," Tm (1e12 m)")


print("")
print("Varying both aspect ratio and focal point position")
print("--------------------------------------")
x = 0.25
y = 0.25
n = 250
m = 4000
dist = distance_to_center(w,n,x,h,m,y)
print("n = ",n,"  m = ",m,"  n.m = ",n*m)
print("x = ",x,"  y = ",y)
print("dist = ",dist*1.e-12," Tm (1e12 m)")
print("--------------------------------------")
x = 0.25
y = 0.75
n = 8000
m = 125
dist = distance_to_center(w,n,x,h,m,y)
print("n = ",n,"  m = ",m,"  n.m = ",n*m)
print("x = ",x,"  y = ",y)
print("dist = ",dist*1.e-12," Tm (1e12 m)")




def dist_to_fp(area,npt):
    """Compute the total distance between a focal point and the center of all cells

    :param S: Area of a convex domain
    :param np: Number of cells
    :return: The total distance to join a focal point inside the domain
    """
    a = np.sqrt(area)         # the side length of a square of equivalent area
    n = int(np.sqrt(npt))     # the number of cell in one direction
    if npt<n*(n+1): m = n
    else: m = n+1
    q = int(npt-n*m)    # the number of cell outside the square (or rectangle)
    dist = 0.
    for i in range(n):
        for j in range(m):
            dist += np.sqrt((i+0.5-x/n)*w**2*+(j+0.5-y/m)*h**2)
    for i in range(q):
        dist += np.sqrt((i+0.5-x/n)*w**2*+(m+0.5-y/m)*h**2)
    return dist



area = 1.e12   # Covered area

npt = 1e6     # Number of cells

dist = dist_to_fp(area,npt)

print("")
print("Generic version")
print("--------------------------------------")
print("area = ",area*1.e-6," km2")
print("npt = ",npt)
print("dist = ",dist*1.e-12," Tm (1e12 m)")
