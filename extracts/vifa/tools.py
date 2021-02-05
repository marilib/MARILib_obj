#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

import numpy
import math


def rad_deg(deg): return deg*math.pi/180   # Translate degrees into radians

def deg_rad(rad): return rad*180/math.pi   # Translate radians into degrees

#===================================================================================================================
# Vector product of U by V
def pv(U, V):
    res = numpy.array([U[1]*V[2]-U[2]*V[1] ,
                       U[2]*V[0]-U[0]*V[2] ,
                       U[0]*V[1]-U[1]*V[0]])
    return res

#===================================================================================================================
# Scalar product of U by V
def ps(U, V):
    res = U[0]*V[0] + U[1]*V[1] + U[2]*V[2]
    return res

#===================================================================================================================
# Return normalised vector V
def renorm(V):
    n = math.sqrt(V[0]**2 + V[1]**2 + V[2]**2)
    if (n>0.):
        N = V/n
    else:
        N = numpy.array([0. ,0. , 0.])
    return N

#===================================================================================================================
# Calculates an angle from its sine and cosine coordinates
# if typ=1 then -180<a_<180, if typ=2 then 0<a_<360
def angle(sin_a, cos_a, typ):
    pi = numpy.pi
    epsilon = 1.e-15
    if (abs(cos_a-1) < epsilon):
        a = 0.
    elif (abs(cos_a+1) < epsilon):
        a = pi
    elif (abs(sin_a-1) < epsilon):
        a = 0.5*pi
    elif (abs(sin_a+1) < epsilon):
        a = -0.5*pi
    else:
        a = numpy.sign(sin_a)*abs(math.acos(cos_a))
    if (typ==2):
        a = a + (1-numpy.sign(a))*pi
    if (a > 2*pi):
        a = a - 2.*pi
    return a


#===================================================================================================================
def rotate(pivot,axis,angle,p0):

    vec = p0-pivot
    r0 = vec - ps(vec,axis)*axis
    Q = numpy.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    RmI = math.sin(angle)*Q + (1-math.cos(angle))*numpy.matmul(Q,Q)
    p1 = p0 + numpy.matmul(RmI,r0)

    return p1
