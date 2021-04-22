#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: DRUOT Thierry
"""

import numpy as np

import unit


#-----------------------------------------------------------------------------------------------------------------------
# Atmosphere
#-----------------------------------------------------------------------------------------------------------------------

def atmosphere(altp, disa=0., full_output=False):
    """
    Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
    """
    g = 9.80665
    r = 287.053
    gam = 1.4

    Z = np.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])
    P = np.array([101325., 0., 0., 0., 0., 0.])
    T = np.array([288.15, 0., 0., 0., 0., 0.])

    if (Z[-1]<altp):
        raise Exception("atmosphere, altitude cannot exceed 50km")

    j = 0
    while (Z[1+j]<=altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<np.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
        else:
            P[j+1] = P[j]*np.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0.<np.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
    else:
        pamb = P[j]*np.exp(-(g/r)*((altp-Z[j])/T[j]))
    tstd = T[j] + dtodz[j]*(altp-Z[j])
    tamb = tstd + disa
    if full_output:
        return pamb,tamb,tstd,dtodz[j]
    else:
        return pamb,tamb

def sound_speed(tamb):
    """Sound speed for ideal gas
    """
    r = 287.053
    gam = 1.4
    vsnd = np.sqrt( gam * r * tamb )
    return vsnd

def air_density(pamb,tamb):
    """Ideal gas density
    """
    r = 287.053
    rho0 = 1.225
    rho = pamb / ( r * tamb )
    sig = rho / rho0
    return rho, sig

def gas_viscosity(tamb, gas="air"):
    mu0,T0,S = [1.715e-5, 273.15, 110.4]
    mu = (mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
    return mu

def reynolds_number(pamb,tamb,mach):
    """Reynolds number based on Sutherland viscosity model
    """
    vsnd = sound_speed(tamb)
    rho,sig = air_density(pamb,tamb)
    mu = gas_viscosity(tamb)
    re = rho*vsnd*mach/mu
    return re

#-----------------------------------------------------------------------------------------------------------------------
# Maths
#-----------------------------------------------------------------------------------------------------------------------

def lin_interp_1d(x,X,Y):
    """linear interpolation without any control

    :param x: current position
    :param X: array of the abscissa of the known points
    :param Y: array of the known values at given abscissa
    :return: y the interpolated value of Y at x

    """
    n = np.size(X)
    for j in range(1,n):
        if x<X[j] :
            y = Y[j-1]+(Y[j]-Y[j-1])*(x-X[j-1])/(X[j]-X[j-1])
            return y
    y = Y[n-2]+(Y[n-1]-Y[n-2])*(x-X[n-2])/(X[n-1]-X[n-2])
    return y


def vander3(X):
    """Return the vandermonde matrix of a dim 3 array A = [X^2, X, 1]
    """
    V = np.array([[X[0]**2, X[0], 1.],
                  [X[1]**2, X[1], 1.],
                  [X[2]**2, X[2], 1.]])
    return V


def trinome(A,Y):
    """calculate trinome coefficients from 3 given points
    A = [X2, X, 1] (Vandermonde matrix)
    """
    X = np.array([A[0][1], A[1][1], A[2][1]])
    X2 = np.array([A[0][0], A[1][0], A[2][0]])

    det = X2[0]*(X[1]-X[2])-X2[1]*(X[0]-X[2])+X2[2]*(X[0]-X[1])

    adet = Y[0]*(X[1]-X[2])-Y[1]*(X[0]-X[2])+Y[2]*(X[0]-X[1])

    bdet = X2[0]*(Y[1]-Y[2])-X2[1]*(Y[0]-Y[2])+X2[2]*(Y[0]-Y[1])

    cdet =  X2[0]*(X[1]*Y[2]-X[2]*Y[1])-X2[1]*(X[0]*Y[2]-X[2]*Y[0]) \
          + X2[2]*(X[0]*Y[1]-X[1]*Y[0])

    if det!=0:
        C = np.array([adet/det, bdet/det, cdet/det])
    elif X[0]!=X[2]:
        C = np.array([0., Y[0]-Y[2], Y[2]*X[0]-Y[0]*X[2]/(X[0]-X[2])])
    else:
        C = np.array([0., 0., (Y[0]+Y[1]+Y[2])/3.])

    return C


def maximize_1d(xini,dx,*fct):
    """Optimize 1 single variable, no constraint.

    :param xini: initial value of the variable.
    :param dx: fixed search step.
    :param fct: function with the signature : ['function_name',a1,a2,a3,...,an] and function_name(x,a1,a2,a3,...,an).

    """
    n = len(fct[0])

    X0 = xini
    Y0 = fct[0][0](X0,*fct[0][1:n])

    X1 = X0+dx
    Y1 = fct[0][0](X1,*fct[0][1:n])

    if Y0>Y1:
        dx = -dx
        X0,X1 = X1,X0

    X2 = X1+dx
    Y2 = fct[0][0](X2,*fct[0][1:n])

    while Y1<Y2:
        X0 = X1
        X1 = X2
        X2 = X2+dx
        Y0 = Y1
        Y1 = Y2
        Y2 = fct[0][0](X2,*fct[0][1:n])

    X = np.array([X0,X1,X2])
    Y = np.array([Y0,Y1,Y2])

    A = vander3(X)     # [X**2, X, numpy.ones(3)]
    C = trinome(A,Y)

    xres = -C[1]/(2.*C[0])
    yres = fct[0][0](xres,*fct[0][1:n])

    rc = 1

    return (xres,yres,rc)

