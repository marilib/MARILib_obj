#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np


#-----------------------------------------------------------------------------------------------------------------------
#
#  Some math functions
#
#-----------------------------------------------------------------------------------------------------------------------

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


