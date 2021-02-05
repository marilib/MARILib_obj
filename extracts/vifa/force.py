#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

import numpy
import math

import data as geo
from tools import rad_deg, angle, renorm, ps, pv


#===================================================================================================================
def force(mass,xcg,vair,psi,theta,phi,alpha,betha,dl,dm,dn):

    g = 9.80665
    rho = 1.225

    Kdl = 0.20
    Kdm = 0.90
    Kdn = 0.90

    uX = numpy.array([1,0,0])
    uY = numpy.array([0,1,0])
    uZ = numpy.array([0,0,1])

    # Rotations
    # -----------------------------------------------------------------------------------------------------------
    Rpsi = numpy.array([[ math.cos(psi) , -math.sin(psi) , 0. ],
                        [ math.sin(psi) ,  math.cos(psi) , 0. ],
                        [ 0.            ,  0.            , 1. ]])

    Rtheta = numpy.array([[ math.cos(theta) , 0. , math.sin(theta) ],
                          [ 0.              , 1. , 0.              ],
                          [-math.sin(theta) , 0. , math.cos(theta) ]])

    Rphi = numpy.array([[ 1. , 0.            ,  0.            ],
                        [ 0. , math.cos(phi) , -math.sin(phi) ],
                        [ 0. , math.sin(phi) ,  math.cos(phi) ]])

    Ratt = numpy.matmul(Rpsi,numpy.matmul(Rtheta,Rphi))

    tRatt = numpy.transpose(Ratt)

    mg = g*mass*numpy.matmul(tRatt,[0.,0.,1.])        # Wairplane weight in airplane frame

    mgApp = -(geo.WingXmac+xcg*geo.WingCmac)*uX           # Weight application point

    # -----------------------------------------------------------------------------------------------------------
    Rbetha = numpy.array([[ math.cos(betha) , math.sin(betha) , 0. ],
                          [-math.sin(betha) , math.cos(betha) , 0. ],
                          [ 0.              , 0.              , 1. ]])

    Ralpha = numpy.array([[ math.cos(alpha) , 0. , math.sin(alpha) ],
                          [ 0.              , 1. , 0.              ],
                          [-math.sin(alpha) , 0. , math.cos(alpha) ]])

    Rba = numpy.matmul(Rbetha,Ralpha)

    Rab = numpy.transpose(Rba)

    VairAf = vair*uX    # Airspeed in Aerodynamic frame

    Vair = numpy.matmul(Rab,VairAf)      # Airspeed in aircraft frame

    Vdir = renorm(Vair)

    # Aerodynamic
    # -------------------------------------------------------------------------------------------------------------------------
    tRalpha = numpy.transpose(Ralpha)

    SpanDir = Rab[:,1]      # Span projection direction

    rWspan = ps(geo.WingRaxe,SpanDir)      # Right wing projected span
    lWspan = ps(geo.WingLaxe,-SpanDir)      # Left wing projected span

    rWar = (2*rWspan)**2/geo.WingArea      # Virtual right wing aspect ratio
    lWar = (2*lWspan)**2/geo.WingArea      # Virtual left wing aspect ratio

    CzaWr = 0.5*(numpy.pi*rWar)/(1+math.sqrt(1+(rWar/2)**2))      # Right wing lift gradiant
    CzaWl = 0.5*(numpy.pi*lWar)/(1+math.sqrt(1+(lWar/2)**2))      # Left wing lift gradiant

    rWortho = renorm(pv(geo.WingCdir,geo.WingRdir))
    lWortho = renorm(pv(-geo.WingCdir,geo.WingLdir))

    rWalphaDir = -renorm(pv(SpanDir,rWortho))
    lWalphaDir = -renorm(pv(SpanDir,lWortho))

    rWliftDir = pv(SpanDir,rWalphaDir)      # Right wing lift direction
    lWliftDir = pv(SpanDir,lWalphaDir)      # Left wing lift direction

    rWcos_a = ps(rWalphaDir,Vdir)
    rWsin_a = ps(pv(Vdir,rWalphaDir),SpanDir)
    rWalpha = angle(rWsin_a, rWcos_a, 1)      # Right wing angle of attack

    lWcos_a = ps(lWalphaDir,Vdir)
    lWsin_a = ps(pv(Vdir,lWalphaDir),SpanDir)
    lWalpha = angle(lWsin_a, lWcos_a, 1)      # Left wing angle of attack

    rAalpha = math.atan(geo.AilCmed*math.sin(dl)/((geo.AilWingCmed-geo.AilCmed)+geo.AilCmed*math.cos(dl)))
    lAalpha = math.atan(geo.AilCmed*math.sin(-dl)/((geo.AilWingCmed-geo.AilCmed)+geo.AilCmed*math.cos(-dl)))

    # -------------------------------------------------------------------------------------------------------------------------
    rHspan = ps(geo.HtpRaxe,SpanDir)      # Right HTP projected span
    lHspan = ps(geo.HtpLaxe,-SpanDir)      # Left HTP projected span

    rHar = (2*rHspan)**2/geo.HtpArea      # Virtual right HTP aspect ratio
    lHar = (2*lHspan)**2/geo.HtpArea      # Virtual left HTP aspect ratio

    CzaHr = 0.5*(math.pi*rHar)/(1+math.sqrt(1+(rHar/2)**2))
    CzaHl = 0.5*(math.pi*lHar)/(1+math.sqrt(1+(lHar/2)**2))

    rHortho = renorm(pv(geo.HtpCdir,geo.HtpRdir))
    lHortho = renorm(pv(-geo.HtpCdir,geo.HtpLdir))

    rHalphaDir = -renorm(pv(SpanDir,rHortho))
    lHalphaDir = -renorm(pv(SpanDir,lHortho))

    rHliftDir = pv(SpanDir,rHalphaDir)
    lHliftDir = pv(SpanDir,lHalphaDir)

    rHcos_a = ps(rHalphaDir,Vdir)
    rHsin_a = ps(pv(Vdir,rHalphaDir),SpanDir)
    rHalpha = angle(rHsin_a, rHcos_a, 1)      # Right HTP angle of attack

    lHcos_a = ps(lHalphaDir,Vdir)
    lHsin_a = ps(pv(Vdir,lHalphaDir),SpanDir)
    lHalpha = angle(lHsin_a, lHcos_a, 1)      # Left HTP angle of attack

    rEalpha = math.atan(geo.ElevCaxe*math.sin(dm)/((geo.ElevHtpCaxe-geo.ElevCaxe)+geo.ElevCaxe*math.cos(dm)))

    # -------------------------------------------------------------------------------------------------------------------------
    vSpan = abs(ps(geo.VtpAxe,tRalpha[:,2]))

    Var = vSpan**2/geo.VtpArea      # Virtual VTP aspect ratio

    CzaV = (math.pi*Var)/(1+math.sqrt(1+(Var/2)**2))      # VTP lift gradiant

    VliftDir = renorm(pv(Vdir,uZ))      # VTP lift direction
    # VTP lift direction

    Valpha = betha      # VTP angle of attack

    Ralpha = math.atan(geo.RudCroot*math.sin(dn)/(geo.RudVtpCroot-(geo.RudCroot+geo.RudCroot*math.cos(dn))))

    # Forces
    # -------------------------------------------------------------------------------------------------------------------------
    Qdyn = 0.5*rho*vair**2

    geo.WingRapp = numpy.array([-geo.WingXmac-0.25*geo.WingCmac , geo.WingYmac , -geo.WingZmac])     # Right wing lift application point
    geo.rWafVec = Qdyn*geo.WingArea*CzaWr*rWalpha*rWliftDir

    geo.AilRapp = numpy.array([-geo.AilWingXmed-0.25*geo.AilWingCmed , geo.AilYmed , -geo.AilZmed])      # Right aileron lift application point
    geo.rAafVec = Qdyn*geo.WingArea*Kdl*CzaWr*rAalpha*rWliftDir

    geo.WingLapp = numpy.array([-geo.WingXmac-0.25*geo.WingCmac , -geo.WingYmac , -geo.WingZmac])      # Left wing lift application point
    geo.lWafVec = Qdyn*geo.WingArea*CzaWl*lWalpha*lWliftDir

    geo.AilLapp = numpy.array([-geo.AilWingXmed-0.25*geo.AilWingCmed , -geo.AilYmed , -geo.AilZmed])      # Left aileron lift application point
    geo.lAafVec = Qdyn*geo.WingArea*Kdl*CzaWl*lAalpha*lWliftDir


    geo.HtpRapp = numpy.array([-geo.HtpXmac-0.25*geo.HtpCmac , geo.HtpYmac , -geo.HtpZmac])      # Right HTP lift application point
    geo.rHafVec = Qdyn*geo.HtpArea*CzaHr*rHalpha*rHliftDir
    geo.rEafVec = Qdyn*geo.HtpArea*Kdm*CzaHr*rEalpha*rHliftDir

    geo.HtpLapp = numpy.array([-geo.HtpXmac-0.25*geo.HtpCmac , -geo.HtpYmac , -geo.HtpZmac])      # Left HTP lift application point
    geo.lHafVec = Qdyn*geo.HtpArea*CzaHl*lHalpha*lHliftDir
    geo.lEafVec = Qdyn*geo.HtpArea*Kdm*CzaHl*rEalpha*lHliftDir

    geo.FusApp = numpy.array([-geo.FusLength*0.70 , 0 , -geo.FusHeight*1.20])
    geo.VafVec = ps(Qdyn*geo.VtpArea*CzaV*Valpha*VliftDir, uY)*uY

    geo.VtpApp = numpy.array([-geo.VtpXmac-0.25*geo.VtpCmac , 0 , -geo.VtpZmac])
    geo.RafVec = ps(-Qdyn*geo.VtpArea*Kdn*CzaV*Ralpha*VliftDir, uY)*uY

    # Moments
    #--------------------------------------------------------------------------------------------------------------

    Ftotal =   mg \
             + geo.rWafVec + geo.lWafVec \
             + geo.rAafVec + geo.lAafVec \
             + geo.rHafVec+geo.rEafVec + geo.lHafVec+geo.lEafVec \
             + geo.VafVec + geo.RafVec

    Mtotal =   pv(mgApp,mg) \
             + pv(geo.WingRapp,geo.rWafVec) + pv(geo.WingLapp,geo.lWafVec) \
             + pv(geo.AilRapp,geo.rAafVec) + pv(geo.AilLapp,geo.lAafVec) \
             + pv(geo.HtpRapp,(geo.rHafVec+geo.rEafVec)) + pv(geo.HtpLapp,(geo.lHafVec+geo.lEafVec)) \
             + pv(geo.FusApp,geo.VafVec) + pv(geo.VtpApp,geo.RafVec)

    geo.Ftotal = Ftotal

    geo.MtotalXg = Mtotal + pv(mgApp,Ftotal)

    geo.MtotalAnchor = numpy.array([1.35*geo.FusLength , 0.95*geo.WingSpan , 0.50*geo.FusHeight])


    return

