#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

import numpy
import math

import data as geo
from tools import rad_deg, angle, norm, renorm, rotate, ps, pv, skew_sym, det, inv


#===================================================================================================================
def force(a0,f0,mass,xcg,vair,psi,theta,phi,alpha,betha,trim,dl,dm,dn,dx,p,q,r):

    g = 9.80665
    rho = 1.225

    Kdl = 0.20
    Kdm = 0.80
    Kdn = 0.90

    origin = numpy.array([0,0,0])
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

    geo.mg = g*mass*numpy.matmul(tRatt,[0.,0.,1.])        # Wairplane weight in airplane frame

    geo.mgApp = numpy.array([-geo.WingXmac-xcg*geo.WingCmac , 0 , -geo.WingZmac])     # Weight application point

    # -----------------------------------------------------------------------------------------------------------
    Rbetha = numpy.array([[ math.cos(betha) , math.sin(betha) , 0. ],
                          [-math.sin(betha) , math.cos(betha) , 0. ],
                          [ 0.              , 0.              , 1. ]])

    Ralpha = numpy.array([[ math.cos(alpha) , 0. , math.sin(alpha) ],
                          [ 0.              , 1. , 0.              ],
                          [-math.sin(alpha) , 0. , math.cos(alpha) ]])
    tRalpha = numpy.transpose(Ralpha)

    Rba = numpy.matmul(Rbetha,Ralpha)

    Rab = numpy.transpose(Rba)

    VairAf = vair*uX    # Airspeed in Aerodynamic frame

    Vair = numpy.matmul(Rab,VairAf)      # Airspeed in aircraft frame

    Vdir = renorm(Vair)

    # Aerodynamic
    # -------------------------------------------------------------------------------------------------------------------------
    WingAxe = rotate(origin, uY, a0, geo.WingAxe)

    WingRaxe = rotate(origin, uY, a0, geo.WingRaxe)
    WingRnorm = renorm(pv(WingAxe,WingRaxe))
    rWspan = norm(WingRaxe - ps(WingRaxe,Vdir)*Vdir)      # Right wing projected span

    WingLaxe = rotate(origin, uY, a0, geo.WingLaxe)
    WingLnorm = renorm(pv(WingAxe,-WingLaxe))
    lWspan = norm(WingLaxe - ps(WingLaxe,Vdir)*Vdir)      # Left wing projected span

    rWar = (2*rWspan)**2/geo.WingArea      # Virtual right wing aspect ratio
    lWar = (2*lWspan)**2/geo.WingArea      # Virtual left wing aspect ratio

    CzaWr = 0.5*(numpy.pi*rWar)/(1+math.sqrt(1+(rWar/2)**2))      # Right wing lift gradiant
    CzaWl = 0.5*(numpy.pi*lWar)/(1+math.sqrt(1+(lWar/2)**2))      # Left wing lift gradiant

    rWalpha = numpy.arcsin(ps(-Vdir,WingRnorm))      # Right wing angle of attack
    lWalpha = numpy.arcsin(ps(-Vdir,WingLnorm))      # Left wing angle of attack

    rAalpha = math.atan(geo.AilCmed*math.sin(dl)/((geo.AilWingCmed-geo.AilCmed)+geo.AilCmed*math.cos(dl)))
    lAalpha = math.atan(geo.AilCmed*math.sin(-dl)/((geo.AilWingCmed-geo.AilCmed)+geo.AilCmed*math.cos(-dl)))

    rWliftDir = renorm(WingRnorm - ps(WingRnorm,Vdir)*Vdir)
    lWliftDir = renorm(WingLnorm - ps(WingLnorm,Vdir)*Vdir)

    # -------------------------------------------------------------------------------------------------------------------------
    HtpAxe = rotate(origin, uY, trim, geo.HtpAxe)

    HtpRaxe = rotate(origin, uY, trim, geo.HtpRaxe)
    HtpRnorm = renorm(pv(HtpAxe,HtpRaxe))
    rHspan = norm(HtpRaxe - ps(HtpRaxe,Vdir)*Vdir)      # Right Htp projected span

    HtpLaxe = rotate(origin, uY, trim, geo.HtpLaxe)
    HtpLnorm = renorm(pv(HtpAxe,-HtpLaxe))
    lHspan = norm(HtpLaxe - ps(HtpLaxe,Vdir)*Vdir)      # Left Htp projected span

    rHar = (2*rHspan)**2/geo.HtpArea      # Virtual right Htp aspect ratio
    lHar = (2*lHspan)**2/geo.HtpArea      # Virtual left Htp aspect ratio

    CzaHr = 0.5*(numpy.pi*rHar)/(1+math.sqrt(1+(rHar/2)**2))      # Right Htp lift gradiant
    CzaHl = 0.5*(numpy.pi*lHar)/(1+math.sqrt(1+(lHar/2)**2))      # Left Htp lift gradiant

    rHalpha = numpy.arcsin(ps(-Vdir,HtpRnorm))      # Right Htp angle of attack
    lHalpha = numpy.arcsin(ps(-Vdir,HtpLnorm))      # Left Htp angle of attack

    rEalpha = math.atan(geo.ElevCaxe*math.sin(dm)/((geo.ElevHtpCaxe-geo.ElevCaxe)+geo.ElevCaxe*math.cos(dm)))

    rHliftDir = renorm(HtpRnorm - ps(HtpRnorm,Vdir)*Vdir)
    lHliftDir = renorm(HtpLnorm - ps(HtpLnorm,Vdir)*Vdir)

    # -------------------------------------------------------------------------------------------------------------------------
    vSpan = abs(ps(geo.VtpAxe,tRalpha[:,2]))

    Var = vSpan**2/geo.VtpArea      # Virtual VTP aspect ratio

    CzaV = (math.pi*Var)/(1+math.sqrt(1+(Var/2)**2))      # VTP lift gradiant

    VliftDir = renorm(pv(Vdir,uZ))      # VTP lift direction

    Valpha = betha      # VTP angle of attack

    Ralpha = math.atan(geo.RudCroot*math.sin(dn)/(geo.RudVtpCroot-(geo.RudCroot+geo.RudCroot*math.cos(dn))))

    # Forces
    # -------------------------------------------------------------------------------------------------------------------------
    Qdyn = 0.5*rho*vair**2
    WingRotP = numpy.array([-geo.WingXaxe-0.5*geo.WingCaxe, 0., -geo.WingZaxe])
    WingRotAxe = uY

    WingRapp = numpy.array([-geo.WingXmac-0.25*geo.WingCmac , geo.WingYmac , -geo.WingZmac])     # Right wing lift application point
    geo.WingRapp = rotate(WingRotP, WingRotAxe, a0, WingRapp)
    geo.rWafVec = Qdyn*geo.WingArea*CzaWr*rWalpha*rWliftDir

    AilRapp = numpy.array([-geo.AilWingXmed-0.25*geo.AilWingCmed , geo.AilYmed , -geo.AilZmed])      # Right aileron lift application point
    geo.AilRapp = rotate(WingRotP, WingRotAxe, a0, AilRapp)
    geo.rAafVec = Qdyn*geo.WingArea*Kdl*CzaWr*rAalpha*rWliftDir

    WingLapp = numpy.array([-geo.WingXmac-0.25*geo.WingCmac , -geo.WingYmac , -geo.WingZmac])      # Left wing lift application point
    geo.WingLapp = rotate(WingRotP, WingRotAxe, a0, WingLapp)
    geo.lWafVec = Qdyn*geo.WingArea*CzaWl*lWalpha*lWliftDir

    AilLapp = numpy.array([-geo.AilWingXmed-0.25*geo.AilWingCmed , -geo.AilYmed , -geo.AilZmed])      # Left aileron lift application point
    geo.AilLapp = rotate(WingRotP, WingRotAxe, a0, AilLapp)
    geo.lAafVec = Qdyn*geo.WingArea*Kdl*CzaWl*lAalpha*lWliftDir

    HtpRotP = numpy.array([-geo.HtpXaxe-0.5*geo.HtpCaxe, 0., -geo.HtpZaxe])
    HtpRotAxe = uY

    HtpRapp = numpy.array([-geo.HtpXmac-0.25*geo.HtpCmac , geo.HtpYmac , -geo.HtpZmac])      # Right HTP lift application point
    geo.HtpRapp = rotate(HtpRotP, HtpRotAxe, trim, HtpRapp)
    geo.rHafVec = Qdyn*geo.HtpArea*CzaHr*rHalpha*rHliftDir
    geo.rEafVec = Qdyn*geo.HtpArea*Kdm*CzaHr*rEalpha*rHliftDir

    HtpLapp = numpy.array([-geo.HtpXmac-0.25*geo.HtpCmac , -geo.HtpYmac , -geo.HtpZmac])      # Left HTP lift application point
    geo.HtpLapp = rotate(HtpRotP, HtpRotAxe, trim, HtpLapp)
    geo.lHafVec = Qdyn*geo.HtpArea*CzaHl*lHalpha*lHliftDir
    geo.lEafVec = Qdyn*geo.HtpArea*Kdm*CzaHl*rEalpha*lHliftDir

    geo.FusApp = numpy.array([-geo.FusLength*0.70 , 0 , -geo.FusHeight*1.20])
    geo.VafVec = ps(Qdyn*geo.VtpArea*CzaV*Valpha*VliftDir, uY)*uY

    geo.VtpApp = numpy.array([-geo.VtpXmac-0.25*geo.VtpCmac , 0 , -geo.VtpZmac])
    geo.RafVec = ps(-Qdyn*geo.VtpArea*Kdn*CzaV*Ralpha*VliftDir, uY)*uY

    geo.lNtfVec = uX*f0*dx
    geo.rNtfVec = uX*f0*dx

    # Moments
    #--------------------------------------------------------------------------------------------------------------

    geo.LiftTotal =   geo.rWafVec + geo.lWafVec + geo.rAafVec + geo.lAafVec \
                    + geo.rHafVec + geo.rEafVec + geo.lHafVec + geo.lEafVec \
                    + geo.VafVec + geo.RafVec

    geo.MliftTotal =    pv(geo.WingRapp,geo.rWafVec) + pv(geo.WingLapp,geo.lWafVec) \
                      + pv(geo.AilRapp,geo.rAafVec) + pv(geo.AilLapp,geo.lAafVec) \
                      + pv(geo.HtpRapp,(geo.rHafVec+geo.rEafVec)) + pv(geo.HtpLapp,(geo.lHafVec+geo.lEafVec)) \
                      + pv(geo.FusApp,geo.VafVec) + pv(geo.VtpApp,geo.RafVec)

    geo.LiftApp = numpy.array([-geo.MliftTotal[1] / geo.LiftTotal[2],
                               geo.MliftTotal[0] / geo.LiftTotal[2],
                               0.])

    geo.MtotalXg = geo.MliftTotal - pv(geo.mgApp,geo.LiftTotal)

    geo.Ftotal =   geo.mg + geo.LiftTotal

    geo.MtotalAnchor = numpy.array([1.35*geo.FusLength , 0.95*geo.WingSpan , 0.50*geo.FusHeight])


    return

