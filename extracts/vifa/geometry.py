#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

import numpy
import math

import data as geo
from tools import rad_deg, renorm


#===================================================================================================================
def geometry(a0,trim):

    # VTP
    #-----------------------------------------------------------------------------------------------------------
    geo.VtpHeight = math.sqrt(geo.VtpAR*geo.VtpArea)

    geo.VtpCroot = 2*geo.VtpArea/(geo.VtpHeight*(1+geo.VtpTR))
    geo.VtpCtip = geo.VtpTR*geo.VtpCroot

    geo.VtpXroot = geo.FusLength*(1-geo.Rcone*(1-geo.TailXwisePos)) - geo.VtpCroot
    geo.VtpXtip = geo.VtpXroot + 0.25*(geo.VtpCroot-geo.VtpCtip) + geo.VtpHeight*math.tan(geo.VtpSweep)

    geo.VtpZroot = geo.FusHeight
    geo.VtpZtip = geo.VtpZroot+geo.VtpHeight
    geo.VtpYtip = 0

    geo.VtpCmac = geo.VtpHeight*(geo.VtpCroot**2+geo.VtpCtip**2+geo.VtpCroot*geo.VtpCtip)/(3*geo.VtpArea)
    geo.VtpXmac = geo.VtpXroot+(geo.VtpXtip-geo.VtpXroot)*geo.VtpHeight*(2*geo.VtpCtip+geo.VtpCroot)/(6*geo.VtpArea)
    geo.VtpZmac = geo.VtpZtip**2*(2*geo.VtpCtip+geo.VtpCroot)/(6*geo.VtpArea)

    geo.RudCtip = (1-geo.RudChordRatio)*geo.VtpCtip
    geo.RudXtip = geo.VtpXtip + geo.RudChordRatio*geo.VtpCtip
    geo.RudZtip = geo.VtpZtip

    geo.RudVtpCroot = (geo.VtpCroot*(1-geo.RudHeightRatioInt) + geo.VtpCtip*geo.RudHeightRatioInt)
    geo.RudCroot = (1-geo.RudChordRatio)*geo.RudVtpCroot
    geo.RudXroot = (geo.VtpXroot*(1-geo.RudHeightRatioInt) + geo.VtpXtip*geo.RudHeightRatioInt) + geo.RudChordRatio*geo.RudVtpCroot
    geo.RudZroot = (geo.VtpZroot*(1-geo.RudHeightRatioInt) + geo.VtpZtip*geo.RudHeightRatioInt)

    # ATTENTION changement de repère
    geo.VtpAxe =   numpy.array([-geo.VtpXtip-0.25*geo.VtpCtip , geo.VtpYtip , -geo.VtpZtip]) \
                 - numpy.array([-geo.VtpXroot-0.25*geo.VtpCroot , 0 , -geo.VtpZroot])
    geo.VtpDir = renorm(geo.VtpAxe)

    # HTP
    #-----------------------------------------------------------------------------------------------------------
    geo.HtpSpan = math.sqrt(geo.HtpAR*geo.HtpArea)
    geo.HtpCaxe = 2*geo.HtpArea/(geo.HtpSpan*(1+geo.HtpTR))
    geo.HtpCtip = geo.HtpTR*geo.HtpCaxe
    geo.HtpYtip = 0.5*geo.HtpSpan

    geo.HtpXaxe = geo.FusLength*(1-geo.Rcone*(1-geo.TailXwisePos)) - geo.HtpCaxe
    geo.HtpYaxe = 0
    geo.HtpZaxe = geo.HtpZwisePos*geo.FusHeight

    geo.HtpXtip = geo.HtpXaxe + 0.25*(geo.HtpCaxe-geo.HtpCtip) + geo.HtpYtip*math.tan(geo.HtpSweep)
    geo.HtpYtip = 0.5*geo.HtpSpan
    geo.HtpZtip = geo.HtpZaxe+geo.HtpYtip*math.tan(geo.HtpDihedral)

    geo.HtpCmac = geo.HtpSpan*(geo.HtpCaxe**2+geo.HtpCtip**2+geo.HtpCaxe*geo.HtpCtip)/(3*geo.HtpArea)
    geo.HtpXmac = geo.HtpXaxe + geo.HtpYtip*(geo.HtpXtip-geo.HtpXaxe)*(geo.HtpCtip*2.+geo.HtpCaxe)/(3*geo.HtpArea)
    geo.HtpYmac = geo.HtpYtip**2*(2*geo.HtpCtip+geo.HtpCaxe)/(3*geo.HtpArea)
    geo.HtpZmac = geo.HtpZaxe+geo.HtpYmac*math.tan(geo.HtpDihedral)

    geo.ElevCtip = (1-geo.ElevChordRatio)*geo.HtpCtip
    geo.ElevXtip = geo.HtpXtip + geo.ElevChordRatio*geo.HtpCtip
    geo.ElevYtip = geo.HtpYtip
    geo.ElevZtip = geo.HtpZtip

    geo.ElevHtpCaxe = (geo.HtpCaxe*(1-geo.ElevSpanRatioInt) + geo.HtpCtip*geo.ElevSpanRatioInt)
    geo.ElevCaxe = (1-geo.ElevChordRatio)*geo.ElevHtpCaxe
    geo.ElevXaxe = (geo.HtpXaxe*(1-geo.ElevSpanRatioInt) + geo.HtpXtip*geo.ElevSpanRatioInt) + geo.ElevChordRatio*geo.ElevHtpCaxe
    geo.ElevYaxe = (geo.HtpYaxe*(1-geo.ElevSpanRatioInt) + geo.HtpYtip*geo.ElevSpanRatioInt)
    geo.ElevZaxe = (geo.HtpZaxe*(1-geo.ElevSpanRatioInt) + geo.HtpZtip*geo.ElevSpanRatioInt)

    # ATTENTION changement de repère
    geo.HtpCdir = numpy.array([-math.cos(trim) , 0. , -math.sin(trim)])

    geo.HtpRaxe =  numpy.array([-geo.HtpXtip-0.25*geo.HtpCtip , geo.HtpYtip , -geo.HtpZtip]) \
                 - numpy.array([-geo.HtpXaxe-0.25*geo.HtpCaxe , 0 , -geo.HtpZaxe])
    geo.HtpRdir = renorm(geo.HtpRaxe)

    geo.HtpAxe = renorm(geo.HtpRaxe)
    geo.HtpRnorm = renorm(geo.HtpRaxe)

    geo.HtpLaxe =  numpy.array([-geo.HtpXtip-0.25*geo.HtpCtip , -geo.HtpYtip , -geo.HtpZtip]) \
                 - numpy.array([-geo.HtpXaxe-0.25*geo.HtpCaxe , 0. , -geo.HtpZaxe])
    geo.HtpLdir = renorm(geo.HtpLaxe)

    # Wing
    # -----------------------------------------------------------------------------------------------------------
    geo.WingSpan = math.sqrt(geo.WingAR*geo.WingArea)
    geo.WingCaxe = 2*geo.WingArea/(geo.WingSpan*(1+geo.WingTR))
    geo.WingCtip = geo.WingTR*geo.WingCaxe
    geo.WingYtip = 0.5*geo.WingSpan
    geo.WingYaxe = 0
    geo.WingZaxe = geo.WingZwisePos*geo.FusHeight

    geo.WingCmac = geo.WingSpan*(geo.WingCaxe**2+geo.WingCtip**2+geo.WingCaxe*geo.WingCtip)/(3*geo.WingArea)
    geo.WingXmac = geo.WingXwisePos*geo.FusLength - 0.25*geo.WingCmac
    geo.WingYmac = geo.WingYtip**2*(2*geo.WingCtip+geo.WingCaxe)/(3*geo.WingArea)
    geo.WingZmac = geo.WingZaxe+geo.WingYmac*math.tan(geo.WingDihedral)

    geo.WingXtipLocal = 0.25*(geo.WingCaxe-geo.WingCtip) + geo.WingYtip*math.tan(geo.WingSweep)

    geo.WingXaxe = geo.WingXmac - geo.WingYtip*geo.WingXtipLocal*(geo.WingCtip*2.+geo.WingCaxe)/(3*geo.WingArea)

    geo.WingXtip = geo.WingXaxe + geo.WingXtipLocal
    geo.WingZtip = geo.WingZaxe+geo.WingYtip*math.tan(geo.WingDihedral)

    geo.tan_phi0 = 0.25*(geo.WingCaxe-geo.WingCtip)/geo.WingYtip + math.tan(geo.WingSweep)

    geo.AilWingCext = (geo.WingCaxe*(1-geo.AilSpanRatioExt) + geo.WingCtip*geo.AilSpanRatioExt)
    geo.AilWingXext = (geo.WingXaxe*(1-geo.AilSpanRatioExt) + geo.WingXtip*geo.AilSpanRatioExt)
    geo.AilCext = (1-geo.AilChordRatio)*geo.AilWingCext
    geo.AilXext = geo.AilWingXext + geo.AilChordRatio*geo.AilWingCext
    geo.AilYext = (geo.WingYaxe*(1-geo.AilSpanRatioExt) + geo.WingYtip*geo.AilSpanRatioExt)
    geo.AilZext = (geo.WingZaxe*(1-geo.AilSpanRatioExt) + geo.WingZtip*geo.AilSpanRatioExt)

    geo.AilWingCint = (geo.WingCaxe*(1-geo.AilSpanRatioInt) + geo.WingCtip*geo.AilSpanRatioInt)
    geo.AilWingXint = (geo.WingXaxe*(1-geo.AilSpanRatioInt) + geo.WingXtip*geo.AilSpanRatioInt)
    geo.AilCint = (1-geo.AilChordRatio)*geo.AilWingCint
    geo.AilXint = geo.AilWingXint + geo.AilChordRatio*geo.AilWingCint
    geo.AilYint = (geo.WingYaxe*(1-geo.AilSpanRatioInt) + geo.WingYtip*geo.AilSpanRatioInt)
    geo.AilZint = (geo.WingZaxe*(1-geo.AilSpanRatioInt) + geo.WingZtip*geo.AilSpanRatioInt)

    geo.AilWingCmed = 0.40*geo.AilWingCint + 0.60*geo.AilWingCext
    geo.AilWingXmed = 0.40*geo.AilWingXint + 0.60*geo.AilWingXext
    geo.AilCmed = 0.40*geo.AilCint + 0.60*geo.AilCext
    geo.AilYmed = 0.40*geo.AilYint + 0.60*geo.AilYext
    geo.AilZmed = 0.40*geo.AilZint + 0.60*geo.AilZext

    # ATTENTION changement de repère
    geo.WingCdir = numpy.array([-math.cos(a0) , 0 , -math.sin(a0)])

    geo.WingRaxe =  numpy.array([-geo.WingXtip-0.25*geo.WingCtip , geo.WingYtip , -geo.WingZtip]) \
                  - numpy.array([-geo.WingXaxe-0.25*geo.WingCaxe , 0. , -geo.WingZaxe])
    geo.WingRdir = renorm(geo.WingRaxe)

    geo.WingLaxe =  numpy.array([-geo.WingXtip-0.25*geo.WingCtip , -geo.WingYtip , -geo.WingZtip]) \
                  - numpy.array([-geo.WingXaxe-0.25*geo.WingCaxe , 0. , -geo.WingZaxe])
    geo.WingLdir = renorm(geo.WingLaxe)


    # Nacelle
    # -------------------------------------------------------------------------------------------------------------------------
    geo.NacYaxe = 0.5*geo.FusWidth + 0.3*(geo.WingYtip-0.5*geo.FusWidth)
    geo.NacXaxe = geo.WingXaxe + (geo.NacYaxe-0.5*geo.FusWidth)*geo.tan_phi0 - 0.55*geo.NacLength
    geo.NacZaxe = geo.NacYaxe*math.tan(geo.WingDihedral) - 0.5*geo.NacWidth

    # ATTENTION changement de repère
    geo.NacApp = numpy.array([-geo.NacXaxe , geo.NacYaxe , -geo.NacZaxe])

