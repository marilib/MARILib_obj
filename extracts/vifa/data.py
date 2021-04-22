#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

from tools import rad_deg

FusWidth = 4
FusHeight = 4
FusLength = 40

WingArea = 200
WingAR = 7
WingTR = 0.25
WingSweep = rad_deg(25)
WingDihedral = rad_deg(5)
WingXwisePos = 0.45
WingZwisePos = 0.15

AilSpanRatioExt = 0.95
AilSpanRatioInt = 0.65
AilChordRatio = 0.65

NacLength = 4
NacWidth = 2

HtpArea = 60
HtpAR = 4
HtpTR = 0.35
HtpSweep = rad_deg(30)
HtpDihedral = rad_deg(5)

ElevSpanRatioInt = 0.20
ElevChordRatio = 0.65


VtpArea = 30
VtpAR = 1.5
VtpTR = 0.35
VtpSweep = rad_deg(30)

RudHeightRatioInt = 0.05
RudChordRatio = 0.65


TailXwisePos = 0.825
HtpZwisePos = 0.80

Rnose = 0.15
Rcone = 0.35

# VTP
#-----------------------------------------------------------------------------------------------------------
VtpHeight = None
VtpCroot = None
VtpCtip = None

VtpXroot = None
VtpXtip = None

VtpZroot = None
VtpZtip = None
VtpYtip = None

VtpCmac = None
VtpXmac = None
VtpZmac = None

RudCtip = None
RudXtip = None
RudZtip = None

RudVtpCroot = None
RudCroot = None
RudXroot = None
RudZroot = None

VtpAxe = None
VtpDir = None

# HTP
#-----------------------------------------------------------------------------------------------------------
HtpSpan = None
HtpCaxe = None
HtpCtip = None
HtpYtip = None

HtpXaxe = None
HtpYaxe = None
HtpZaxe = None

HtpXtip = None
HtpYtip = None
HtpZtip = None

HtpCmac = None
HtpXmac = None
HtpYmac = None
HtpZmac = None

ElevCtip = None
ElevXtip = None
ElevYtip = None
ElevZtip = None

ElevHtpCaxe = None
ElevCaxe = None
ElevXaxe = None
ElevYaxe = None
ElevZaxe = None

HtpCdir = None

HtpRaxe = None
HtpRdir = None

HtpLaxe = None
HtpLdir = None

# Wing
# -----------------------------------------------------------------------------------------------------------
WingSpan = None
WingCaxe = None
WingCtip = None
WingYtip = None
WingYaxe = None
WingZaxe = None

WingCmac = None
WingXmac = None
WingYmac = None
WingZmac = None

WingXtipLocal = None

WingXaxe = None

WingXtip = None
WingZtip = None

tan_phi0 = None

AilWingCext = None
AilWingXext = None
AilCext = None
AilXext = None
AilYext = None
AilZext = None

AilWingCint = None
AilWingXint = None
AilCint = None
AilXint = None
AilYint = None
AilZint = None

AilWingCmed = None
AilWingXmed = None
AilCmed = None
AilYmed = None
AilZmed = None

WingCdir = None

WingRaxe = None
WingRdir = None

WingLaxe = None
WingLdir = None

# Nacelle
# -------------------------------------------------------------------------------------------------------------------------
NacYaxe = None
NacXaxe = None
NacZaxe = None

NacApp = None

# Contours
# -------------------------------------------------------------------------------------------------------------------------
FusXYZ = None

VtpXYZ = None
RudXYZ = None

rHtpXYZ = None
lHtpXYZ = None
rElevXYZ = None
lElevXYZ = None

rWingXYZ = None
lWingXYZ = None
rAilXYZ = None
lAilXYZ = None

rNacXYZ = None
lNacXYZ = None

# Forces
# -------------------------------------------------------------------------------------------------------------------------
WingRapp = None
rWafVec = None

WingLapp = None
lWafVec = None

AilRapp = None
rAafVec = None

AilLapp = None
lAafVec = None

HtpRapp = None
rHafVec = None
rEafVec = None

HtpLapp = None
lHafVec = None
lEafVec = None

FusApp = None
VafVec = None

VtpApp = None
RafVec = None

# Moments
# -------------------------------------------------------------------------------------------------------------------------
MtotalXg = None
MtotalAnchor = None
