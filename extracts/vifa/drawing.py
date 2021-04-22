#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

import numpy

import data as geo

import matplotlib.pyplot as plt


#===================================================================================================================
def init_plot_view(window_title,plot_title):

    fig,axes = plt.subplots(1,1)
    fig.canvas.set_window_title(window_title)
    fig.suptitle(plot_title, fontsize=14)
    fig.set_size_inches(11,11)
    axes.set_aspect('equal', 'box')

    plt.plot(numpy.array([0,100,100,0,0]), numpy.array([0,0,100,100,0]))      # Draw a square box of 100m side

    xSideView = 50 - (geo.WingXmac + 0.25*geo.WingCmac)       # Top view positionning
    ySideView = 82

    xBackView = 50
    yBackView = 55

    xTopView = 50 - (geo.WingXmac + 0.25*geo.WingCmac)      # Top view positionning
    yTopView = 25

    plt.text(10,ySideView,"LEFT")
    plt.text(10,yTopView,"TOP")
    plt.text(10,yBackView,"BACK")

    return xTopView,yTopView,xSideView,ySideView,xBackView,yBackView

#===================================================================================================================
def plot_side_view(plt,x_shift,y_shift,dl,dm,kScale):

    eps = numpy.finfo(float).eps

    plt.arrow(x_shift-geo.LiftApp[0], y_shift-geo.LiftApp[2], -kScale*geo.LiftTotal[0], -kScale*geo.LiftTotal[2], linewidth=0.1, width=0.2, color="darkturquoise", shape="full", zorder=1)

    plt.arrow(x_shift-geo.mgApp[0], y_shift-geo.mgApp[2], -kScale*geo.mg[0], -kScale*geo.mg[2], linewidth=0.1, width=0.2, color="black", shape="full", zorder=1)

    plt.arrow(x_shift-geo.NacLapp[0], y_shift-geo.NacLapp[2], -kScale*geo.lNtfVec[0], -kScale*geo.lNtfVec[2], linewidth=0.1, width=0.2, color="darkred", shape="full", zorder=1)

    if (numpy.linalg.norm(geo.rWafVec)>eps):
        plt.arrow(x_shift-geo.WingRapp[0], y_shift-geo.WingRapp[2], -kScale*geo.rWafVec[0], -kScale*geo.rWafVec[2], linewidth=0.1, width=0.2, color="blue", shape="left", zorder=1)

    plt.fill(x_shift+geo.FusXYZ[0:,0], y_shift+geo.FusXYZ[0:,2], c="white", zorder=2)
    plt.plot(x_shift+geo.FusXYZ[0:,0], y_shift+geo.FusXYZ[0:,2], c="grey", zorder=3)

    if (numpy.linalg.norm(geo.lWafVec)>eps):
        plt.arrow(x_shift-geo.WingLapp[0], y_shift-geo.WingLapp[2], -kScale*geo.lWafVec[0], -kScale*geo.lWafVec[2], linewidth=0.1, width=0.2, color="blue", shape="right", zorder=4)

    plt.fill(x_shift+geo.VtpXYZ[0:,0], y_shift+geo.VtpXYZ[0:,2], c="white", zorder=5)
    plt.plot(x_shift+geo.VtpXYZ[0:,0], y_shift+geo.VtpXYZ[0:,2], c="grey", zorder=6)
    plt.fill(x_shift+geo.RudXYZ[0:,0], y_shift+geo.RudXYZ[0:,2], c="white", zorder=7)
    plt.plot(x_shift+geo.RudXYZ[0:,0], y_shift+geo.RudXYZ[0:,2], c="grey", zorder=8)

    if (dm>0):
        plt.fill(x_shift+geo.lHtpXYZ[0:,0], y_shift+geo.lHtpXYZ[0:,2], c="white", zorder=9)
        plt.plot(x_shift+geo.lHtpXYZ[0:,0], y_shift+geo.lHtpXYZ[0:,2], c="grey", zorder=10)
        plt.fill(x_shift+geo.lElevXYZ[0:,0], y_shift+geo.lElevXYZ[0:,2], c="white", zorder=11)
        plt.plot(x_shift+geo.lElevXYZ[0:,0], y_shift+geo.lElevXYZ[0:,2], c="grey", zorder=12)
    else:
        plt.fill(x_shift+geo.lElevXYZ[0:,0], y_shift+geo.lElevXYZ[0:,2], c="white", zorder=9)
        plt.plot(x_shift+geo.lElevXYZ[0:,0], y_shift+geo.lElevXYZ[0:,2], c="grey", zorder=10)
        plt.fill(x_shift+geo.lHtpXYZ[0:,0], y_shift+geo.lHtpXYZ[0:,2], c="white", zorder=11)
        plt.plot(x_shift+geo.lHtpXYZ[0:,0], y_shift+geo.lHtpXYZ[0:,2], c="grey", zorder=12)

    if (numpy.linalg.norm(geo.lHafVec)>eps):
        plt.arrow(x_shift-geo.HtpLapp[0], y_shift-geo.HtpLapp[2], -kScale*geo.lHafVec[0], -kScale*geo.lHafVec[2], linewidth=0.1, width=0.2, color="blue", shape="right", zorder=13)

    if (numpy.linalg.norm(geo.lEafVec)>eps):
        plt.arrow(x_shift-geo.HtpLapp[0], y_shift-geo.HtpLapp[2], -kScale*geo.lEafVec[0], -kScale*geo.lEafVec[2], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=14)

    if (numpy.linalg.norm(geo.lHtpRafVec)>eps):
        plt.arrow(x_shift-geo.HtpLapp[0], y_shift-geo.HtpLapp[2], -kScale*geo.lHtpRafVec[0], -kScale*geo.lHtpRafVec[2], linewidth=0.1, width=0.2, color="darkcyan", shape="full", zorder=14)

    if (dl>0):
        plt.fill(x_shift+geo.lAilXYZ[0:,0], y_shift+geo.lAilXYZ[0:,2], c="white", zorder=15)
        plt.plot(x_shift+geo.lAilXYZ[0:,0], y_shift+geo.lAilXYZ[0:,2], c="grey", zorder=16)
        plt.fill(x_shift+geo.lWingXYZ[0:,0], y_shift+geo.lWingXYZ[0:,2], c="white", zorder=17)
        plt.plot(x_shift+geo.lWingXYZ[0:,0], y_shift+geo.lWingXYZ[0:,2], c="grey", zorder=18)
    else:
        plt.fill(x_shift+geo.lWingXYZ[0:,0], y_shift+geo.lWingXYZ[0:,2], c="white", zorder=15)
        plt.plot(x_shift+geo.lWingXYZ[0:,0], y_shift+geo.lWingXYZ[0:,2], c="grey", zorder=16)
        plt.fill(x_shift+geo.lAilXYZ[0:,0], y_shift+geo.lAilXYZ[0:,2], c="white", zorder=17)
        plt.plot(x_shift+geo.lAilXYZ[0:,0], y_shift+geo.lAilXYZ[0:,2], c="grey", zorder=18)

    if (numpy.linalg.norm(geo.lAafVec)>eps):
        plt.arrow(x_shift-geo.AilLapp[0], y_shift-geo.AilLapp[2], -kScale*geo.lAafVec[0], -kScale*geo.lAafVec[2], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=19)

    if (numpy.linalg.norm(geo.lRafVec)>eps):
        plt.arrow(x_shift-geo.RotLapp[0], y_shift-geo.RotLapp[2], -kScale*geo.lRafVec[0], -kScale*geo.lRafVec[2], linewidth=0.1, width=0.2, color="darkcyan", shape="full", zorder=19)


    if (geo.VafVec[1]<-eps):
        plt.arrow(x_shift-geo.FusApp[0], y_shift-geo.FusApp[2], -kScale*geo.VafVec[0], -kScale*geo.VafVec[2], linewidth=0.1, width=0.2, color="blue", shape="full", zorder=20)

    if (geo.RafVec[1]<-eps):
        plt.arrow(x_shift-geo.VtpApp[0], y_shift-geo.VtpApp[2], -kScale*geo.RafVec[0], -kScale*geo.RafVec[2], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=20)

    plt.fill(x_shift+geo.rNacXYZ[0:,0], y_shift+geo.rNacXYZ[0:,2], c="white", zorder=20)
    plt.plot(x_shift+geo.lNacXYZ[0:,0], y_shift+geo.lNacXYZ[0:,2], c="grey", zorder=21)

    if (numpy.linalg.norm(geo.MtotalXg)>eps):
        plt.arrow(x_shift+geo.MtotalAnchor[0], y_shift+geo.MtotalAnchor[2], 0, -0.1*kScale*geo.MtotalXg[1], linewidth=0.2, width=0.4, color="grey", shape="full", zorder=22)



#===================================================================================================================
def plot_back_view(plt,x_shift,y_shift,kScale):

    plt.plot(x_shift+geo.rNacXYZ[0:,1], y_shift+geo.rNacXYZ[0:,2], c="grey", zorder=1)
    plt.plot(x_shift+geo.lNacXYZ[0:,1], y_shift+geo.lNacXYZ[0:,2], c="grey", zorder=1)

    plt.fill(x_shift+geo.rWingXYZ[0:,1], y_shift+geo.rWingXYZ[0:,2], c="white", zorder=2)
    plt.plot(x_shift+geo.rWingXYZ[0:,1], y_shift+geo.rWingXYZ[0:,2], c="grey", zorder=3)
    plt.plot(x_shift+geo.rAilXYZ[0:,1], y_shift+geo.rAilXYZ[0:,2], c="grey", zorder=4)

    plt.fill(x_shift+geo.lWingXYZ[0:,1], y_shift+geo.lWingXYZ[0:,2], c="white", zorder=2)
    plt.plot(x_shift+geo.lWingXYZ[0:,1], y_shift+geo.lWingXYZ[0:,2], c="grey", zorder=3)
    plt.plot(x_shift+geo.lAilXYZ[0:,1], y_shift+geo.lAilXYZ[0:,2], c="grey", zorder=4)

    plt.plot(x_shift+geo.FusXYZ[0:,1], y_shift+geo.FusXYZ[0:,2], c="grey", zorder=5)

    plt.fill(x_shift+geo.rHtpXYZ[0:,1], y_shift+geo.rHtpXYZ[0:,2], c="white", zorder=6)
    plt.plot(x_shift+geo.rHtpXYZ[0:,1], y_shift+geo.rHtpXYZ[0:,2], c="grey", zorder=7)
    plt.plot(x_shift+geo.rElevXYZ[0:,1], y_shift+geo.rElevXYZ[0:,2], c="grey", zorder=8)

    plt.fill(x_shift+geo.lHtpXYZ[0:,1], y_shift+geo.lHtpXYZ[0:,2], c="white", zorder=6)
    plt.plot(x_shift+geo.lHtpXYZ[0:,1], y_shift+geo.lHtpXYZ[0:,2], c="grey", zorder=7)
    plt.plot(x_shift+geo.lElevXYZ[0:,1], y_shift+geo.lElevXYZ[0:,2], c="grey", zorder=8)

    plt.fill(x_shift+geo.VtpXYZ[0:,1], y_shift+geo.VtpXYZ[0:,2], c="white", zorder=9)
    plt.plot(x_shift+geo.VtpXYZ[0:,1], y_shift+geo.VtpXYZ[0:,2], c="grey", zorder=10)
    plt.plot(x_shift+geo.RudXYZ[0:,1], y_shift+geo.RudXYZ[0:,2], c="grey", zorder=11)


    eps = numpy.finfo(float).eps

    plt.arrow(x_shift+geo.LiftApp[1], y_shift-geo.LiftApp[2], kScale*geo.LiftTotal[1], -kScale*geo.LiftTotal[2], linewidth=0.1, width=0.2, color="darkturquoise", shape="full", zorder=20)

    plt.arrow(x_shift+geo.mgApp[1], y_shift-geo.mgApp[2], kScale*geo.mg[1], -kScale*geo.mg[2], linewidth=0.1, width=0.2, color="black", shape="full", zorder=20)

    if (numpy.linalg.norm(geo.rWafVec)>eps):
        plt.arrow(x_shift+geo.WingRapp[1], y_shift-geo.WingRapp[2], kScale*geo.rWafVec[1], -kScale*geo.rWafVec[2], linewidth=0.1, width=0.2, color="blue", shape="left", zorder=20)
    if (numpy.linalg.norm(geo.lWafVec)>eps):
        plt.arrow(x_shift+geo.WingLapp[1], y_shift-geo.WingLapp[2], kScale*geo.lWafVec[1], -kScale*geo.lWafVec[2], linewidth=0.1, width=0.2, color="blue", shape="right", zorder=20)
    if (numpy.linalg.norm(geo.rAafVec)>eps):
        plt.arrow(x_shift+geo.AilRapp[1], y_shift-geo.AilRapp[2], kScale*geo.rAafVec[1], -kScale*geo.rAafVec[2], linewidth=0.1, width=0.2, color="brown", shape="right", zorder=20)
    if (numpy.linalg.norm(geo.lAafVec)>eps):
        plt.arrow(x_shift+geo.AilLapp[1], y_shift-geo.AilLapp[2], kScale*geo.lAafVec[1], -kScale*geo.lAafVec[2], linewidth=0.1, width=0.2, color="brown", shape="left", zorder=20)
    if (numpy.linalg.norm(geo.rRafVec)>eps):
        plt.arrow(x_shift+geo.RotRapp[1], y_shift-geo.RotRapp[2], kScale*geo.rRafVec[1], -kScale*geo.rRafVec[2], linewidth=0.1, width=0.2, color="darkcyan", shape="left", zorder=20)
    if (numpy.linalg.norm(geo.lRafVec)>eps):
        plt.arrow(x_shift+geo.RotLapp[1], y_shift-geo.RotLapp[2], kScale*geo.lRafVec[1], -kScale*geo.lRafVec[2], linewidth=0.1, width=0.2, color="darkcyan", shape="right", zorder=20)

    if (numpy.linalg.norm(geo.rHafVec)>eps):
        plt.arrow(x_shift+geo.HtpRapp[1], y_shift-geo.HtpRapp[2], kScale*geo.rHafVec[1], -kScale*geo.rHafVec[2], linewidth=0.1, width=0.2, color="blue", shape="left", zorder=20)
    if (numpy.linalg.norm(geo.lHafVec)>eps):
        plt.arrow(x_shift+geo.HtpLapp[1], y_shift-geo.HtpLapp[2], kScale*geo.lHafVec[1], -kScale*geo.lHafVec[2], linewidth=0.1, width=0.2, color="blue", shape="right", zorder=20)
    if (numpy.linalg.norm(geo.rEafVec)>eps):
        plt.arrow(x_shift+geo.HtpRapp[1], y_shift-geo.HtpRapp[2], kScale*geo.rEafVec[1], -kScale*geo.rEafVec[2], linewidth=0.1, width=0.2, color="brown", shape="right", zorder=20)
    if (numpy.linalg.norm(geo.lEafVec)>eps):
        plt.arrow(x_shift+geo.HtpLapp[1], y_shift-geo.HtpLapp[2], kScale*geo.lEafVec[1], -kScale*geo.lEafVec[2], linewidth=0.1, width=0.2, color="brown", shape="left", zorder=20)
    if (numpy.linalg.norm(geo.rHtpRafVec)>eps):
        plt.arrow(x_shift+geo.HtpRapp[1], y_shift-geo.HtpRapp[2], kScale*geo.rHtpRafVec[1], -kScale*geo.rHtpRafVec[2], linewidth=0.1, width=0.2, color="darkcyan", shape="left", zorder=20)
    if (numpy.linalg.norm(geo.lHtpRafVec)>eps):
        plt.arrow(x_shift+geo.HtpLapp[1], y_shift-geo.HtpLapp[2], kScale*geo.lHtpRafVec[1], -kScale*geo.lHtpRafVec[2], linewidth=0.1, width=0.2, color="darkcyan", shape="right", zorder=20)

    if (numpy.linalg.norm(geo.VafVec)>eps):
        plt.arrow(x_shift+geo.FusApp[1], y_shift-geo.FusApp[2], kScale*geo.VafVec[1], -kScale*geo.VafVec[2], linewidth=0.1, width=0.2, color="blue", shape="left", zorder=20)
    if (numpy.linalg.norm(geo.RafVec)>eps):
        plt.arrow(x_shift+geo.VtpApp[1], y_shift-geo.VtpApp[2], kScale*geo.RafVec[1], -kScale*geo.RafVec[2], linewidth=0.1, width=0.2, color="brown", shape="right", zorder=20)
    if (numpy.linalg.norm(geo.rotVafVec)>eps):
        plt.arrow(x_shift+geo.VtpApp[1], y_shift-geo.VtpApp[2], kScale*geo.rotVafVec[1], -kScale*geo.rotVafVec[2], linewidth=0.1, width=0.2, color="darkcyan", shape="full", zorder=20)

    if (numpy.linalg.norm(geo.MtotalXg)>eps):
        plt.arrow(x_shift+geo.MtotalAnchor[1], y_shift+geo.MtotalAnchor[2], 0, -0.1*kScale*geo.MtotalXg[0], linewidth=0.2, width=0.4, color="grey", shape="full", zorder=22)


#===================================================================================================================
def plot_top_view(plt,x_shift,y_shift,kScale):

    plt.plot(x_shift+geo.rNacXYZ[0:,0], y_shift+geo.rNacXYZ[0:,1], c="grey", zorder=1)
    plt.plot(x_shift+geo.lNacXYZ[0:,0], y_shift+geo.lNacXYZ[0:,1], c="grey", zorder=1)

    plt.fill(x_shift+geo.rWingXYZ[0:,0], y_shift+geo.rWingXYZ[0:,1], c="white", zorder=2)
    plt.plot(x_shift+geo.rWingXYZ[0:,0], y_shift+geo.rWingXYZ[0:,1], c="grey", zorder=3)
    plt.plot(x_shift+geo.rAilXYZ[0:,0], y_shift+geo.rAilXYZ[0:,1], c="grey", zorder=4)

    plt.fill(x_shift+geo.lWingXYZ[0:,0], y_shift+geo.lWingXYZ[0:,1], c="white", zorder=2)
    plt.plot(x_shift+geo.lWingXYZ[0:,0], y_shift+geo.lWingXYZ[0:,1], c="grey", zorder=3)
    plt.plot(x_shift+geo.lAilXYZ[0:,0], y_shift+geo.lAilXYZ[0:,1], c="grey", zorder=4)

    plt.plot(x_shift+geo.FusXYZ[0:,0], y_shift+geo.FusXYZ[0:,1], c="grey", zorder=5)

    plt.fill(x_shift+geo.rHtpXYZ[0:,0], y_shift+geo.rHtpXYZ[0:,1], c="white", zorder=6)
    plt.plot(x_shift+geo.rHtpXYZ[0:,0], y_shift+geo.rHtpXYZ[0:,1], c="grey", zorder=7)
    plt.plot(x_shift+geo.rElevXYZ[0:,0], y_shift+geo.rElevXYZ[0:,1], c="grey", zorder=8)

    plt.fill(x_shift+geo.lHtpXYZ[0:,0], y_shift+geo.lHtpXYZ[0:,1], c="white", zorder=6)
    plt.plot(x_shift+geo.lHtpXYZ[0:,0], y_shift+geo.lHtpXYZ[0:,1], c="grey", zorder=7)
    plt.plot(x_shift+geo.lElevXYZ[0:,0], y_shift+geo.lElevXYZ[0:,1], c="grey", zorder=8)

    plt.fill(x_shift+geo.VtpXYZ[0:,0], y_shift+geo.VtpXYZ[0:,1], c="white", zorder=9)
    plt.plot(x_shift+geo.VtpXYZ[0:,0], y_shift+geo.VtpXYZ[0:,1], c="grey", zorder=10)
    plt.fill(x_shift+geo.RudXYZ[0:,0], y_shift+geo.RudXYZ[0:,1], c="white", zorder=11)
    plt.plot(x_shift+geo.RudXYZ[0:,0], y_shift+geo.RudXYZ[0:,1], c="grey", zorder=12)

    eps = numpy.finfo(float).eps

    plt.arrow(x_shift-geo.NacLapp[0], y_shift+geo.NacLapp[1], -kScale*geo.lNtfVec[0], kScale*geo.lNtfVec[1], linewidth=0.1, width=0.2, color="darkred", shape="full", zorder=20)

    plt.arrow(x_shift-geo.NacRapp[0], y_shift+geo.NacRapp[1], -kScale*geo.rNtfVec[0], kScale*geo.rNtfVec[1], linewidth=0.1, width=0.2, color="darkred", shape="full", zorder=20)

    if (geo.rWafVec[2]<-eps):
        plt.arrow(x_shift-geo.WingRapp[0], y_shift+geo.WingRapp[1], -kScale*geo.rWafVec[0], kScale*geo.rWafVec[1], linewidth=0.1, width=0.2, color="blue", shape="left", zorder=20)
    if (geo.lWafVec[2]<-eps):
        plt.arrow(x_shift-geo.WingLapp[0], y_shift+geo.WingLapp[1], -kScale*geo.lWafVec[0], kScale*geo.lWafVec[1], linewidth=0.1, width=0.2, color="blue", shape="right", zorder=20)
    if (geo.rAafVec[2]<-eps):
        plt.arrow(x_shift-geo.AilRapp[0], y_shift+geo.AilRapp[1], -kScale*geo.rAafVec[0], kScale*geo.rAafVec[1], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=20)
    if (geo.lAafVec[2]<-eps):
        plt.arrow(x_shift-geo.AilLapp[0], y_shift+geo.AilLapp[1], -kScale*geo.lAafVec[0], kScale*geo.lAafVec[1], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=20)

    if (geo.rHafVec[2]<-eps):
        plt.arrow(x_shift-geo.HtpRapp[0], y_shift+geo.HtpRapp[1], -kScale*geo.rHafVec[0], kScale*geo.rHafVec[1], linewidth=0.1, width=0.2, color="blue", shape="left", zorder=20)
    if (geo.lHafVec[2]<-eps):
        plt.arrow(x_shift-geo.HtpLapp[0], y_shift+geo.HtpLapp[1], -kScale*geo.lHafVec[0], kScale*geo.lHafVec[1], linewidth=0.1, width=0.2, color="blue", shape="right", zorder=20)
    if (geo.rEafVec[2]<-eps):
        plt.arrow(x_shift-geo.HtpRapp[0], y_shift+geo.HtpRapp[1], -kScale*geo.rEafVec[0], kScale*geo.rEafVec[1], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=20)
    if (geo.lEafVec[2]<-eps):
        plt.arrow(x_shift-geo.HtpLapp[0], y_shift+geo.HtpLapp[1], -kScale*geo.lEafVec[0], kScale*geo.lEafVec[1], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=20)

    if (numpy.linalg.norm(geo.VafVec)>eps):
        plt.arrow(x_shift-geo.FusApp[0], y_shift+geo.FusApp[1], -kScale*geo.VafVec[0], kScale*geo.VafVec[1], linewidth=0.1, width=0.2, color="blue", shape="full", zorder=20)
    if (numpy.linalg.norm(geo.RafVec)>eps):
        plt.arrow(x_shift-geo.VtpApp[0], y_shift+geo.VtpApp[1], -kScale*geo.RafVec[0], kScale*geo.RafVec[1], linewidth=0.1, width=0.2, color="brown", shape="full", zorder=20)
    if (numpy.linalg.norm(geo.rotVafVec)>eps):
        plt.arrow(x_shift-geo.VtpApp[0], y_shift+geo.VtpApp[1], -kScale*geo.rotVafVec[0], kScale*geo.rotVafVec[1], linewidth=0.1, width=0.2, color="darkcyan", shape="full", zorder=20)

    if (numpy.linalg.norm(geo.MtotalXg)>eps):
        plt.arrow(x_shift+geo.MtotalAnchor[0], y_shift, 0, -0.1*kScale*geo.MtotalXg[2], linewidth=0.2, width=0.4, color="grey", shape="full", zorder=22)


