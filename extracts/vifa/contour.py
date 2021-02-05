#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry
"""

import numpy
import math

import data as geo

from tools import renorm, rotate


#===================================================================================================================
def contour(a0,trim,dl,dm,dn):

    uX = numpy.array([1,0,0])
    uY = numpy.array([0,1,0])
    uZ = numpy.array([0,0,1])

    # Fuselage
    # -------------------------------------------------------------------------------------------------------------------------
    # from nose to nose, upper first, back through lower
    geo.FusXYZ = numpy.array([[ 0                 , 0 , 0.35*geo.FusHeight ],
                          [0.01*geo.FusLength , 0 , 0.45*geo.FusHeight ],
                          [0.12*geo.FusLength , 0 , 0.95*geo.FusHeight ],
                          [0.15*geo.FusLength , 0 , geo.FusHeight      ],
                          [0.95*geo.FusLength , 0 , geo.FusHeight      ],
                          [0.98*geo.FusLength , 0 , 0.97*geo.FusHeight ],
                          [geo.FusLength      , 0 , 0.90*geo.FusHeight ],
                          [0.98*geo.FusLength , 0 , 0.80*geo.FusHeight ],
                          [0.95*geo.FusLength , 0 , 0.71*geo.FusHeight ],
                          [0.65*geo.FusLength , 0 , 0.05*geo.FusHeight ],
                          [0.62*geo.FusLength , 0 , 0.01*geo.FusHeight ],
                          [0.60*geo.FusLength , 0 , 0                  ],
                          [0.15*geo.FusLength , 0 , 0                  ],
                          [0.07*geo.FusLength , 0 , 0.04*geo.FusHeight ],
                          [0.01*geo.FusLength , 0 , 0.25*geo.FusHeight ],
                          [0                  , 0 , 0.35*geo.FusHeight ]])

    # VTP
    # -------------------------------------------------------------------------------------------------------------------------
    geo.VtpXYZ = numpy.array([[geo.VtpXroot              , 0 , geo.VtpZroot ],
                          [geo.VtpXtip               , 0 , geo.VtpZtip  ],
                          [geo.RudXtip               , 0 , geo.RudZtip  ],
                          [geo.RudXroot              , 0 , geo.RudZroot ],
                          [geo.RudXroot+geo.RudCroot , 0 , geo.RudZroot ],
                          [geo.VtpXroot+geo.VtpCroot , 0 , geo.VtpZroot ]])


    # Ruder
    # -------------------------------------------------------------------------------------------------------------------------
    p0 = numpy.array([geo.RudXroot              , 0 , geo.RudZroot ])
    p1 = numpy.array([geo.RudXtip               , 0 , geo.RudZtip  ])
    p2 = numpy.array([geo.RudXtip+geo.RudCtip   , 0 , geo.RudZtip  ])
    p3 = numpy.array([geo.RudXroot+geo.RudCroot , 0 , geo.RudZroot ])

    axis = renorm(p0-p1)
    p2 = rotate(p0,axis,dn,p2)
    p3 = rotate(p0,axis,dn,p3)

    geo.RudXYZ = numpy.array([p0,p1,p2,p3,p0])


    # HTP
    # -------------------------------------------------------------------------------------------------------------------------
    geo.HtpRotP = numpy.array([geo.HtpXaxe+0.5*geo.HtpCaxe, 0., geo.HtpZaxe])
    geo.HtpRotAxe = uY

    p0 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.HtpXaxe               ,  geo.HtpYaxe  , geo.HtpZaxe  ]))
    p1 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.HtpXtip               ,  geo.HtpYtip  , geo.HtpZtip  ]))
    p2 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.ElevXtip              ,  geo.ElevYtip , geo.ElevZtip ]))
    p3 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.ElevXaxe              ,  geo.ElevYaxe , geo.ElevZaxe ]))
    p4 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.ElevXaxe+geo.ElevCaxe ,  geo.ElevYaxe , geo.ElevZaxe ]))
    p5 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.HtpXaxe+geo.HtpCaxe   ,  geo.HtpYaxe  , geo.HtpZaxe  ]))

    geo.rHtpXYZ = numpy.array([p0, p1, p2, p3, p4, p5, p0])
    geo.lHtpXYZ = numpy.copy(geo.rHtpXYZ)
    geo.lHtpXYZ[:,1] = -geo.rHtpXYZ[:,1]


    # Elevator
    # -------------------------------------------------------------------------------------------------------------------------
    p0 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.ElevXaxe              ,  geo.ElevYaxe , geo.ElevZaxe ]))
    p1 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.ElevXtip              ,  geo.ElevYtip , geo.ElevZtip ]))
    p2 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.ElevXtip+geo.ElevCtip ,  geo.ElevYtip , geo.ElevZtip ]))
    p3 = rotate(geo.HtpRotP, geo.HtpRotAxe, trim, numpy.array([geo.ElevXaxe+geo.ElevCaxe ,  geo.ElevYaxe , geo.ElevZaxe ]))

    axis = renorm(p1-p0)
    p2 = rotate(p0,axis,dm,p2)
    p3 = rotate(p0,axis,dm,p3)

    geo.rElevXYZ = numpy.array([p0,p1,p2,p3,p0])

    geo.lElevXYZ = numpy.copy(geo.rElevXYZ)
    geo.lElevXYZ[:,1] = -geo.rElevXYZ[:,1]
 

    # Wing
    # -------------------------------------------------------------------------------------------------------------------------
    geo.WingRotP = numpy.array([geo.WingXaxe+0.5*geo.WingCaxe, 0., geo.WingZaxe])
    geo.WingRotAxe = uY

    p0 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.WingXaxe              ,  0            , geo.WingZaxe ]))
    p1 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.WingXtip              ,  geo.WingYtip , geo.WingZtip ]))
    p2 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXext ,  geo.WingYtip , geo.AilZext ],              ))
    p3 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXint               ,  geo.AilYint  , geo.AilZint  ]))
    p4 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXint+geo.AilCint   ,  geo.AilYint  , geo.AilZint  ]))
    p5 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.WingXaxe+geo.WingCaxe ,  0            , geo.WingZaxe ]))

    geo.rWingXYZ = numpy.array([p0, p1, p2, p3, p4, p5, p0])
    geo.lWingXYZ = numpy.copy(geo.rWingXYZ)
    geo.lWingXYZ[:,1] = -geo.rWingXYZ[:,1]


    #
    #
    #
    # geo.rWingXYZ = numpy.array([[geo.WingXaxe              ,  0            , geo.WingZaxe ],
    #                             [geo.WingXtip              ,  geo.WingYtip , geo.WingZtip ],
    #                             [geo.AilXext ,  geo.WingYtip , geo.AilZext ],
    #                             [geo.AilXint               ,  geo.AilYint  , geo.AilZint  ],
    #                             [geo.AilXint+geo.AilCint   ,  geo.AilYint  , geo.AilZint  ],
    #                             [geo.WingXaxe+geo.WingCaxe ,  0            , geo.WingZaxe ],
    #                             [geo.WingXaxe              ,  0            , geo.WingZaxe ]])
    #
    #
    # geo.lWingXYZ = numpy.copy(geo.rWingXYZ)
    # geo.lWingXYZ[:,1] = -geo.rWingXYZ[:,1]
    #
    
 

    # Aileron
    # -------------------------------------------------------------------------------------------------------------------------
    p0 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXint             , geo.AilYint  , geo.AilZint ]))
    p1 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXext             , geo.WingYtip , geo.AilZext ]))
    p2 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXext+geo.AilCext , geo.WingYtip , geo.AilZext ]))
    p3 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXint+geo.AilCint , geo.AilYint  , geo.AilZint ]))

    axis = renorm(p1-p0)
    p2 = rotate(p0,axis,dl,p2)
    p3 = rotate(p0,axis,dl,p3)

    geo.rAilXYZ = numpy.array([p0,p1,p2,p3,p0])

    p0 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXint             , -geo.AilYint  , geo.AilZint ]))
    p1 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXext             , -geo.WingYtip , geo.AilZext ]))
    p2 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXext+geo.AilCext , -geo.WingYtip , geo.AilZext ]))
    p3 = rotate(geo.WingRotP, geo.WingRotAxe, a0, numpy.array([geo.AilXint+geo.AilCint , -geo.AilYint  , geo.AilZint ]))

    axis = renorm(p1-p0)
    p2 = rotate(p0,axis,dl,p2)
    p3 = rotate(p0,axis,dl,p3)

    geo.lAilXYZ = numpy.array([p0,p1,p2,p3,p0])
    
   
    

    # Nacelle
    # -------------------------------------------------------------------------------------------------------------------------
    a = 0.5/(1+math.sqrt(2))

    geo.rNacXYZ = numpy.array([[geo.NacXaxe               , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 1
                          [geo.NacXaxe               , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 2
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 3
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 4
                          [geo.NacXaxe               , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 1
                          [geo.NacXaxe               , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 2
                          [geo.NacXaxe               , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 5
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 6
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 3
                          [geo.NacXaxe               , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 2
                          [geo.NacXaxe               , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 2
                          [geo.NacXaxe               , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 5
                          [geo.NacXaxe               , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 7
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 8
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 6
                          [geo.NacXaxe               , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 5
                          [geo.NacXaxe               , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 7
                          [geo.NacXaxe               , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 9
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 10
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 8
                          [geo.NacXaxe               , geo.NacYaxe+0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 7
                          [geo.NacXaxe               , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 9
                          [geo.NacXaxe               , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 11
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 12
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 10
                          [geo.NacXaxe               , geo.NacYaxe+a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 9
                          [geo.NacXaxe               , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 11
                          [geo.NacXaxe               , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 13
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 14
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 12
                          [geo.NacXaxe               , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe-0.5*geo.NacWidth ],	# 11
                          [geo.NacXaxe               , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 13
                          [geo.NacXaxe               , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 15
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 16
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 14
                          [geo.NacXaxe               , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe-a*geo.NacWidth   ],	# 13
                          [geo.NacXaxe               , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 15
                          [geo.NacXaxe               , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 1
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ],	# 4
                          [geo.NacXaxe+geo.NacLength , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 16
                          [geo.NacXaxe               , geo.NacYaxe-0.5*geo.NacWidth , geo.NacZaxe+a*geo.NacWidth   ],	# 15
                          [geo.NacXaxe               , geo.NacYaxe-a*geo.NacWidth   , geo.NacZaxe+0.5*geo.NacWidth ]])  # 1

    geo.lNacXYZ = numpy.copy(geo.rNacXYZ)
    geo.lNacXYZ[:,1] = -geo.rNacXYZ[:,1]
    
