#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import unit, util



#-----------------------------------------------------------------------------------------------------------------------
# Physical components
#-----------------------------------------------------------------------------------------------------------------------

class Component(object):
    def __init__(self, airplane):
        self.airplane = airplane

        self.wet_area = 0.
        self.aero_length = 1.
        self.mass = 0.

        self.form_factor = 0.

    def eval_geometry(self):
        pass

    def eval_mass(self):
        pass

    def eval(self):
        self.eval_geometry()
        self.eval_mass()

    def sketch_curve(self, name): # TODO: is the docstring up to date ?
        """Contour curves for 3 view drawing
        nose1 : modern nose (A220, A350, 787)
        cone1 : classical tail cone
        sec1 : circle
        """
        curve = {
        "nose1":np.array([[ 0.0000 , 0.3339 , 0.3339 , 0.0000 ,  0.0000 ] ,
                          [ 0.0050 , 0.3848 , 0.3084 , 0.0335 , -0.0335 ] ,
                          [ 0.0150 , 0.4253 , 0.2881 , 0.0652 , -0.0652 ] ,
                          [ 0.0500 , 0.5033 , 0.2490 , 0.1101 , -0.1101 ] ,
                          [ 0.1000 , 0.5811 , 0.2100 , 0.1585 , -0.1585 ] ,
                          [ 0.1800 , 0.6808 , 0.1600 , 0.2215 , -0.2215 ] ,
                          [ 0.2773 , 0.7704 , 0.1151 , 0.2859 , -0.2859 ] ,
                          [ 0.4191 , 0.8562 , 0.0721 , 0.3624 , -0.3624 ] ,
                          [ 0.5610 , 0.9198 , 0.0402 , 0.4211 , -0.4211 ] ,
                          [ 0.7738 , 0.9816 , 0.0092 , 0.4761 , -0.4761 ] ,
                          [ 0.9156 , 0.9962 , 0.0019 , 0.4976 , -0.4976 ] ,
                          [ 1.0000 , 1.0000 , 0.0000 , 0.5000 , -0.5000 ]]),

        "cone1":np.array([[ 0.0000 , 1.0000 , 0.0000 , 0.5000 , -0.5000 ] ,
                          [ 0.0213 , 1.0000 , 0.0082 , 0.5000 , -0.5000 ] ,
                          [ 0.0638 , 1.0000 , 0.0230 , 0.4956 , -0.4956 ] ,
                          [ 0.1064 , 1.0000 , 0.0393 , 0.4875 , -0.4875 ] ,
                          [ 0.1489 , 1.0000 , 0.0556 , 0.4794 , -0.4794 ] ,
                          [ 0.1915 , 1.0000 , 0.0786 , 0.4720 , -0.4720 ] ,
                          [ 0.2766 , 1.0000 , 0.1334 , 0.4566 , -0.4566 ] ,
                          [ 0.3617 , 1.0000 , 0.1964 , 0.4330 , -0.4330 ] ,
                          [ 0.4894 , 1.0000 , 0.3024 , 0.3822 , -0.3822 ] ,
                          [ 0.6170 , 1.0000 , 0.4159 , 0.3240 , -0.3240 ] ,
                          [ 0.7447 , 1.0000 , 0.5374 , 0.2577 , -0.2577 ] ,
                          [ 0.8723 , 1.0000 , 0.6627 , 0.1834 , -0.1834 ] ,
                          [ 0.8936 , 0.9963 , 0.6901 , 0.1679 , -0.1679 ] ,
                          [ 0.9149 , 0.9881 , 0.7139 , 0.1524 , -0.1524 ] ,
                          [ 0.9362 , 0.9800 , 0.7413 , 0.1333 , -0.1333 ] ,
                          [ 0.9574 , 0.9652 , 0.7687 , 0.1097 , -0.1097 ] ,
                          [ 0.9787 , 0.9533 , 0.8043 , 0.0788 , -0.0788 ] ,
                          [ 0.9894 , 0.9377 , 0.8280 , 0.0589 , -0.0589 ] ,
                          [ 1.0000 , 0.9103 , 0.8784 , 0.0162 , -0.0162 ]]),

        "sec1":np.array([[  0.5000000 , 0.0000000 ,  0.0000000 ] ,
                         [  0.4903926 , 0.0975452 , -0.0975452 ] ,
                         [  0.4619398 , 0.1913417 , -0.1913417 ] ,
                         [  0.4157348 , 0.2777851 , -0.2777851 ] ,
                         [  0.3535534 , 0.3535534 , -0.3535534 ] ,
                         [  0.2777851 , 0.4157348 , -0.4157348 ] ,
                         [  0.1913417 , 0.4619398 , -0.4619398 ] ,
                         [  0.0975452 , 0.4903926 , -0.4903926 ] ,
                         [  0.0000000 , 0.5000000 , -0.5000000 ] ,
                         [- 0.0975452 , 0.4903926 , -0.4903926 ] ,
                         [- 0.1913417 , 0.4619398 , -0.4619398 ] ,
                         [- 0.2777851 , 0.4157348 , -0.4157348 ] ,
                         [- 0.3535534 , 0.3535534 , -0.3535534 ] ,
                         [- 0.4157348 , 0.2777851 , -0.2777851 ] ,
                         [- 0.4619398 , 0.1913417 , -0.1913417 ] ,
                         [- 0.4903926 , 0.0975452 , -0.0975452 ] ,
                         [- 0.5000000 , 0.0000000 ,  0.0000000 ]])
        }
        return [curve[n] for n in name]


class Cabin(Component):
    def __init__(self, airplane, n_pax, n_aisle, n_front):
        super(Cabin, self).__init__(airplane)

        self.n_pax = n_pax
        self.n_aisle = n_aisle
        self.n_front = n_front

        self.width = None
        self.length = None

        self.m_furnishing = None
        self.m_op_item = None

        self.seat_pitch = unit.m_inch(32)
        self.seat_width = unit.m_inch(19)
        self.aisle_width = unit.m_inch(20)

    def eval_geometry(self):
        self.width = self.seat_width*self.n_front + self.aisle_width*self.n_aisle + 0.08
        self.length = self.seat_pitch*(self.n_pax/self.n_front) + 2*self.width

    def eval_mass(self):
        self.m_furnishing = (0.063*self.n_pax**2 + 9.76*self.n_pax)                     # Furnishings mass
        self.m_op_item = max(160., 5.2*(self.n_pax*self.airplane.design_range*1e-6))    # Operator items mass
        self.mass = self.m_furnishing + self.m_op_item

    def sketch_3view(self, side=None):
        return "cabin", {}


class Fuselage(Component):
    def __init__(self, airplane):
        super(Fuselage, self).__init__(airplane)

        self.width = None
        self.height = None
        self.length = None
        self.cabin_center = None

        self.position = 0.
        self.wall_thickness = 0.15  # Overall fuselage wall thickness
        self.front_ratio = 1.2      # Cabin start over fuselage width
        self.rear_ratio = 1.5       # Cabin end to tail cone over fuselage width

        self.nose_cone_length = None
        self.tail_cone_length = None

        self.form_factor = 1.05 # Form factor for drag calculation

    def eval_geometry(self):
        cabin = self.airplane.cabin

        self.width = cabin.width  + 2 * self.wall_thickness
        self.height = 1.25*(cabin.width - 0.15)
        self.cabin_center = self.front_ratio * self.width + 0.5*cabin.length
        self.length = (self.front_ratio + self.rear_ratio)*self.width + cabin.length
        self.nose_cone_length = 2.00 * self.width
        self.tail_cone_length = 3.45 * self.width
        self.wet_area = 2.70*self.length*self.width
        self.aero_length = self.length

    def eval_mass(self):
        self.mass = 5.80*(np.pi*self.length*self.width)**1.2                   # Statistical regression versus fuselage built surface

    def sketch_3view(self, side=None):
        body_width = self.width
        body_height = self.height
        body_length = self.length

        nose,cone,section = self.sketch_curve(["nose1","cone1","sec1"])

        r_nose = self.nose_cone_length / self.length    # Fuselage length ratio of nose evolutive part
        r_cone = self.tail_cone_length / self.length    # Fuselage length ratio of tail cone evolutive part

        cyl_yz = np.stack([section[0:,0]*body_width , section[0:,1]*body_height+0.5*body_height , section[0:,2]*body_height+0.5*body_height], axis=1)

        body_front = np.vstack([np.stack([cyl_yz[0:,0] , cyl_yz[0:,1]],axis=1) , np.stack([cyl_yz[::-1,0] , cyl_yz[::-1,2]],axis=1)])

        nose_xz = np.stack([nose[0:,0]*body_length*r_nose , nose[0:,1]*body_height , nose[0:,2]*body_height], axis=1)
        cone_xz = np.stack([(1-r_cone)*body_length + cone[0:,0]*body_length*r_cone , cone[0:,1]*body_height , cone[0:,2]*body_height], axis=1)
        body_xz = np.vstack([nose_xz , cone_xz])

        body_side = np.vstack([np.stack([body_xz[0:-2,0] , body_xz[0:-2,1]],axis=1) , np.stack([body_xz[:0:-1,0] , body_xz[:0:-1,2]],axis=1)])

        nose_xy = np.stack([nose[0:,0]*body_length*r_nose , nose[0:,3]*body_width , nose[0:,4]*body_width], axis=1)
        cone_xy = np.stack([(1-r_cone)*body_length + cone[0:,0]*body_length*r_cone , cone[0:,3]*body_width , cone[0:,4]*body_width], axis=1)
        body_xy = np.vstack([nose_xy , cone_xy])

        body_top = np.vstack([np.stack([body_xy[1:-2,0]  , body_xy[1:-2,1]],axis=1) , np.stack([body_xy[:0:-1,0] , body_xy[:0:-1,2]],axis=1)])

        return "body", {"xy":body_top , "yz":body_front, "xz":body_side}


class Wing(Component):
    def __init__(self, airplane, area, aspect_ratio, taper_ratio, toc_ratio, sweep25, dihedral):
        super(Wing, self).__init__(airplane)

        self.area = area
        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.sweep25 = sweep25              # Sweep angle at 25% of chords
        self.dihedral = dihedral

        self.span = None
        self.root_c = None
        self.root_toc = None
        self.root_loc = None
        self.kink_c = None
        self.kink_toc = None
        self.kink_loc = None
        self.tip_c = None
        self.tip_toc = None
        self.tip_loc = None
        self.mac = None
        self.mac_loc = None
        self.mac_position = None    # X wise MAC position versus wing root
        self.position = None        # X wise wing root position versus fuselage

        self.front_spar_ratio = 0.15
        self.rear_spar_ratio = 0.70
        self.form_factor = 1.4

    def eval_geometry(self):
        fuselage = self.airplane.fuselage
        landing_gears = self.airplane.landing_gears

        self.tip_toc = 0.10
        self.kink_toc = self.tip_toc + 0.01
        self.root_toc = self.kink_toc + 0.03

        self.span = np.sqrt(self.area*self.aspect_ratio)

        y_root = 0.5*fuselage.width
        y_kink = 1.05*landing_gears.leg_length
        y_tip = 0.5*self.span

        Phi100intTE = max(0., 2. * (self.sweep25 - unit.rad_deg(32.)))
        tan_phi100 = np.tan(Phi100intTE)
        A = ((1-0.25*self.taper_ratio)*y_kink+0.25*self.taper_ratio*y_root-y_tip) / (0.75*y_kink+0.25*y_root-y_tip)
        B = (np.tan(self.sweep25)-tan_phi100) * ((y_tip-y_kink)*(y_kink-y_root)) / (0.25*y_root+0.75*y_kink-y_tip)
        self.root_c = (self.area-B*(y_tip-y_root)) / (y_root+y_kink+A*(y_tip-y_root)+self.taper_ratio*(y_tip-y_kink))
        self.kink_c = A*self.root_c + B
        self.tip_c = self.taper_ratio*self.root_c

        tan_phi0 = 0.25*(self.kink_c-self.tip_c)/(y_tip-y_kink) + np.tan(self.sweep25)

        self.mac = 2.*( 3.*y_root*self.root_c**2 \
                       +(y_kink-y_root)*(self.root_c**2+self.kink_c**2+self.root_c*self.kink_c) \
                       +(y_tip-y_kink)*(self.kink_c**2+self.tip_c**2+self.kink_c*self.tip_c) \
                      )/(3*self.area)

        y_mac = (  3.*self.root_c*y_root**2 \
                 +(y_kink-y_root)*(self.kink_c*(y_root+y_kink*2.)+self.root_c*(y_kink+y_root*2.)) \
                 +(y_tip-y_kink)*(self.tip_c*(y_kink+y_tip*2.)+self.kink_c*(y_tip+y_kink*2.)) \
                )/(3.*self.area)

        self.mac_position = ( (y_kink-y_root)*tan_phi0*((y_kink-y_root)*(self.kink_c*2.+self.root_c) \
                             +(y_tip-y_kink)*(self.kink_c*2.+self.tip_c))+(y_tip-y_root)*tan_phi0*(y_tip-y_kink)*(self.tip_c*2.+self.kink_c) \
                            )/(3*self.area)

        self.position = fuselage.cabin_center - (self.mac_position + 0.45*self.mac)    # Set wing root position

        x_root = self.position
        x_kink = x_root + (y_kink-y_root)*tan_phi0
        x_tip = x_root + (y_tip-y_root)*tan_phi0

        x_mac = x_root+( (x_kink-x_root)*((y_kink-y_root)*(self.kink_c*2.+self.root_c) \
                            +(y_tip-y_kink)*(self.kink_c*2.+self.tip_c))+(x_tip-x_root)*(y_tip-y_kink)*(self.tip_c*2.+self.kink_c) \
                           )/(self.area*3.)

        z_root = 0.
        z_kink = z_root+(y_kink-y_root)*np.tan(self.dihedral)
        z_tip = z_root+(y_tip-y_root)*np.tan(self.dihedral)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.kink_loc = np.array([x_kink, y_kink, z_kink])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, None])

        self.wet_area = 2*(self.area - fuselage.width*self.root_c)
        self.aero_length = self.mac

    def eval_mass(self):
        aerodynamics = self.airplane.aerodynamics
        mass = self.airplane.mass

        hld_conf_ld = aerodynamics.hld_conf_ld

        cz_max_ld,cz0 = aerodynamics.wing_high_lift(hld_conf_ld)

        A = 32*self.area**1.1
        B = 4.*self.span**2 * np.sqrt(mass.mtow*mass.mzfw)
        C = 1.1e-6*(1.+2.*self.aspect_ratio)/(1.+self.aspect_ratio)
        D = (0.6*self.root_toc+0.3*self.kink_toc+0.1*self.tip_toc) * (self.area/self.span)
        E = np.cos(self.sweep25)**2
        F = 1200.*max(0., cz_max_ld - 1.8)**1.5

        self.mass = (A + (B*C)/(D*E) + F)   # Shevell formula + high lift device regression

    def sketch_3view(self, side=None):
        wing_x_root = self.root_loc[0]
        wing_y_root = self.root_loc[1]
        wing_z_root = self.root_loc[2]
        wing_c_root = self.root_c
        wing_toc_r = self.root_toc
        wing_x_kink = self.kink_loc[0]
        wing_y_kink = self.kink_loc[1]
        wing_z_kink = self.kink_loc[2]
        wing_c_kink = self.kink_c
        wing_toc_k = self.kink_toc
        wing_x_tip = self.tip_loc[0]
        wing_y_tip = self.tip_loc[1]
        wing_z_tip = self.tip_loc[2]
        wing_c_tip = self.tip_c
        wing_toc_t = self.tip_toc

        wing_xy = np.array([[wing_x_root             ,  wing_y_root ],
                            [wing_x_tip              ,  wing_y_tip  ],
                            [wing_x_tip+wing_c_tip   ,  wing_y_tip  ],
                            [wing_x_kink+wing_c_kink ,  wing_y_kink ],
                            [wing_x_root+wing_c_root ,  wing_y_root ],
                            [wing_x_root+wing_c_root , -wing_y_root ],
                            [wing_x_kink+wing_c_kink , -wing_y_kink ],
                            [wing_x_tip+wing_c_tip   , -wing_y_tip  ],
                            [wing_x_tip              , -wing_y_tip  ],
                            [wing_x_root             , -wing_y_root ],
                            [wing_x_root             ,  wing_y_root ]])

        wing_yz = np.array([[ wing_y_root  , wing_z_root                        ],
                            [ wing_y_kink  , wing_z_kink                        ],
                            [ wing_y_tip   , wing_z_tip                         ],
                            [ wing_y_tip   , wing_z_tip+wing_toc_t*wing_c_tip   ],
                            [ wing_y_kink  , wing_z_kink+wing_toc_k*wing_c_kink ],
                            [ wing_y_root  , wing_z_root+wing_toc_r*wing_c_root ],
                            [-wing_y_root  , wing_z_root+wing_toc_r*wing_c_root ],
                            [-wing_y_kink  , wing_z_kink+wing_toc_k*wing_c_kink ],
                            [-wing_y_tip   , wing_z_tip+wing_toc_t*wing_c_tip   ],
                            [-wing_y_tip   , wing_z_tip                         ],
                            [-wing_y_kink  , wing_z_kink                        ],
                            [-wing_y_root  , wing_z_root                        ],
                            [ wing_y_root  , wing_z_root                        ]])

        wing_xz = np.array([[wing_x_tip                  , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                            [wing_x_tip+0.1*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip-0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip+0.7*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip-0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip+wing_c_tip       , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                            [wing_x_tip+0.7*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip+0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip+0.1*wing_c_tip   , wing_z_tip+wing_toc_t*wing_c_tip+0.5*wing_toc_t*wing_c_tip ],
                            [wing_x_tip                  , wing_z_tip+wing_toc_t*wing_c_tip                           ],
                            [wing_x_kink                 , wing_z_kink+0.5*wing_toc_k*wing_c_kink                     ],
                            [wing_x_root                 , wing_z_root+0.5*wing_toc_r*wing_c_root                     ],
                            [wing_x_root+0.1*wing_c_root , wing_z_root                                                ],
                            [wing_x_root+0.7*wing_c_root , wing_z_root                                                ],
                            [wing_x_root+wing_c_root     , wing_z_root+0.5*wing_toc_r*wing_c_root                     ],
                            [wing_x_kink+wing_c_kink     , wing_z_kink+0.5*wing_toc_k*wing_c_kink                     ],
                            [wing_x_tip+wing_c_tip       , wing_z_tip+wing_toc_t*wing_c_tip                           ]])

        return "wing", {"xy":wing_xy, "yz":wing_yz, "xz":wing_xz}


class Tank(Component):
    def __init__(self, airplane):
        super(Tank, self).__init__(airplane)

        self.fuel_volume = None
        self.cantilever_volume = None
        self.central_volume = None

    def eval_geometry(self):
        fuselage = self.airplane.fuselage
        wing = self.airplane.wing

        dsr = wing.rear_spar_ratio - wing.front_spar_ratio

        root_sec = dsr * wing.root_toc*wing.root_c**2
        kink_sec = dsr * wing.kink_toc*wing.kink_c**2
        tip_sec = dsr * wing.tip_toc*wing.tip_c**2

        self.cantilever_volume = (2./3.)*(
            0.9*(wing.kink_loc[1]-wing.root_loc[1])*(root_sec+kink_sec+np.sqrt(root_sec*kink_sec))
          + 0.7*(wing.tip_loc[1]-wing.kink_loc[1])*(kink_sec+tip_sec+np.sqrt(kink_sec*tip_sec)))

        self.central_volume = 0.8*dsr * fuselage.width * wing.root_toc * wing.root_c**2
        self.fuel_volume = self.central_volume + self.cantilever_volume

    def sketch_3view(self, side=None):
        return "tank", {}


class HTP(Component):
    def __init__(self, airplane, aspect_ratio, taper_ratio, toc_ratio, sweep25, dihedral, volume):
        super(HTP, self).__init__(airplane)

        self.area = 0.25 * airplane.wing.area
        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.toc_ratio = toc_ratio
        self.sweep25 = sweep25              # Sweep angle at 25% of chords
        self.dihedral = dihedral
        self.volume = volume

        self.span = None
        self.axe_c = None
        self.axe_loc = None
        self.tip_c = None
        self.tip_loc = None
        self.mac = None
        self.mac_position = None    # X wise MAC position versus HTP root
        self.position = None        # X wise HTP axe position versus fuselage
        self.lever_arm = None

        self.form_factor = 1.4

    def eval_geometry(self):
        fuselage = self.airplane.fuselage
        wing = self.airplane.wing

        self.span = np.sqrt(self.area*self.aspect_ratio)
        self.axe_c = (2/self.span) * (self.area / (1+self.taper_ratio))
        self.tip_c = self.taper_ratio * self.axe_c
        self.mac = (2/3) * self.axe_c * (1+self.taper_ratio-self.taper_ratio/(1+self.taper_ratio))

        chord_gradient = 2 * (self.tip_c - self.axe_c) / self.span
        tan_sweep0 = np.tan(self.sweep25) - 0.25*chord_gradient
        self.mac_position = (1/3) * (self.span/2) * ((1+2*self.taper_ratio)/(1+self.taper_ratio)) * tan_sweep0

        self.position = fuselage.length - 1.30 * self.axe_c
        self.lever_arm = (self.position + self.mac_position + 0.25*self.mac) - (wing.position + wing.mac_position + 0.25*wing.mac)

        self.wet_area = 1.63*self.area
        self.aero_length = self.mac

        y_axe = 0.
        y_tip = 0.5*self.span

        z_axe = 0.80 * fuselage.width
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        x_axe = self.position
        x_tip = x_axe + 0.25*(self.axe_c-self.tip_c) + y_tip*np.tan(self.sweep25)

        self.axe_loc = np.array([x_axe, y_axe, z_axe])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])

    def eval_area(self):
        wing = self.airplane.wing

        self.area = self.volume * wing.area * wing.mac / self.lever_arm

    def solve_area(self):
        def fct_htp(x):
            self.area = x
            self.eval_geometry()
            self.eval_area()
            return x-self.area
        xini = self.area
        output_dict = fsolve(fct_htp, x0=xini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.area = output_dict[0][0]

    def eval_mass(self):
        self.mass = 22. * self.area

    def sketch_3view(self, side=None):
        htp_span = self.span
        htp_dihedral = self.dihedral
        htp_t_o_c = self.toc_ratio
        htp_x_axe = self.axe_loc[0]
        htp_z_axe = self.axe_loc[2]
        htp_c_axe = self.axe_c
        htp_x_tip = self.tip_loc[0]
        htp_z_tip = self.tip_loc[2]
        htp_c_tip = self.tip_c

        htp_xy = np.array([[htp_x_axe           ,  0            ],
                           [htp_x_tip           ,  0.5*htp_span ],
                           [htp_x_tip+htp_c_tip ,  0.5*htp_span ],
                           [htp_x_axe+htp_c_axe ,  0            ],
                           [htp_x_tip+htp_c_tip , -0.5*htp_span ],
                           [htp_x_tip           , -0.5*htp_span ],
                           [htp_x_axe           ,  0            ]])

        htp_xz = np.array([[htp_x_tip              , htp_z_tip                          ],
                           [htp_x_tip+0.1*htp_c_tip , htp_z_tip+0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+0.7*htp_c_tip , htp_z_tip+0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+htp_c_tip     , htp_z_tip                         ],
                           [htp_x_tip+0.7*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+0.1*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip               , htp_z_tip                         ],
                           [htp_x_axe               , htp_z_axe                         ],
                           [htp_x_axe+0.1*htp_c_axe , htp_z_axe-0.5*htp_t_o_c*htp_c_axe ],
                           [htp_x_axe+0.7*htp_c_axe , htp_z_axe-0.5*htp_t_o_c*htp_c_axe ],
                           [htp_x_axe+htp_c_axe     , htp_z_axe                         ],
                           [htp_x_tip+htp_c_tip     , htp_z_tip                         ],
                           [htp_x_tip+0.7*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip+0.1*htp_c_tip , htp_z_tip-0.5*htp_t_o_c*htp_c_tip ],
                           [htp_x_tip               , htp_z_tip                         ]])

        htp_yz = np.array([[ 0           , htp_z_axe                                                        ],
                           [ 0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)                     ],
                           [ 0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)-htp_t_o_c*htp_c_tip ],
                           [ 0            , htp_z_axe-htp_t_o_c*htp_c_axe                                   ],
                           [-0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)-htp_t_o_c*htp_c_tip ],
                           [-0.5*htp_span , htp_z_axe+0.5*htp_span*np.tan(htp_dihedral)                     ],
                           [ 0            , htp_z_axe                                                       ]])

        return "htp", {"xy":htp_xy , "yz":htp_yz, "xz":htp_xz}


class VTP(Component):
    def __init__(self, airplane, aspect_ratio, taper_ratio, toc_ratio, sweep25, thrust_volume):
        super(VTP, self).__init__(airplane)

        self.area = 0.20 * airplane.wing.area
        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.toc_ratio = toc_ratio
        self.sweep25 = sweep25              # Sweep angle at 25% of chords
        self.thrust_volume = thrust_volume

        self.height = None
        self.root_c = None
        self.tip_c = None
        self.mac = None
        self.mac_position = None    # X wise MAC position versus HTP root
        self.position = None        # X wise HTP axe position versus fuselage
        self.lever_arm = None

        self.form_factor = 1.4

    def eval_geometry(self):
        fuselage = self.airplane.fuselage
        wing = self.airplane.wing
        htp = self.airplane.htp

        self.height = np.sqrt(self.area*self.aspect_ratio)
        self.root_c = (2/self.height) * (self.area / (1+self.taper_ratio))
        self.tip_c = self.taper_ratio * self.root_c
        self.mac = (2/3) * self.root_c * (1+self.taper_ratio-self.taper_ratio/(1+self.taper_ratio))

        chord_gradient = 2 * (self.tip_c - self.root_c) / self.height
        tan_sweep0 = np.tan(self.sweep25) - 0.25*chord_gradient
        self.mac_position = (1/3) * (self.height/2) * ((1+2*self.taper_ratio)/(1+self.taper_ratio)) * tan_sweep0

        self.position = htp.position - 0.35 * self.root_c
        self.lever_arm = (self.position + self.mac_position + 0.25*self.mac) - (wing.position + wing.mac_position + 0.25*wing.mac)

        self.wet_area = 2.0*self.area
        self.aero_length = self.mac

        x_root = self.position
        x_tip = x_root + 0.25*(self.root_c-self.tip_c) + self.height*np.tan(self.sweep25)

        y_root = 0.
        y_tip = 0.

        z_root = fuselage.width
        z_tip = z_root + self.height

        self.root_loc = np.array([x_root, y_root, z_root])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])

    def eval_area(self):
        nacelles = self.airplane.nacelles

        self.area = self.thrust_volume * (1.e-3*nacelles.engine_slst) * nacelles.span_position / self.lever_arm

    def solve_area(self):
        def fct_vtp(x):
            self.area = x
            self.eval_geometry()
            self.eval_area()
            return x-self.area
        xini = self.area
        output_dict = fsolve(fct_vtp, x0=xini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.area = output_dict[0][0]

    def eval_mass(self):
        self.mass = 25. * self.area

    def sketch_3view(self, side=None):
        vtp_t_o_c = self.toc_ratio
        vtp_x_root = self.root_loc[0]
        vtp_y_root = self.root_loc[1]
        vtp_z_root = self.root_loc[2]
        vtp_c_root = self.root_c
        vtp_x_tip = self.tip_loc[0]
        vtp_y_tip = self.tip_loc[1]
        vtp_z_tip = self.tip_loc[2]
        vtp_c_tip = self.tip_c

        vtp_xz = np.array([[vtp_x_root            , vtp_z_root ],
                           [vtp_x_tip             , vtp_z_tip  ],
                           [vtp_x_tip+vtp_c_tip   , vtp_z_tip  ],
                           [vtp_x_root+vtp_c_root , vtp_z_root ],
                           [vtp_x_root            , vtp_z_root ]])

        vtp_xy = np.array([[vtp_x_root                , vtp_y_root                            ],
                           [vtp_x_root+0.1*vtp_c_root , vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root+0.7*vtp_c_root , vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root+vtp_c_root     , vtp_y_root                            ],
                           [vtp_x_root+0.7*vtp_c_root , vtp_y_root - 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root+0.1*vtp_c_root , vtp_y_root - 0.5*vtp_t_o_c*vtp_c_root ],
                           [vtp_x_root                , vtp_y_root                            ],
                           [vtp_x_tip                 , vtp_y_tip                             ],
                           [vtp_x_tip+0.1*vtp_c_tip   , vtp_y_tip + 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip+0.7*vtp_c_tip   , vtp_y_tip + 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip+vtp_c_tip       , vtp_y_tip                             ],
                           [vtp_x_tip+0.7*vtp_c_tip   , vtp_y_tip - 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip+0.1*vtp_c_tip   , vtp_y_tip - 0.5*vtp_t_o_c*vtp_c_tip   ],
                           [vtp_x_tip                 , vtp_y_tip                             ]])

        vtp_yz = np.array([[vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root , vtp_z_root ],
                           [vtp_y_tip + 0.5*vtp_t_o_c*vtp_c_tip   , vtp_z_tip  ],
                           [vtp_y_tip - 0.5*vtp_t_o_c*vtp_c_tip   , vtp_z_tip  ],
                           [vtp_y_root - 0.5*vtp_t_o_c*vtp_c_root , vtp_z_root ],
                           [vtp_y_root + 0.5*vtp_t_o_c*vtp_c_root , vtp_z_root ]])

        return "vtp", {"xy":vtp_xy , "yz":vtp_yz, "xz":vtp_xz}


class Nacelles(Component):
    def __init__(self, airplane, engine_slst, engine_bpr, z_ratio):
        super(Nacelles, self).__init__(airplane)

        self.engine_slst = engine_slst
        self.engine_bpr = engine_bpr

        self.diameter = None
        self.length = None
        self.engine_loc = None
        self.span_position = 2.3 + 0.5*engine_bpr**0.7 + 5.E-6*engine_slst
        self.z_ratio = z_ratio
        self.ground_clearence = None      # INFO: ground_clearence must be higher or equal to 1 m

        self.pylon_mass = None
        self.engine_mass = None

        self.form_factor = 1.15

    def eval_geometry(self):
        wing = self.airplane.wing
        fuselage = self.airplane.fuselage
        landing_gears = self.airplane.landing_gears

        self.diameter = 0.5*self.engine_bpr**0.7 + 5.E-6*self.engine_slst
        self.length = 0.86*self.diameter + self.engine_bpr**0.37      # statistical regression
        self.span_position = 0.6 * fuselage.width + 1.5 * self.diameter

        # INFO: Ground clearence constraint can be realeased by changing wing dihedral and or nacelle span position
        self.ground_clearence = landing_gears.leg_length - landing_gears.attachment_loc[2] + self.span_position*np.tan(wing.dihedral) - (self.z_ratio+0.5)*self.diameter

        knac = np.pi * self.diameter * self.length
        self.wet_area = knac*(1.48 - 0.0076*knac)       # statistical regression, all engines

        self.aero_length = self.length
        self.form_factor = 1.15

        tan_phi0 = 0.25*(wing.kink_c-wing.tip_c)/(wing.tip_loc[1]-wing.kink_loc[1]) + np.tan(wing.sweep25)

        y_int = self.span_position
        x_int = wing.root_loc[0] + (y_int-wing.root_loc[1])*tan_phi0 - 0.7*self.length
        z_int = wing.root_loc[2] + (y_int-wing.root_loc[2])*np.tan(wing.dihedral) - self.z_ratio*self.diameter

        self.engine_loc = np.array([x_int, y_int, z_int])

    def eval_mass(self):
        propulsion = self.airplane.propulsion

        self.engine_mass = (1250. + 0.021*self.engine_slst)
        self.pylon_mass = (0.0031*self.engine_slst)
        self.mass = (self.engine_mass + self.pylon_mass) * propulsion.n_engine

    def sketch_3view(self, side=1):
        nac_length = self.length
        nac_height = self.diameter
        nac_width =  self.diameter
        nac_x = self.engine_loc[0]
        nac_y = self.engine_loc[1] * side
        nac_z = self.engine_loc[2]

        section, = self.sketch_curve(["sec1"])

        nac_xz = np.array([[nac_x                , nac_z+0.4*nac_height ] ,
                           [nac_x+0.1*nac_length , nac_z+0.5*nac_height ] ,
                           [nac_x+0.5*nac_length , nac_z+0.5*nac_height ] ,
                           [nac_x+nac_length     , nac_z+0.3*nac_height ] ,
                           [nac_x+nac_length     , nac_z-0.3*nac_height ] ,
                           [nac_x+0.5*nac_length , nac_z-0.5*nac_height ] ,
                           [nac_x+0.1*nac_length , nac_z-0.5*nac_height ] ,
                           [nac_x                , nac_z-0.4*nac_height ] ,
                           [nac_x                , nac_z+0.4*nac_height ]])

        nac_xy = np.array([[nac_x                , nac_y+0.4*nac_width ] ,
                           [nac_x+0.1*nac_length , nac_y+0.5*nac_width ] ,
                           [nac_x+0.5*nac_length , nac_y+0.5*nac_width ] ,
                           [nac_x+nac_length     , nac_y+0.3*nac_width ] ,
                           [nac_x+nac_length     , nac_y-0.3*nac_width ] ,
                           [nac_x+0.5*nac_length , nac_y-0.5*nac_width ] ,
                           [nac_x+0.1*nac_length , nac_y-0.5*nac_width ] ,
                           [nac_x                , nac_y-0.4*nac_width ] ,
                           [nac_x                , nac_y+0.4*nac_width ]])

        d_nac_yz = np.stack([section[0:,0]*nac_width , section[0:,1]*nac_height , section[0:,2]*nac_height], axis=1)

        d_fan_yz = np.stack([section[0:,0]*0.80*nac_width , section[0:,1]*0.80*nac_height , section[0:,2]*0.80*nac_height], axis=1)

        nac_yz = np.vstack([np.stack([nac_y+d_nac_yz[0:,0] , nac_z+d_nac_yz[0:,1]],axis=1) ,
                               np.stack([nac_y+d_nac_yz[::-1,0] , nac_z+d_nac_yz[::-1,2]],axis=1)])

        disk_yz = np.vstack([np.stack([nac_y+d_fan_yz[0:,0] , nac_z+d_fan_yz[0:,1]],axis=1) ,
                             np.stack([nac_y+d_fan_yz[::-1,0] , nac_z+d_fan_yz[::-1,2]],axis=1)])

        return "wing_nacelle", {"xy":nac_xy , "yz":nac_yz, "xz":nac_xz, "disk":disk_yz}


class LandingGears(Component):
    def __init__(self, airplane, leg_length):
        super(LandingGears, self).__init__(airplane)

        self.leg_length = leg_length
        self.attachment_loc = [0.,leg_length,0.]

    def eval_geometry(self):
        wing = self.airplane.wing

        y_loc = 1.01 * self.leg_length
        r = (y_loc-wing.root_loc[1])/(wing.kink_loc[1]-wing.root_loc[1])
        z_loc = wing.root_loc[2]*(1-r) + wing.kink_loc[2]*r
        chord = wing.root_c*(1-r) + wing.kink_c*r
        x_chord = wing.root_loc[0]*(1-r) + wing.kink_loc[0]*r
        x_loc = x_chord + chord - 1.02*wing.rear_spar_ratio*wing.kink_c
        self.attachment_loc = [x_loc, y_loc, z_loc]

    def eval_mass(self):
        mass = self.airplane.mass

        self.mass = (0.015*mass.mtow**1.03 + 0.012*mass.mlw)

    def sketch_3view(self, side=None):
        return "ldg", {}


class Systems(Component):
    def __init__(self, airplane):
        super(Systems, self).__init__(airplane)

    def eval_geometry(self):
        pass

    def eval_mass(self):
        mass = self.airplane.mass

        self.mass = 0.545*mass.mtow**0.8    # global mass of all systems

    def sketch_3view(self, side=None):
        return "sys", {}

