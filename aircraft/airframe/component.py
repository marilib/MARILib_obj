#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
import unit
import earth


class Component(object):
    """Define common features for all airplane components.

    Every component of the :class:'Airframe' inherits the basic features written in :class:'Component'

    **Attributs**
        * aircraft : the aircraft to which the component belongs.
        Needed for some pre-design methods (call to requirements) or multi-components interaction.
        * frame_origin : [x,y,z] origin of the *local* coordinate system inside the global aircraft coordinate system.
        * frame_angles : [psi,theta,phi] Euler angles to describe rotation of the local coordinate system.
        * mass : the net mass of the component
        * cg : [xg,yg,zg] the position of center of mass in **local** coordinates.
        * inertia_tensor : the inertia tensor of the component
        * gross_wet_area : wetted area of the component alone
        * net_wet_area : wetted area of the component in the assembly (without footprints)
        * aero_length : characteristic length of the component in the direction of the flow. Used for Reynolds number.
        * form_factor : factor on skin friction to account for lift independent pressure drag

    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.frame_origin = np.full(3,None)

        self.mass = None
        self.cg = np.full(3,None)
        self.inertia_tensor = np.full((3,3),None)

        self.gross_wet_area = 0.    # wetted area of the component alone
        self.net_wet_area = 0.      # wetted area of the component in the assembly (without footprints)
        self.aero_length = 0.       # characteristic length of the component in the direction of the flow
        self.form_factor = 0.       # factor on skin friction to account for lift independent pressure drag

    def get_mass_mwe(self):
        raise self.mass

    def get_mass_owe(self):
        return self.mass

    def get_cg_mwe(self):
        raise self.cg

    def get_cg_owe(self):
        raise self.cg

    def get_inertia_tensor(self):
        return self.inertia_tensor

    def get_net_wet_area(self):
        return self.net_wet_area

    def get_aero_length(self):
        return self.aero_length

    def get_form_factor(self):
        return self.form_factor


class Cabin(Component):

    def __init__(self, aircraft):

        super(Cabin, self).__init__(aircraft)

        self.width = None
        self.length = None
        self.co2_metric_area = None

        self.m_furnishing = None
        self.m_op_item = None

        self.cg_furnishing = None
        self.cg_op_item = None

    def eval_geometry(self):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.requirement.n_pax_front
        n_aisle = self.aircraft.requirement.n_aisle

        self.width = 0.38*n_pax_front + 1.05*n_aisle + 0.15     # Statistical regression
        self.length = 6.3*(self.width - 0.24) + 0.005*(n_pax_ref/n_pax_front)**2.25     # Statistical regression

        self.projected_area = 0.95*self.length*self.width       # Factor 0.95 accounts for tapered parts

    def eval_mass(self):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        design_range = self.aircraft.requirement.design_range

        self.m_furnishing = (0.063*n_pax_ref**2 + 9.76*n_pax_ref)       # Furnishings mass
        self.m_op_item = 5.2*(n_pax_ref*design_range*1e-6)          # Operator items mass

        x_cg_furnishing = self.frame_origin[0] + 0.55*self.length      # Rear cabin is heavier because of higher density
        x_cg_op_item = x_cg_furnishing    # Operator items cg

        self.cg_furnishing = np.array([x_cg_furnishing, 0., 0.])
        self.cg_op_item = np.array([x_cg_op_item, 0., 0.])

        self.mass = self.m_furnishing + self.m_op_item
        self.cg = (self.cg_furnishing*self.m_furnishing + self.cg_op_item*self.m_op_item) / self.mass

    def get_mass_mwe(self):
        return self.m_furnishing

    def get_cg_mwe(self):
        return self.cg_furnishing


class Fuselage(Component):

    def __init__(self, aircraft):

        super(Fuselage, self).__init__(aircraft)

        self.width = None
        self.height = None
        self.length = None
        self.tail_cone_length = None

    def eval_geometry(self):
        self.frame_origin = [0., 0., 0.]

        cabin_width = self.aircraft.airframe.cabin.width
        cabin_length = self.aircraft.airframe.cabin.length

        fwd_limit = 4.      # Cabin starts 4 meters behind fuselage nose

        self.aircraft.airframe.cabin.frame_origin = [fwd_limit, 0., 0.]     # cabin position inside the fuselage
        self.aircraft.airframe.cabin.frame_angles = [0., 0., 0.]            # cabin orientation inside the fuselage

        self.width = cabin_width + 0.4      # fuselage walls are supposed 0.2m thick
        self.height = 1.25*(cabin_width - 0.15)
        self.length = fwd_limit + cabin_length + 1.50*self.width
        self.tail_cone_length = 3.45*self.width

        self.gross_wet_area = 2.70*self.length*np.sqrt(self.width*self.height)
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.length
        self.form_factor = 1.05

    def eval_mass(self):
        kfus = np.pi*self.length*np.sqrt(self.width*self.height)
        self.mass = 5.47*kfus**1.2      # Statistical regression versus fuselage built surface
        self.cg = np.array([0.50*self.length, 0., 0.40*self.height])     # Middle of the fuselage


class Wing(Component):

    def __init__(self, aircraft):

        super(Wing, self).__init__(aircraft)

        design_range = self.aircraft.requirement.design_range
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.requirement.n_pax_front
        n_aisle = self.aircraft.requirement.n_aisle

        self.morphing = "aspect_ratio_driven"   # "aspect_ratio_driven" or "span_driven"
        self.area = 60. + 88.*n_pax_ref*design_range*1.e-9
        self.span = None
        self.aspect_ratio = 9.          # Default value
        self.taper_ratio = 0.25
        self.sweep0 = None
        self.sweep25 = None
        self.sweep100 = None
        self.dihedral = None
        self.setting = None
        self.hld_type = 9

        self.loc_root = np.full(3,None)     # Position of root chord leading edge
        self.toc_root = None                # thickness over chord ratio of root chord
        self.c_root = None                  # root chord length

        x_kink = 1.2*(0.38*n_pax_front + 1.05*n_aisle + 0.55)

        self.loc_kink = np.array([x_kink, None, None])     # Position of kink chord leading edge
        self.toc_kink = None                # thickness over chord ratio of kink chord
        self.c_kink = None                  # kink chord length

        self.loc_tip = np.full(3,None)      # Position of tip chord leading edge
        self.toc_tip = None                 # thickness over chord ratio of tip chord
        self.c_tip = None                   # tip chord length

        self.loc_mac = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        wing_attachment = self.aircraft.arrangement.wing_attachment
        cruise_mach = self.aircraft.requirement.cruise_mach
        body_width = self.aircraft.airframe.body.width
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height

        self.toc_tip = 0.10
        self.toc_kink = self.toc_tip + 0.01
        self.toc_root = self.toc_kink + 0.03

        self.sweep25 = 1.6*max(0.,(cruise_mach - 0.5))     # Empirical law

        self.dihedral = unit.rad_deg(5.)

        if(self.morphing=="aspect_ratio_driven"):   # Aspect ratio is driving parameter
            self.span = np.sqrt(self.aspect_ratio*self.area)
        elif(self.morphing=="span_driven"): # Span is driving parameter
            self.aspect_ratio = self.span**2/self.area
        else:
            print("geometry_predesign_, wing_morphing index is unkown")

        y_root = 0.5*body_width
        y_kink = self.loc_kink[0]
        y_tip = 0.5*self.span

        if(15<unit.deg_rad(self.sweep25)):  # With kink
          Phi100intTE = max( 0. , 2.*(self.sweep25-unit.rad_deg(32.)) )
          tan_phi100 = np.tan(Phi100intTE)
          A = ((1-0.25*self.taper_ratio)*y_kink+0.25*self.taper_ratio*y_root-y_tip) / (0.75*y_kink+0.25*y_root-y_tip)
          B = (np.tan(self.sweep25)-tan_phi100) * ((y_tip-y_kink)*(y_kink-y_root)) / (0.25*y_root+0.75*y_kink-y_tip)
          self.c_root = (self.area-B*(y_tip-y_root)) / (y_root+y_kink+A*(y_tip-y_root)+self.taper_ratio*(y_tip-y_kink))
          self.c_kink = A*self.c_root + B
          self.c_tip = self.taper_ratio*self.c_root

        else:		# Without kink
          self.c_root = 2.*self.area / (2.*y_root*(1.-self.taper_ratio) + (1.+self.taper_ratio)*np.sqrt(self.aspect_ratio*self.area))
          self.c_tip = self.taper_ratio*self.c_root
          self.c_kink = ((y_tip-y_kink)*self.c_root + (y_kink-y_root)*self.c_tip) / (y_tip-y_root)


        tan_phi0 = 0.25*(self.c_kink-self.c_tip)/(y_tip-y_kink) + np.tan(self.sweep25)

        self.mac = 2.*( 3.*y_root*self.c_root**2 \
                       +(y_kink-y_root)*(self.c_root**2+self.c_kink**2+self.c_root*self.c_kink) \
                       +(y_tip-y_kink)*(self.c_kink**2+self.c_tip**2+self.c_kink*self.c_tip) \
                      )/(3*self.area)

        y_mac = (  3.*self.c_root*y_root**2 \
                 +(y_kink-y_root)*(self.c_kink*(y_root+y_kink*2.)+self.c_root*(y_kink+y_root*2.)) \
                 +(y_tip-y_kink)*(self.c_tip*(y_kink+y_tip*2.)+self.c_kink*(y_tip+y_kink*2.)) \
                )/(3.*self.area)

        x_mac_local = ( (y_kink-y_root)*tan_phi0*((y_kink-y_root)*(self.c_kink*2.+self.c_root) \
                       +(y_tip-y_kink)*(self.c_kink*2.+self.c_tip))+(y_tip-y_root)*tan_phi0*(y_tip-y_kink)*(self.c_tip*2.+self.c_kink) \
                      )/(3*self.area)

        x_root = 0.33*body_length**1.1 - (x_mac_local + 0.25*self.mac)
        x_kink = x_root + (y_kink-y_root)*tan_phi0
        x_tip = x_root + (y_tip-y_root)*tan_phi0

        x_mac = x_root+( (x_kink-x_root)*((y_kink-y_root)*(self.c_kink*2.+self.c_root) \
                            +(y_tip-y_kink)*(self.c_kink*2.+self.c_tip))+(x_tip-x_root)*(y_tip-y_kink)*(self.c_tip*2.+self.c_kink) \
                           )/(self.area*3.)
        if (wing_attachment=="low"):
            z_root = 0.
        else:
            z_root = body_height - 0.5*self.toc_root*self.c_root

        z_kink = z_root+(y_kink-y_root)*np.tan(self.dihedral)
        z_tip = z_root+(y_tip-y_root)*np.tan(self.dihedral)

        self.loc_root = np.array([x_root, y_root, z_root])
        self.loc_kink = np.array([x_kink, y_kink, z_kink])
        self.loc_tip = np.array([x_tip, y_tip, z_tip])
        self.loc_mac = np.array([x_mac, y_mac, None])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.00*(self.area - self.c_root*body_width)
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

        # Wing setting
        #-----------------------------------------------------------------------------------------------------------
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()

        disa = 0.
        rca = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        mass = 0.95*self.aircraft.weight_cg.mtow

        pamb,tamb,tstd,dtodz = earth.atmosphere(rca,disa)

        cza_wing = self.cza(mach, body_width, self.aspect_ratio, self.span, self.sweep25)

        # AoA = 2.5° at cruise start
        self.setting = (0.97*mass*g)/(0.5*gam*pamb*mach**2*self.area*cza_wing) - unit.rad_deg(2.5)

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        mzfw = self.aircraft.weight_cg.mzfw
        hld_conf_ld = self.aircraft.aerodynamics.hld_conf_ld

        (cz_max_ld,cz0) = self.high_lift(hld_conf_ld)

        A = 32*self.area**1.1
        B = 4.*self.span**2 * np.sqrt(mtow*mzfw)
        C = 1.1e-6*(1.+2.*self.aspect_ratio)/(1.+self.aspect_ratio)
        D = (0.6*self.toc_root+0.3*self.toc_kink+0.1*self.toc_tip) * (self.area/self.span)
        E = np.cos(self.sweep25)**2
        F = 1200.*(cz_max_ld - 1.8)**1.5

        self.mass = A + (B*C)/(D*E) + F   # Shevell formula + high lift device regression

        self.cg =  0.25*(self.loc_root + 0.40*np.array([self.c_root, 0., 0.])) \
                 + 0.55*(self.loc_kink + 0.40*np.array([self.c_kink, 0., 0.])) \
                 + 0.20*(self.loc_tip + 0.40*np.array([self.c_tip, 0., 0.]))

    def  cza(self, mach, body_width, aspect_ratio, span, sweep):
        """
        Polhamus formula
        """
        cza =  (np.pi*aspect_ratio*(1.07*(1+body_width/span)**2)*(1.-body_width/span)) \
             / (1+np.sqrt(1.+0.25*aspect_ratio**2*(1+np.tan(sweep)**2-mach**2)))
        return cza

    def high_lift(self, hld_conf):
        """
        0 =< hld_type =< 10
        0 =< hld_conf =< 1
        Typically : hld_conf = 1 ==> cz_max_ld
                  : hld_conf = 0.1 to 0.5 ==> cz_max_to
        """

        # Maximum lift coefficients of different airfoils, DUBS 1987
        cz_max_ld = {0 : 1.45 ,  # Clean
                     1 : 2.25 ,  # Flap only, Rotation without slot
                     2 : 2.60 ,  # Flap only, Rotation single slot      (ATR)
                     3 : 2.80 ,  # Flap only, Rotation double slot
                     4 : 2.80 ,  # Fowler Flap
                     5 : 2.00 ,  # Slat only
                     6 : 2.45 ,  # Slat + Flap rotation without slot
                     7 : 2.70 ,  # Slat + Flap rotation single slot
                     8 : 2.90 ,  # Slat + Flap rotation double slot
                     9 : 3.00 ,  # Slat + Fowler                      (A320)
                     10 : 3.20,  # Slat + Fowler + Fowler double slot (A321)
                     }.get(self.hld_type, "Erreur - high_lift_, HLDtype out of range")    # 9 is default if x not found

        if (self.hld_type<5):
            cz_max_base = 1.45      # Flap only
        else:
            if (hld_conf==0):
                cz_max_base = 1.45  # Clean
            else:
                cz_max_base = 2.00  # Slat + Flap

        cz_max = (1-hld_conf)*cz_max_base + hld_conf*cz_max_ld
        cz_0 = cz_max - cz_max_base  # Assumed the Lift vs AoA is just translated upward and Cz0 clean equal to zero
        return cz_max, cz_0


class VTP_classic(Component):

    def __init__(self, aircraft):

        super(VTP_classic, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Coupling variable
        self.height = None
        self.aspect_ratio = 1.7     # Design rule
        self.taper_ratio = 0.40     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.volume = 0.4           # Design rule
        self.x_anchor = None
        self.lever_arm = None

        self.loc_root = np.full(3,None)     # Position of root chord leading edge
        self.c_root = None                  # root chord length

        self.loc_tip = np.full(3,None)      # Position of tip chord leading edge
        self.c_tip = None                   # tip chord length

        self.loc_mac = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        tail_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_sweep25 = self.aircraft.airframe.wing.sweep25

        self.height = np.sqrt(self.aspect_ratio*self.area)
        self.c_root = 2*self.area/(self.height*(1+self.taper_ratio))
        self.c_tip = self.taper_ratio*self.c_root

        self.sweep25 = max(unit.rad_deg(25.), wing_sweep25 + unit.rad_deg(10.)) # Empirical law

        self.x_anchor = 0.85       # Locate self versus end body length
        x_root = body_length*(1-tail_cone_length/body_length*(1-self.x_anchor)) - self.c_root
        x_tip = x_root + 0.25*(self.c_root-self.c_tip) + self.height*np.tan(self.sweep25)

        y_root = 0.
        y_tip = 0.

        z_root = body_height
        z_tip = z_root + self.height

        self.mac = self.height*(self.c_root**2+self.c_tip**2+self.c_root*self.c_tip)/(3*self.area)
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.c_tip+self.c_root)/(6*self.area)
        y_mac = 0.
        z_mac = z_tip**2*(2*self.c_tip+self.c_root)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (x_mac + 0.25*self.mac)

        self.loc_root = np.array([x_root, y_root, z_root])
        self.loc_tip = np.array([x_tip, y_tip, z_tip])
        self.loc_mac = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 25. * self.area
        self.cg = self.loc_mac + 0.20*np.array([self.mac, 0., 0.])


class VTP_T(Component):

    def __init__(self, aircraft):

        super(VTP_T, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Coupling variable
        self.height = None
        self.aspect_ratio = 1.2     # Design rule
        self.taper_ratio = 0.80     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.volume = 0.4           # Design rule
        self.x_anchor = None
        self.lever_arm = None

        self.loc_root = np.full(3,None)     # Position of root chord leading edge
        self.c_root = None                  # root chord length

        self.loc_tip = np.full(3,None)      # Position of tip chord leading edge
        self.c_tip = None                   # tip chord length

        self.loc_mac = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        tail_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_sweep25 = self.aircraft.airframe.wing.sweep25

        self.height = np.sqrt(self.aspect_ratio*self.area)
        self.c_root = 2*self.area/(self.height*(1+self.taper_ratio))
        self.c_tip = self.taper_ratio*self.c_root

        self.sweep25 = max(unit.rad_deg(25.), wing_sweep25 + unit.rad_deg(10.)) # Empirical law

        self.x_anchor = 0.85       # Locate self versus end body length
        x_root = body_length*(1-tail_cone_length/body_length*(1-self.x_anchor)) - self.c_root
        x_tip = x_root + 0.25*(self.c_root-self.c_tip) + self.height*np.tan(self.sweep25)

        y_root = 0.
        y_tip = 0.

        z_root = body_height
        z_tip = z_root + self.height

        self.mac = self.height*(self.c_root**2+self.c_tip**2+self.c_root*self.c_tip)/(3*self.area)
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.c_tip+self.c_root)/(6*self.area)
        y_mac = 0.
        z_mac = z_tip**2*(2*self.c_tip+self.c_root)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (x_mac + 0.25*self.mac)

        self.loc_root = np.array([x_root, y_root, z_root])
        self.loc_tip = np.array([x_tip, y_tip, z_tip])
        self.loc_mac = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 28. * self.area
        self.cg = self.loc_mac + 0.20*np.array([self.mac, 0., 0.])


class VTP_H(Component):

    def __init__(self, aircraft):

        super(VTP_H, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Coupling variable
        self.height = None
        self.aspect_ratio = 1.5     # Design rule
        self.taper_ratio = 0.40     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.volume = 0.4           # Design rule
        self.lever_arm = None

        self.loc_root = np.full(3,None)     # Position of root chord leading edge
        self.c_root = None                  # root chord length

        self.loc_tip = np.full(3,None)      # Position of tip chord leading edge
        self.c_tip = None                   # tip chord length

        self.loc_mac = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        htp_loc_tip = self.aircraft.airframe.horizontal_stab.loc_tip
        wing_sweep25 = self.aircraft.airframe.wing.sweep25

        self.height = np.sqrt(self.aspect_ratio*(0.5*self.area))
        self.c_root = 2*(0.5*self.area)/(self.height*(1+self.taper_ratio))
        self.c_tip = self.taper_ratio*self.c_root

        self.sweep25 = max(unit.rad_deg(25.), wing_sweep25 + unit.rad_deg(10.)) # Empirical law

        x_root = htp_loc_tip[0]
        x_tip = x_root + 0.25*(self.c_root-self.c_tip) + self.height*np.tan(self.sweep25)

        y_root = htp_loc_tip[1]
        y_tip = htp_loc_tip[1]

        z_root = htp_loc_tip[2]
        z_tip = z_root + self.height

        self.mac = self.height*(self.c_root**2+self.c_tip**2+self.c_root*self.c_tip)/(3*(0.5*self.area))
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.c_tip+self.c_root)/(6*(0.5*self.area))
        y_mac = y_tip
        z_mac = z_tip**2*(2*self.c_tip+self.c_root)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (x_mac + 0.25*self.mac)

        self.loc_root = np.array([x_root, y_root, z_root])
        self.loc_tip = np.array([x_tip, y_tip, z_tip])
        self.loc_mac = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, y_root, z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 25. * self.area
        self.cg = self.loc_mac + 0.20*np.array([self.mac, 0., 0.])


class HTP_classic(Component):

    def __init__(self, aircraft):

        super(HTP_classic, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Coupling variable
        self.span = None
        self.aspect_ratio = 5.0     # Design rule
        self.taper_ratio = 0.35     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.dihedral = unit.rad_deg(5)     # HTP dihedral
        self.volume = 0.94                  # Design rule
        self.lever_arm = None

        self.loc_root = np.full(3,None)     # Position of root chord leading edge
        self.c_root = None                  # root chord length

        self.loc_tip = np.full(3,None)      # Position of tip chord leading edge
        self.c_tip = None                   # tip chord length

        self.loc_mac = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_height = self.aircraft.airframe.body.height
        vtp_loc_root = self.aircraft.airframe.vertical_stab.loc_root
        vtp_c_root = self.aircraft.airframe.vertical_stab.c_root
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_loc_mac = self.aircraft.airframe.wing.loc_mac
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = htp_z_wise_anchor*body_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.c_axe = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.c_tip = self.taper_ratio*self.c_axe

        self.sweep25 = wing_sweep25 + unit.rad_deg(5)     # Design rule

        self.mac = self.span*(self.c_axe**2+self.c_tip**2+self.c_axe*self.c_tip)/(3.*self.area)
        y_mac = y_tip**2*(2*self.c_tip+self.c_axe)/(3*self.area)
        z_mac = z_tip**2*(2*self.c_tip+self.c_axe)/(3*self.area)
        x_tip_local = 0.25*(self.c_axe-self.c_tip) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.c_tip*2.+self.c_axe)/(3.*self.area)

        x_axe = vtp_loc_root[0] + 0.50*vtp_c_root - 0.2*self.c_axe

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_loc_mac[0] + 0.25*wing_mac)

        self.loc_axe = np.array([x_axe, y_axe, z_axe])
        self.loc_tip = np.array([x_tip, y_tip, z_tip])
        self.loc_mac = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.loc_axe

        self.gross_wet_area = 1.63*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 22. * self.area
        self.cg = self.loc_mac + 0.20*np.array([self.mac, 0., 0.])


class HTP_T(Component):

    def __init__(self, aircraft):

        super(HTP_T, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Coupling variable
        self.span = None
        self.aspect_ratio = 5.0     # Design rule
        self.taper_ratio = 0.35     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.dihedral = unit.rad_deg(5)     # HTP dihedral
        self.volume = 0.94                  # Design rule
        self.lever_arm = None

        self.loc_root = np.full(3,None)     # Position of root chord leading edge
        self.c_root = None                  # root chord length

        self.loc_tip = np.full(3,None)      # Position of tip chord leading edge
        self.c_tip = None                   # tip chord length

        self.loc_mac = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_height = self.aircraft.airframe.body.height
        vtp_loc_tip = self.aircraft.airframe.vertical_stab.loc_tip
        vtp_c_tip = self.aircraft.airframe.vertical_stab.c_tip
        vtp_height = self.aircraft.airframe.vertical_stab.height
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_loc_mac = self.aircraft.airframe.wing.loc_mac
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = body_height + vtp_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.c_axe = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.c_tip = self.taper_ratio*self.c_axe

        self.sweep25 = wing_sweep25 + unit.rad_deg(5)     # Design rule

        self.mac = self.span*(self.c_axe**2+self.c_tip**2+self.c_axe*self.c_tip)/(3.*self.area)
        y_mac = y_tip**2*(2*self.c_tip+self.c_axe)/(3*self.area)
        z_mac = z_tip**2*(2*self.c_tip+self.c_axe)/(3*self.area)
        x_tip_local = 0.25*(self.c_axe-self.c_tip) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.c_tip*2.+self.c_axe)/(3.*self.area)

        x_axe = vtp_loc_tip[0] + 0.30*vtp_c_tip - 0.80*self.c_tip

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_loc_mac[0] + 0.25*wing_mac)

        self.loc_axe = np.array([x_axe, y_axe, z_axe])
        self.loc_tip = np.array([x_tip, y_tip, z_tip])
        self.loc_mac = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.loc_axe

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 22. * self.area
        self.cg = self.loc_mac + 0.20*np.array([self.mac, 0., 0.])


class HTP_H(Component):

    def __init__(self, aircraft):

        super(HTP_H, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Coupling variable
        self.span = None
        self.aspect_ratio = 5.0     # Design rule
        self.taper_ratio = 0.45     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.dihedral = unit.rad_deg(5)     # HTP dihedral
        self.volume = 0.94                  # Design rule
        self.lever_arm = None

        self.loc_root = np.full(3,None)     # Position of root chord leading edge
        self.c_root = None                  # root chord length

        self.loc_tip = np.full(3,None)      # Position of tip chord leading edge
        self.c_tip = None                   # tip chord length

        self.loc_mac = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        body_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_loc_mac = self.aircraft.airframe.wing.loc_mac
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = htp_z_wise_anchor*body_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.c_axe = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.c_tip = self.taper_ratio*self.c_axe

        self.sweep25 = wing_sweep25 + unit.rad_deg(5)     # Design rule

        self.mac = self.span*(self.c_axe**2+self.c_tip**2+self.c_axe*self.c_tip)/(3.*self.area)
        y_mac = y_tip**2*(2*self.c_tip+self.c_axe)/(3*self.area)
        z_mac = z_tip**2*(2*self.c_tip+self.c_axe)/(3*self.area)
        x_tip_local = 0.25*(self.c_axe-self.c_tip) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.c_tip*2.+self.c_axe)/(3.*self.area)

        htp_x_wise_anchor = 0.85
        x_axe = body_length*(1-body_cone_length/body_length*(1-htp_x_wise_anchor)) - self.c_axe

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_loc_mac[0] + 0.25*wing_mac)

        self.loc_axe = np.array([x_axe, y_axe, z_axe])
        self.loc_tip = np.array([x_tip, y_tip, z_tip])
        self.loc_mac = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.loc_axe

        self.gross_wet_area = 1.63*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 22. * self.area
        self.cg = self.loc_mac + 0.20*np.array([self.mac, 0., 0.])


class Tank_wing_box(Component):

    def __init__(self, aircraft):

        super(Tank_wing_box, self).__init__(aircraft)

        self.structure_ratio = 0.
        self.fuel_density = None

        self.cantilever_volume = None
        self.central_volume = None
        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = None
        self.fuel_max_fwd_mass = None
        self.fuel_max_bwd_cg = None
        self.fuel_max_bwd_mass = None

    def eval_geometry(self):
        body_width = self.aircraft.airframe.body.width
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac
        wing_toc_root = self.aircraft.airframe.wing.toc_root
        wing_loc_root = self.aircraft.airframe.wing.loc_root
        wing_toc_kink = self.aircraft.airframe.wing.toc_kink
        wing_toc_tip = self.aircraft.airframe.wing.toc_tip

        self.cantilever_volume =   0.275 \
                                 * (wing_area*wing_mac*(0.50*wing_toc_root + 0.30*wing_toc_kink + 0.20*wing_toc_tip)) \
                                 * (1. - self.structure_ratio)

        self.central_volume =   1.3 \
                              * body_width * wing_toc_root * wing_mac**2 \
                              * (1. - self.structure_ratio)

        self.max_volume = self.central_volume + self.cantilever_volume

        self.frame_origin = [wing_loc_root[0], 0., wing_loc_root[2]]

    def eval_mass(self):
        energy_source = self.aircraft.arrangement.energy_source
        wing_c_root = self.aircraft.airframe.wing.c_root
        wing_loc_root = self.aircraft.airframe.wing.loc_root
        wing_c_kink = self.aircraft.airframe.wing.c_kink
        wing_loc_kink = self.aircraft.airframe.wing.loc_kink
        wing_c_tip = self.aircraft.airframe.wing.c_tip
        wing_loc_tip = self.aircraft.airframe.wing.loc_tip

        self.fuel_cantilever_cg =  0.25*(wing_loc_root + 0.40*np.array([wing_c_root, 0., 0.])) \
                                  + 0.65*(wing_loc_kink + 0.40*np.array([wing_c_kink, 0., 0.])) \
                                  + 0.10*(wing_loc_tip + 0.40*np.array([wing_c_tip, 0., 0.]))

        self.fuel_central_cg = wing_loc_root + 0.40*np.array([wing_c_root, 0., 0.])

        self.fuel_total_cg = (  self.fuel_central_cg*self.central_volume
                              + self.fuel_cantilever_cg*self.cantilever_volume
                              ) / (self.central_volume + self.cantilever_volume)

        # REMARK : if energy_source is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(energy_source)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        cant_str_volume = self.cantilever_volume / (1. - self.structure_ratio)*self.structure_ratio
        cent_str_volume = self.central_volume / (1. - self.structure_ratio)*self.structure_ratio

        self.mass = 1750.*(cant_str_volume + cent_str_volume)
        self.cg = self.fuel_total_cg

        self.fuel_max_fwd_cg = self.fuel_central_cg    # Fuel max forward CG, central tank is forward only within backward swept wing
        self.fuel_max_fwd_mass = self.central_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.fuel_cantilever_cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.cantilever_volume*self.fuel_density


class Tank_wing_pod(Component):

    def __init__(self, aircraft):

        super(Tank_wing_pod, self).__init__(aircraft)

        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.requirement.n_pax_front
        n_aisle = self.aircraft.requirement.n_aisle

        self.structure_ratio = 0.05
        self.surface_mass = 50.
        self.fuel_density = None

        self.pod_length = 0.30*(7.8*(0.38*n_pax_front + 1.05*n_aisle + 0.55) + 0.005*(n_pax_ref/n_pax_front)**2.25)
        self.pod_width = 0.70*(0.38*n_pax_front + 1.05*n_aisle + 0.55)
        self.pod_volume = None
        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = None
        self.fuel_max_fwd_mass = None
        self.fuel_max_bwd_cg = None
        self.fuel_max_bwd_mass = None

    def eval_geometry(self):
        body_width = self.aircraft.airframe.body.width
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_dihedral = self.aircraft.airframe.wing.dihedral
        wing_loc_root = self.aircraft.airframe.wing.loc_root
        wing_c_kink = self.aircraft.airframe.wing.c_kink
        wing_loc_kink = self.aircraft.airframe.wing.loc_kink
        wing_c_tip = self.aircraft.airframe.wing.c_tip
        wing_loc_tip = self.aircraft.airframe.wing.loc_tip

        tan_phi0 = 0.25*(wing_c_kink-wing_c_tip)/(wing_loc_tip[1]-wing_loc_kink[1]) + np.tan(wing_sweep25)

        if (self.aircraft.arrangement.nacelle_attachment == "pod"):
            pod_y_axe = 0.8 * body_width + 1.5 * self.pod_width
        else:
            pod_y_axe = 0.8 * body_width + 3.0 * self.pod_width

        pod_x_axe = wing_loc_root[0] + (pod_y_axe-wing_loc_root[1])*tan_phi0 - 0.40*self.pod_length
        pod_z_axe = (pod_y_axe - 0.5 * body_width) * np.tan(wing_dihedral)

        self.frame_origin = [pod_x_axe, pod_y_axe, pod_z_axe]

        wing_c_axe = wing_c_kink - (wing_c_kink-wing_c_tip)/(wing_loc_tip[1]-wing_loc_kink[1])*(pod_y_axe-wing_loc_kink[1])
        wing_x_axe = wing_loc_kink[0] - (wing_loc_kink[0]-wing_loc_tip[0])/(wing_loc_tip[1]-wing_loc_kink[1])*(pod_y_axe-wing_loc_kink[1])

        self.pod_net_wetted_area = 2.*(0.85*3.14*self.pod_width*self.pod_length)
        self.aero_length = self.pod_length
        self.form_factor = 1.05

        self.pod_volume =   0.85 \
                          * 2.0*self.pod_length*(0.25*np.pi*self.pod_width**2) \
                          * (1. - self.structure_ratio)                             # for both pods

        self.max_volume = self.pod_volume

    def eval_mass(self):
        energy_source = self.aircraft.arrangement.energy_source

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(energy_source)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        self.mass = self.pod_net_wetted_area*self.surface_mass
        self.cg = self.frame_origin + 0.45*np.array([self.pod_length, 0., 0.])

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.pod_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.pod_volume*self.fuel_density


class Tank_piggy_back(Component):

    def __init__(self, aircraft):

        super(Tank_piggy_back, self).__init__(aircraft)

        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.requirement.n_pax_front
        n_aisle = self.aircraft.requirement.n_aisle

        self.structure_ratio = 0.05
        self.surface_mass = 50.
        self.fuel_density = None

        self.pod_length = 0.30*(7.8*(0.38*n_pax_front + 1.05*n_aisle + 0.55) + 0.005*(n_pax_ref/n_pax_front)**2.25)
        self.pod_width = 0.70*(0.38*n_pax_front + 1.05*n_aisle + 0.55)
        self.pod_volume = None
        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = None
        self.fuel_max_fwd_mass = None
        self.fuel_max_bwd_cg = None
        self.fuel_max_bwd_mass = None

    def eval_geometry(self):
        body_width = self.aircraft.airframe.body.width
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_dihedral = self.aircraft.airframe.wing.dihedral
        wing_loc_mac = self.aircraft.airframe.wing.loc_mac
        wing_c_kink = self.aircraft.airframe.wing.c_kink
        wing_loc_kink = self.aircraft.airframe.wing.loc_kink
        wing_c_tip = self.aircraft.airframe.wing.c_tip
        wing_loc_tip = self.aircraft.airframe.wing.loc_tip

        pod_x_axe = wing_loc_mac[0] - 0.35*self.pod_length
        pod_y_axe = 0.
        pod_z_axe = 1.07*body_width + 0.85*self.pod_width

        self.frame_origin = [pod_x_axe, pod_y_axe, pod_z_axe]

        self.pod_net_wetted_area = 0.85*3.14*self.pod_width*self.pod_length
        self.aero_length = self.pod_length
        self.form_factor = 1.05

        self.pod_volume =   0.85 \
                          * self.pod_length*(0.25*np.pi*self.pod_width**2) \
                          * (1. - self.structure_ratio)

        self.max_volume = self.pod_volume

    def eval_mass(self):
        energy_source = self.aircraft.arrangement.energy_source

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(energy_source)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        self.mass = self.pod_net_wetted_area*self.surface_mass
        self.cg = self.frame_origin[0] + 0.45*np.array([self.pod_length, 0., 0.])

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.pod_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.pod_volume*self.fuel_density


class Landing_gear(Component):

    def __init__(self, aircraft):
        super(Landing_gear, self).__init__(aircraft)

    def eval_geometry(self):
        pass

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        mlw = self.aircraft.weight_cg.mlw
        wing_c_root = self.aircraft.airframe.wing.c_root
        wing_loc_root = self.aircraft.airframe.wing.loc_root

        self.mass = 0.02*mtow**1.03 + 0.012*mlw    # Landing gears
        self.cg = wing_loc_root[0] + 0.70*wing_c_root


class System(Component):

    def __init__(self, aircraft):
        super(System, self).__init__(aircraft)

    def eval_geometry(self):
        pass

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        body_cg = self.aircraft.airframe.body.cg
        wing_cg = self.aircraft.airframe.wing.cg
        horizontal_stab_cg = self.aircraft.airframe.horizontal_stab.cg
        vertical_stab_cg = self.aircraft.airframe.vertical_stab.cg
        landing_gear_cg = self.aircraft.airframe.landing_gear.cg

        self.mass = 0.545*mtow**0.8    # global mass of all systems

        self.cg =   0.50*body_cg \
                  + 0.20*wing_cg \
                  + 0.10*landing_gear_cg \
                  + 0.05*horizontal_stab_cg \
                  + 0.05*vertical_stab_cg
    #              + 0.10*power_system_cg \         # TODO
