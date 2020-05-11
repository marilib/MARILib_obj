#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin

.. note:: All physical parameters are given in SI units.
"""

import numpy as np

from marilib.context import earth, unit


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
        self.aero_length = 1.       # characteristic length of the component in the direction of the flow
        self.form_factor = 0.       # factor on skin friction to account for lift independent pressure drag

    def eval_geometry(self):
        raise NotImplementedError

    def eval_mass(self):
        raise NotImplementedError

    def get_mass_mwe(self):
        return self.mass

    def get_mass_owe(self):
        return self.mass

    def get_cg_mwe(self):
        return self.cg

    def get_cg_owe(self):
        return self.cg

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
        self.nominal_payload = None
        self.maximum_payload = None

        self.cg_furnishing = np.full(3,None)
        self.cg_op_item = np.full(3,None)

        self.max_fwd_req_cg = np.full(3,None)
        self.max_fwd_mass = None
        self.max_bwd_req_cg = np.full(3,None)
        self.max_bwd_mass = None

    def eval_geometry(self):
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.requirement.n_pax_front
        n_aisle = self.aircraft.requirement.n_aisle

        self.width = 0.38*n_pax_front + 1.05*n_aisle + 0.15     # Statistical regression
        self.length = 6.3*(self.width - 0.24) + 0.005*(n_pax_ref/n_pax_front)**2.25     # Statistical regression

        self.projected_area = 0.95*self.length*self.width       # Factor 0.95 accounts for tapered parts

    def eval_mass(self):
        design_range = self.aircraft.requirement.design_range
        n_pax_ref = self.aircraft.requirement.n_pax_ref
        m_pax_nominal = self.aircraft.requirement.m_pax_nominal
        m_pax_max = self.aircraft.requirement.m_pax_max
        cabin_frame_origin = self.aircraft.airframe.cabin.frame_origin

        self.m_furnishing = (0.063*n_pax_ref**2 + 9.76*n_pax_ref)       # Furnishings mass
        self.m_op_item = 5.2*(n_pax_ref*design_range*1e-6)          # Operator items mass

        self.nominal_payload = n_pax_ref * m_pax_nominal
        self.maximum_payload = n_pax_ref * m_pax_max

        self.max_fwd_req_cg = cabin_frame_origin + 0.35*np.array([self.length, 0., 0.])        # Payload max forward CG
        self.max_fwd_mass = 0.60*n_pax_ref*m_pax_max       # Payload mass for max forward CG

        self.max_bwd_req_cg = cabin_frame_origin + 0.70*np.array([self.length, 0., 0.])        # Payload max backward CG
        self.max_bwd_mass = 0.70*n_pax_ref*m_pax_max       # Payload mass for max backward CG

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


class Cargo(Component):

    def __init__(self, aircraft):
        super(Cargo, self).__init__(aircraft)

    def eval_geometry(self):
        self.frame_origin = self.aircraft.airframe.cabin.frame_origin

    def eval_mass(self):
        n_pax_front = self.aircraft.requirement.n_pax_front
        body_width = self.aircraft.airframe.body.width
        body_length = self.aircraft.airframe.body.length
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_root_c = self.aircraft.airframe.wing.root_c
        cabin_length = self.aircraft.airframe.cabin.length

        if (n_pax_front>=6):
            forward_hold_length = wing_root_loc[0] - self.frame_origin[0]
            backward_hold_length = self.frame_origin[0]+cabin_length - (wing_root_loc[0]+wing_root_c)
            hold_length = forward_hold_length + backward_hold_length

            self.mass = (4.36*body_width*body_length)        # Container and pallet mass
            self.cg =   (self.frame_origin + 0.5*np.array([forward_hold_length, 0., 0.]))*(forward_hold_length/hold_length) \
                      + (wing_root_loc + np.array([wing_root_c+0.5*backward_hold_length, 0., 0.]))*(backward_hold_length/hold_length)
        else:
            self.mass = 0.
            self.cg = np.array([0., 0., 0.])

    def get_mass_mwe(self):
        return 0.

    def get_cg_mwe(self):
        return np.array([0., 0., 0.])



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
        self.aircraft.airframe.cargo.frame_origin = [fwd_limit, 0., 0.]     # cabin position inside the fuselage

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
        self.taper_ratio = None
        self.sweep0 = None
        self.sweep25 = None
        self.sweep100 = None
        self.dihedral = None
        self.setting = None
        self.hld_type = 9
        self.induced_drag_factor = None

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_toc = None                # thickness over chord ratio of root chord
        self.root_c = None                  # root chord length

        y_kink = 1.75*(0.38*n_pax_front + 1.05*n_aisle + 0.55)

        self.kink_loc =  np.full(3,None)    # Position of kink chord leading edge
        self.kink_toc = None                # thickness over chord ratio of kink chord
        self.kink_c = None                  # kink chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_toc = None                 # thickness over chord ratio of tip chord
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        wing_attachment = self.aircraft.arrangement.wing_attachment
        cruise_mach = self.aircraft.requirement.cruise_mach
        body_width = self.aircraft.airframe.body.width
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        mtow = self.aircraft.weight_cg.mtow

        self.tip_toc = 0.10
        self.kink_toc = self.tip_toc + 0.01
        self.root_toc = self.kink_toc + 0.03

        self.sweep25 = 1.6*max(0.,(cruise_mach - 0.5))     # Empirical law

        self.dihedral = unit.rad_deg(5.)

        if(self.morphing=="aspect_ratio_driven"):   # Aspect ratio is driving parameter
            self.span = np.sqrt(self.aspect_ratio*self.area)
        elif(self.morphing=="span_driven"): # Span is driving parameter
            self.aspect_ratio = self.span**2/self.area
        else:
            print("geometry_predesign_, wing_morphing index is unkown")

        # Correlation between span loading and tapper ratio
        self.taper_ratio = 0.3 - 0.025*(1e-3*mtow/self.span)

        y_root = 0.5*body_width
        y_kink = 1.75*body_width
        y_tip = 0.5*self.span

        if(15< unit.deg_rad(self.sweep25)):  # With kink
          Phi100intTE = max(0., 2. * (self.sweep25 - unit.rad_deg(32.)))
          tan_phi100 = np.tan(Phi100intTE)
          A = ((1-0.25*self.taper_ratio)*y_kink+0.25*self.taper_ratio*y_root-y_tip) / (0.75*y_kink+0.25*y_root-y_tip)
          B = (np.tan(self.sweep25)-tan_phi100) * ((y_tip-y_kink)*(y_kink-y_root)) / (0.25*y_root+0.75*y_kink-y_tip)
          self.root_c = (self.area-B*(y_tip-y_root)) / (y_root+y_kink+A*(y_tip-y_root)+self.taper_ratio*(y_tip-y_kink))
          self.kink_c = A*self.root_c + B
          self.tip_c = self.taper_ratio*self.root_c

        else:		# Without kink
          self.root_c = 2.*self.area / (2.*y_root*(1.-self.taper_ratio) + (1.+self.taper_ratio)*np.sqrt(self.aspect_ratio*self.area))
          self.tip_c = self.taper_ratio*self.root_c
          self.kink_c = ((y_tip-y_kink)*self.root_c + (y_kink-y_root)*self.tip_c) / (y_tip-y_root)


        tan_phi0 = 0.25*(self.kink_c-self.tip_c)/(y_tip-y_kink) + np.tan(self.sweep25)

        self.mac = 2.*( 3.*y_root*self.root_c**2 \
                       +(y_kink-y_root)*(self.root_c**2+self.kink_c**2+self.root_c*self.kink_c) \
                       +(y_tip-y_kink)*(self.kink_c**2+self.tip_c**2+self.kink_c*self.tip_c) \
                      )/(3*self.area)

        y_mac = (  3.*self.root_c*y_root**2 \
                 +(y_kink-y_root)*(self.kink_c*(y_root+y_kink*2.)+self.root_c*(y_kink+y_root*2.)) \
                 +(y_tip-y_kink)*(self.tip_c*(y_kink+y_tip*2.)+self.kink_c*(y_tip+y_kink*2.)) \
                )/(3.*self.area)

        x_mac_local = ( (y_kink-y_root)*tan_phi0*((y_kink-y_root)*(self.kink_c*2.+self.root_c) \
                       +(y_tip-y_kink)*(self.kink_c*2.+self.tip_c))+(y_tip-y_root)*tan_phi0*(y_tip-y_kink)*(self.tip_c*2.+self.kink_c) \
                      )/(3*self.area)

        x_root = 0.33*body_length**1.1 - (x_mac_local + 0.25*self.mac)
        x_kink = x_root + (y_kink-y_root)*tan_phi0
        x_tip = x_root + (y_tip-y_root)*tan_phi0

        x_mac = x_root+( (x_kink-x_root)*((y_kink-y_root)*(self.kink_c*2.+self.root_c) \
                            +(y_tip-y_kink)*(self.kink_c*2.+self.tip_c))+(x_tip-x_root)*(y_tip-y_kink)*(self.tip_c*2.+self.kink_c) \
                           )/(self.area*3.)
        if (wing_attachment=="low"):
            z_root = 0.
        else:
            z_root = body_height - 0.5*self.root_toc*self.root_c

        z_kink = z_root+(y_kink-y_root)*np.tan(self.dihedral)
        z_tip = z_root+(y_tip-y_root)*np.tan(self.dihedral)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.kink_loc = np.array([x_kink, y_kink, z_kink])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, None])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.00*(self.area - self.root_c*body_width)
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

        self.induced_drag_factor = (1.05 + (body_width / self.span)**2)  / (np.pi * self.aspect_ratio)

        # Wing setting
        #-----------------------------------------------------------------------------------------------------------
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()

        disa = 0.
        rca = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        mass = 0.95*self.aircraft.weight_cg.mtow

        pamb,tamb,tstd,dtodz = earth.atmosphere(rca, disa)

        cza_wing = self.cza(mach)

        # AoA = 2.5° at cruise start
        self.setting = (0.97*mass*g) / (0.5*gam*pamb*mach**2*self.area*cza_wing) - unit.rad_deg(2.5)

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        mzfw = self.aircraft.weight_cg.mzfw
        hld_conf_ld = self.aircraft.aerodynamics.hld_conf_ld

        (cz_max_ld,cz0) = self.high_lift(hld_conf_ld)

        A = 32*self.area**1.1
        B = 4.*self.span**2 * np.sqrt(mtow*mzfw)
        C = 1.1e-6*(1.+2.*self.aspect_ratio)/(1.+self.aspect_ratio)
        D = (0.6*self.root_toc+0.3*self.kink_toc+0.1*self.tip_toc) * (self.area/self.span)
        E = np.cos(self.sweep25)**2
        F = 1200.*(cz_max_ld - 1.8)**1.5

        self.mass = A + (B*C)/(D*E) + F   # Shevell formula + high lift device regression

        self.cg =  0.25*(self.root_loc + 0.40*np.array([self.root_c, 0., 0.])) \
                 + 0.55*(self.kink_loc + 0.40*np.array([self.kink_c, 0., 0.])) \
                 + 0.20*(self.tip_loc + 0.40*np.array([self.tip_c, 0., 0.]))

    def cza(self, mach):
        """Polhamus formula
        """
        body_width = self.aircraft.airframe.body.width
        wing_span = self.aircraft.airframe.wing.span
        wing_ar = self.aircraft.airframe.wing.aspect_ratio
        sweep25 = self.aircraft.airframe.wing.sweep25

        cza =  (np.pi*wing_ar*(1.07*(1+body_width/wing_span)**2)*(1.-body_width/wing_span)) \
             / (1+np.sqrt(1.+0.25*wing_ar**2*(1+np.tan(sweep25)**2-mach**2)))
        return cza

    def wing_np(self, hld_conf):
        """Wing neutral point
        """
        wing_c_mac = self.aircraft.airframe.wing.mac
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc

        loc_np = wing_mac_loc + (0.25+0.10*hld_conf)*np.array([wing_c_mac, 0., 0.])
        return loc_np

    def high_lift(self, hld_conf):
        """Retrieves max lift and zero aoa lift of a given deflection (from 0 to 1)
        0 =< hld_type =< 10 : type of high lift device
        0 =< hld_conf =< 1  : (slat) flap deflection
        Typically : hld_conf = 1 ==> cz_max_ld
                  : hld_conf = 0.1 to 0.5 ==> cz_max_to
        """

        # Maximum lift coefficients of different airfoils, DUBS 1987
        czmax_ld = {0 : 1.45 ,  # Clean
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
            czmax_base = 1.45      # Flap only
        else:
            if (hld_conf==0):
                czmax_base = 1.45  # Clean
            else:
                czmax_base = 2.00  # Slat + Flap

        czmax = (1-hld_conf)*czmax_base + hld_conf*czmax_ld
        cz0 = czmax - czmax_base  # Assumed the Lift vs AoA is just translated upward and Cz0 clean equal to zero
        return czmax, cz0


class VtpClassic(Component):

    def __init__(self, aircraft):
        super(VtpClassic, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Coupling variable
        self.height = None
        self.aspect_ratio = 1.7     # Design rule
        self.taper_ratio = 0.40     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.thrust_volume_factor = 0.4    # Design rule
        self.wing_volume_factor = 0.07    # Design rule
        self.anchor_ratio = None
        self.lever_arm = None

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_c = None                  # root chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        tail_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.height = np.sqrt(self.aspect_ratio*self.area)
        self.root_c = 2*self.area/(self.height*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.root_c

        self.sweep25 = max(unit.rad_deg(25.), wing_sweep25 + unit.rad_deg(10.)) # Empirical law

        self.anchor_ratio = 0.85       # Locate self versus end body length
        x_root = body_length*(1-tail_cone_length/body_length*(1-self.anchor_ratio)) - self.root_c
        x_tip = x_root + 0.25*(self.root_c-self.tip_c) + self.height*np.tan(self.sweep25)

        y_root = 0.
        y_tip = 0.

        z_root = body_height
        z_tip = z_root + self.height

        self.mac = self.height*(self.root_c**2+self.tip_c**2+self.root_c*self.tip_c)/(3*self.area)
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.tip_c+self.root_c)/(6*self.area)
        y_mac = 0.
        z_mac = z_tip**2*(2*self.tip_c+self.root_c)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 25. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        nacelle_loc_ext = self.aircraft.airframe.nacelle.frame_origin
        wing_area = self.aircraft.airframe.wing.area
        wing_span = self.aircraft.airframe.wing.span
        area_1 = self.thrust_volume_factor*(1.e-3*reference_thrust*nacelle_loc_ext[1])/self.lever_arm
        area_2 = self.wing_volume_factor*(wing_area*wing_span)/self.lever_arm
        self.area = max(area_1,area_2)


class VtpTtail(Component):

    def __init__(self, aircraft):
        super(VtpTtail, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Coupling variable
        self.height = None
        self.aspect_ratio = 1.2     # Design rule
        self.taper_ratio = 0.80     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.thrust_volume_factor = 0.4    # Design rule
        self.wing_volume_factor = 0.07    # Design rule
        self.anchor_ratio = None
        self.lever_arm = None

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_c = None                  # root chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        tail_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.height = np.sqrt(self.aspect_ratio*self.area)
        self.root_c = 2*self.area/(self.height*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.root_c

        self.sweep25 = max(unit.rad_deg(25.), wing_sweep25 + unit.rad_deg(10.)) # Empirical law

        self.anchor_ratio = 0.85       # Locate self versus end body length
        x_root = body_length*(1-tail_cone_length/body_length*(1-self.anchor_ratio)) - self.root_c
        x_tip = x_root + 0.25*(self.root_c-self.tip_c) + self.height*np.tan(self.sweep25)

        y_root = 0.
        y_tip = 0.

        z_root = body_height
        z_tip = z_root + self.height

        self.mac = self.height*(self.root_c**2+self.tip_c**2+self.root_c*self.tip_c)/(3*self.area)
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.tip_c+self.root_c)/(6*self.area)
        y_mac = 0.
        z_mac = z_tip**2*(2*self.tip_c+self.root_c)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, 0., z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 28. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        nacelle_loc_ext = self.aircraft.airframe.nacelle.frame_origin
        wing_area = self.aircraft.airframe.wing.area
        wing_span = self.aircraft.airframe.wing.span
        area_1 = self.thrust_volume_factor*(1.e-3*reference_thrust*nacelle_loc_ext[1])/self.lever_arm
        area_2 = self.wing_volume_factor*(wing_area*wing_span)/self.lever_arm
        self.area = max(area_1,area_2)


class VtpHtail(Component):

    def __init__(self, aircraft):
        super(VtpHtail, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.20*wing_area  # Coupling variable
        self.height = None
        self.aspect_ratio = 1.5     # Design rule
        self.taper_ratio = 0.40     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.thrust_volume_factor = 0.4    # Design rule
        self.wing_volume_factor = 0.07    # Design rule
        self.lever_arm = None

        self.root_loc = np.full(3,None)     # Position of root chord leading edge
        self.root_c = None                  # root chord length

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        htp_tip_loc = self.aircraft.airframe.horizontal_stab.tip_loc
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.height = np.sqrt(self.aspect_ratio*(0.5*self.area))
        self.root_c = 2*(0.5*self.area)/(self.height*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.root_c

        self.sweep25 = max(unit.rad_deg(25.), wing_sweep25 + unit.rad_deg(10.)) # Empirical law

        x_root = htp_tip_loc[0]
        x_tip = x_root + 0.25*(self.root_c-self.tip_c) + self.height*np.tan(self.sweep25)

        y_root = htp_tip_loc[1]
        y_tip = htp_tip_loc[1]

        z_root = htp_tip_loc[2]
        z_tip = z_root + self.height

        self.mac = self.height*(self.root_c**2+self.tip_c**2+self.root_c*self.tip_c)/(3*(0.5*self.area))
        x_mac = x_root+(x_tip-x_root)*self.height*(2*self.tip_c+self.root_c)/(6*(0.5*self.area))
        y_mac = y_tip
        z_mac = z_tip**2*(2*self.tip_c+self.root_c)/(6*self.area)

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.root_loc = np.array([x_root, y_root, z_root])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = [x_root, y_root, z_root]

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 25. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        nacelle_loc_ext = self.aircraft.airframe.nacelle.frame_origin
        wing_area = self.aircraft.airframe.wing.area
        wing_span = self.aircraft.airframe.wing.span
        area_1 = self.thrust_volume_factor*(1.e-3*reference_thrust*nacelle_loc_ext[1])/self.lever_arm
        area_2 = self.wing_volume_factor*(wing_area*wing_span)/self.lever_arm
        self.area = max(area_1,area_2)


class HtpClassic(Component):

    def __init__(self, aircraft):
        super(HtpClassic, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Coupling variable
        self.span = None
        self.aspect_ratio = 5.0     # Design rule
        self.taper_ratio = 0.35     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.dihedral = unit.rad_deg(5)     # HTP dihedral
        self.volume_factor = 0.94                  # Design rule
        self.lever_arm = None

        self.axe_loc = np.full(3,None)     # Position of the virtual central chord
        self.axe_c = None                  # Length of the virtual central chord

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_height = self.aircraft.airframe.body.height
        vtp_root_loc = self.aircraft.airframe.vertical_stab.root_loc
        vtp_root_c = self.aircraft.airframe.vertical_stab.root_c
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = htp_z_wise_anchor*body_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.axe_c = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.axe_c

        self.sweep25 = wing_sweep25 + unit.rad_deg(5)     # Design rule

        self.mac = self.span*(self.axe_c**2+self.tip_c**2+self.axe_c*self.tip_c)/(3.*self.area)
        y_mac = y_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        z_mac = z_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        x_tip_local = 0.25*(self.axe_c-self.tip_c) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.tip_c*2.+self.axe_c)/(3.*self.area)

        x_axe = vtp_root_loc[0] + 0.50*vtp_root_c - 0.2*self.axe_c

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.axe_loc = np.array([x_axe, y_axe, z_axe])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.axe_loc

        self.gross_wet_area = 1.63*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 22. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        self.area = self.volume_factor*(wing_area*wing_mac/self.lever_arm)


class HtpTtail(Component):

    def __init__(self, aircraft):
        super(HtpTtail, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Coupling variable
        self.span = None
        self.aspect_ratio = 5.0     # Design rule
        self.taper_ratio = 0.35     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.dihedral = unit.rad_deg(5)     # HTP dihedral
        self.volume_factor = 0.94                  # Design rule
        self.lever_arm = None

        self.axe_loc = np.full(3,None)     # Position of the central chord
        self.axe_c = None                  # Length of the central chord

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_height = self.aircraft.airframe.body.height
        vtp_tip_loc = self.aircraft.airframe.vertical_stab.tip_loc
        vtp_tip_c = self.aircraft.airframe.vertical_stab.tip_c
        vtp_height = self.aircraft.airframe.vertical_stab.height
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = body_height + vtp_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.axe_c = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.axe_c

        self.sweep25 = wing_sweep25 + unit.rad_deg(5)     # Design rule

        self.mac = self.span*(self.axe_c**2+self.tip_c**2+self.axe_c*self.tip_c)/(3.*self.area)
        y_mac = y_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        z_mac = z_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        x_tip_local = 0.25*(self.axe_c-self.tip_c) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.tip_c*2.+self.axe_c)/(3.*self.area)

        x_axe = vtp_tip_loc[0] + 0.30*vtp_tip_c - 0.80*self.tip_c

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.axe_loc = np.array([x_axe, y_axe, z_axe])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.axe_loc

        self.gross_wet_area = 2.01*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 22. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        self.area = self.volume_factor*(wing_area*wing_mac/self.lever_arm)


class HtpHtail(Component):

    def __init__(self, aircraft):
        super(HtpHtail, self).__init__(aircraft)

        wing_area = aircraft.airframe.wing.area

        self.area = 0.33*wing_area  # Coupling variable
        self.span = None
        self.aspect_ratio = 5.0     # Design rule
        self.taper_ratio = 0.45     # Design rule
        self.toc = 0.10             # Design rule
        self.sweep25 = None
        self.dihedral = unit.rad_deg(5)     # HTP dihedral
        self.volume_factor = 0.94                  # Design rule
        self.lever_arm = None

        self.axe_loc = np.full(3,None)     # Position of the virtual central chord
        self.axe_c = None                  # Length of the virtual central chord

        self.tip_loc = np.full(3,None)      # Position of tip chord leading edge
        self.tip_c = None                   # tip chord length

        self.mac_loc = np.full(3,None)      # Position of MAC chord leading edge
        self.mac = None

    def eval_geometry(self):
        body_length = self.aircraft.airframe.body.length
        body_height = self.aircraft.airframe.body.height
        body_cone_length = self.aircraft.airframe.body.tail_cone_length
        wing_sweep25 = self.aircraft.airframe.wing.sweep25
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_mac = self.aircraft.airframe.wing.mac

        self.span = np.sqrt(self.aspect_ratio*self.area)
        y_axe = 0.
        y_tip = 0.5*self.span

        htp_z_wise_anchor = 0.80       # Locate HTP versus end body height
        z_axe = htp_z_wise_anchor*body_height
        z_tip = z_axe + y_tip*np.tan(self.dihedral)

        self.axe_c = 2.*self.area/(self.span*(1+self.taper_ratio))
        self.tip_c = self.taper_ratio*self.axe_c

        self.sweep25 = wing_sweep25 + unit.rad_deg(5)     # Design rule

        self.mac = self.span*(self.axe_c**2+self.tip_c**2+self.axe_c*self.tip_c)/(3.*self.area)
        y_mac = y_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        z_mac = z_tip**2*(2*self.tip_c+self.axe_c)/(3*self.area)
        x_tip_local = 0.25*(self.axe_c-self.tip_c) + y_tip*np.tan(self.sweep25)
        x_mac_local = y_tip*x_tip_local*(self.tip_c*2.+self.axe_c)/(3.*self.area)

        htp_x_wise_anchor = 0.85
        x_axe = body_length*(1-body_cone_length/body_length*(1-htp_x_wise_anchor)) - self.axe_c

        x_tip = x_axe + x_tip_local
        x_mac = x_axe + x_mac_local

        self.lever_arm = (x_mac + 0.25*self.mac) - (wing_mac_loc[0] + 0.25*wing_mac)

        self.axe_loc = np.array([x_axe, y_axe, z_axe])
        self.tip_loc = np.array([x_tip, y_tip, z_tip])
        self.mac_loc = np.array([x_mac, y_mac, z_mac])

        self.frame_origin = self.axe_loc

        self.gross_wet_area = 1.63*self.area
        self.net_wet_area = self.gross_wet_area

        self.aero_length = self.mac
        self.form_factor = 1.40

    def eval_mass(self):
        self.mass = 22. * self.area
        self.cg = self.mac_loc + 0.20*np.array([self.mac, 0., 0.])

    def eval_area(self):
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        self.area = self.volume_factor*(wing_area*wing_mac/self.lever_arm)


class TankWingBox(Component):

    def __init__(self, aircraft):
        super(TankWingBox, self).__init__(aircraft)

        self.shell_parameter = unit.Pam3pkg_barLpkg(700.)   # bar.L/kg
        self.shell_density = None
        self.fuel_pressure = unit.Pa_bar(0.)
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
        wing_root_toc = self.aircraft.airframe.wing.root_toc
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_kink_toc = self.aircraft.airframe.wing.kink_toc
        wing_tip_toc = self.aircraft.airframe.wing.tip_toc
        energy_source = self.aircraft.arrangement.energy_source

        self.shell_density = earth.tank_shell_density(energy_source)

        shell_ratio = self.fuel_pressure/(self.shell_parameter*self.shell_density)

        self.cantilever_volume =   0.275 \
                                 * (wing_area*wing_mac*(0.50*wing_root_toc + 0.30*wing_kink_toc + 0.20*wing_tip_toc)) \
                                 * (1. - shell_ratio)

        self.central_volume =   1.3 \
                              * body_width * wing_root_toc * wing_mac**2 \
                              * (1. - shell_ratio)

        self.max_volume = self.central_volume + self.cantilever_volume

        self.frame_origin = [wing_root_loc[0], 0., wing_root_loc[2]]

    def eval_mass(self):
        energy_source = self.aircraft.arrangement.energy_source
        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc

        self.fuel_cantilever_cg =  0.25*(wing_root_loc + 0.40*np.array([wing_root_c, 0., 0.])) \
                                  + 0.65*(wing_kink_loc + 0.40*np.array([wing_kink_c, 0., 0.])) \
                                  + 0.10*(wing_tip_loc + 0.40*np.array([wing_tip_c, 0., 0.]))

        self.fuel_central_cg = wing_root_loc + 0.40*np.array([wing_root_c, 0., 0.])

        self.fuel_total_cg = (  self.fuel_central_cg*self.central_volume
                              + self.fuel_cantilever_cg*self.cantilever_volume
                              ) / (self.central_volume + self.cantilever_volume)

        # REMARK : if energy_source is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(energy_source)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        self.mass = (self.fuel_pressure/self.shell_parameter)*self.max_volume
        self.cg = self.fuel_total_cg

        self.fuel_max_fwd_cg = self.fuel_central_cg    # Fuel max forward CG, central tank is forward only within backward swept wing
        self.fuel_max_fwd_mass = self.central_volume*self.fuel_density

        self.fuel_max_bwd_cg = self.fuel_cantilever_cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.cantilever_volume*self.fuel_density


class TankWingPod(Component):

    def __init__(self, aircraft):
        super(TankWingPod, self).__init__(aircraft)

        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.requirement.n_pax_front
        n_aisle = self.aircraft.requirement.n_aisle

        self.shell_parameter = unit.Pam3pkg_barLpkg(700.)   # bar.L/kg
        self.shell_density = None
        self.fuel_pressure = unit.Pa_bar(0.)
        self.fuel_density = None

        self.length = 0.30*(7.8*(0.38*n_pax_front + 1.05*n_aisle + 0.55) + 0.005*(n_pax_ref/n_pax_front)**2.25)
        self.width = 0.70*(0.38*n_pax_front + 1.05*n_aisle + 0.55)
        self.wing_axe_c = None
        self.wing_axe_x = None
        self.volume = None
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
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        wing_kink_c = self.aircraft.airframe.wing.kink_c
        wing_kink_loc = self.aircraft.airframe.wing.kink_loc
        wing_tip_c = self.aircraft.airframe.wing.tip_c
        wing_tip_loc = self.aircraft.airframe.wing.tip_loc
        energy_source = self.aircraft.arrangement.energy_source

        self.shell_density = earth.tank_shell_density(energy_source)

        tan_phi0 = 0.25*(wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1]) + np.tan(wing_sweep25)

        if (self.aircraft.arrangement.nacelle_attachment == "pod"):
            y_axe = 0.8 * body_width + 1.5 * self.width
        else:
            y_axe = 0.8 * body_width + 3.0 * self.width

        x_axe = wing_root_loc[0] + (y_axe-wing_root_loc[1])*tan_phi0 - 0.40*self.length
        z_axe = (y_axe - 0.5 * body_width) * np.tan(wing_dihedral)

        self.frame_origin = [x_axe, y_axe, z_axe]

        self.wing_axe_c = wing_kink_c - (wing_kink_c-wing_tip_c)/(wing_tip_loc[1]-wing_kink_loc[1])*(y_axe-wing_kink_loc[1])
        self.wing_axe_x = wing_kink_loc[0] - (wing_kink_loc[0]-wing_tip_loc[0])/(wing_tip_loc[1]-wing_kink_loc[1])*(y_axe-wing_kink_loc[1])

        self.net_wet_area = 2.*(0.85*3.14*self.width*self.length)
        self.aero_length = self.length
        self.form_factor = 1.05

        shell_ratio = self.fuel_pressure/(self.shell_parameter*self.shell_density)

        self.volume =   0.85 \
                          * 2.0*self.length*(0.25*np.pi*self.width**2) \
                          * (1. - shell_ratio)                             # for both pods

        self.max_volume = self.volume

    def eval_mass(self):
        energy_source = self.aircraft.arrangement.energy_source

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(energy_source)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        self.mass = (self.fuel_pressure/self.shell_parameter)*self.max_volume
        self.cg = self.frame_origin + 0.45*np.array([self.length, 0., 0.])

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.volume*self.fuel_density


class TankPiggyBack(Component):

    def __init__(self, aircraft):
        super(TankPiggyBack, self).__init__(aircraft)

        n_pax_ref = self.aircraft.requirement.n_pax_ref
        n_pax_front = self.aircraft.requirement.n_pax_front
        n_aisle = self.aircraft.requirement.n_aisle

        self.shell_parameter = unit.Pam3pkg_barLpkg(700.)   # bar.L/kg
        self.shell_density = None
        self.fuel_pressure = unit.Pa_bar(0.)
        self.fuel_density = None

        self.length = 0.60*(7.8*(0.38*n_pax_front + 1.05*n_aisle + 0.55) + 0.005*(n_pax_ref/n_pax_front)**2.25)
        self.width = 0.70*(0.38*n_pax_front + 1.05*n_aisle + 0.55)
        self.volume = None
        self.max_volume = None
        self.mfw_volume_limited = None

        self.fuel_max_fwd_cg = None
        self.fuel_max_fwd_mass = None
        self.fuel_max_bwd_cg = None
        self.fuel_max_bwd_mass = None

    def eval_geometry(self):
        body_width = self.aircraft.airframe.body.width
        wing_mac_loc = self.aircraft.airframe.wing.mac_loc
        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_root_loc = self.aircraft.airframe.wing.root_loc
        energy_source = self.aircraft.arrangement.energy_source

        self.shell_density = earth.tank_shell_density(energy_source)

        x_axe = wing_mac_loc[0] - 0.40*self.length
        y_axe = 0.
        z_axe = 1.07*body_width + 0.85*self.width

        self.frame_origin = [x_axe, y_axe, z_axe]

        self.wing_axe_c = wing_root_c
        self.wing_axe_x = wing_root_loc[0]

        self.net_wet_area = 0.85*3.14*self.width*self.length
        self.aero_length = self.length
        self.form_factor = 1.05

        shell_ratio = self.fuel_pressure/(self.shell_parameter*self.shell_density)

        self.volume =   0.85 \
                          * self.length*(0.25*np.pi*self.width**2) \
                          * (1. - shell_ratio)

        self.max_volume = self.volume

    def eval_mass(self):
        energy_source = self.aircraft.arrangement.energy_source

        # REMARK : if fuel is "Battery", fuel density will be battery density
        self.fuel_density = earth.fuel_density(energy_source)
        self.mfw_volume_limited = self.max_volume*self.fuel_density

        self.mass = (self.fuel_pressure/self.shell_parameter)*self.max_volume
        self.cg = self.frame_origin[0] + 0.45*np.array([self.length, 0., 0.])

        self.fuel_max_fwd_cg = self.cg    # Fuel max Forward CG
        self.fuel_max_fwd_mass = self.volume*self.fuel_density

        self.fuel_max_bwd_cg = self.cg    # Fuel max Backward CG
        self.fuel_max_bwd_mass = self.volume*self.fuel_density


class LandingGear(Component):

    def __init__(self, aircraft):
        super(LandingGear, self).__init__(aircraft)

    def eval_geometry(self):
        wing_root_c = self.aircraft.airframe.wing.root_c
        wing_root_loc = self.aircraft.airframe.wing.root_loc

        self.frame_origin = wing_root_loc[0] + 0.85*wing_root_c

    def eval_mass(self):
        mtow = self.aircraft.weight_cg.mtow
        mlw = self.aircraft.weight_cg.mlw

        self.mass = 0.02*mtow**1.03 + 0.012*mlw    # Landing gears
        self.cg = self.frame_origin


