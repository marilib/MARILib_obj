#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: DRUOT Thierry
"""

import numpy as np
from marilib.utils import unit


#-----------------------------------------------------------------------------------------------------------------------
# Atmosphere
#-----------------------------------------------------------------------------------------------------------------------

def atmosphere(altp, disa=0., full_output=False):
    """
    Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
    """
    g = 9.80665
    r = 287.053
    gam = 1.4

    Z = np.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])
    P = np.array([101325., 0., 0., 0., 0., 0.])
    T = np.array([288.15, 0., 0., 0., 0., 0.])

    if (Z[-1]<altp):
        raise Exception("atmosphere, altitude cannot exceed 50km")

    j = 0
    while (Z[1+j]<=altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<np.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
        else:
            P[j+1] = P[j]*np.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0.<np.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
    else:
        pamb = P[j]*np.exp(-(g/r)*((altp-Z[j])/T[j]))
    tstd = T[j] + dtodz[j]*(altp-Z[j])
    tamb = tstd + disa
    if full_output:
        return pamb,tamb,tstd,dtodz[j]
    else:
        return pamb,tamb

def sound_speed(tamb):
    """Sound speed for ideal gas
    """
    r = 287.053
    gam = 1.4
    vsnd = np.sqrt( gam * r * tamb )
    return vsnd

def air_density(pamb,tamb):
    """Ideal gas density
    """
    r = 287.053
    rho0 = 1.225
    rho = pamb / ( r * tamb )
    sig = rho / rho0
    return rho, sig

def gas_viscosity(tamb, gas="air"):
    mu0,T0,S = [1.715e-5, 273.15, 110.4]
    mu = (mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
    return mu

def reynolds_number(pamb,tamb,mach):
    """Reynolds number based on Sutherland viscosity model
    """
    vsnd = sound_speed(tamb)
    rho,sig = air_density(pamb,tamb)
    mu = gas_viscosity(tamb)
    re = rho*vsnd*mach/mu
    return re


#-----------------------------------------------------------------------------------------------------------------------
# Airplane component
#-----------------------------------------------------------------------------------------------------------------------

class Airplane(object):
    def __init__(self, design_range=unit.m_NM(2500), cruise_mach=0.78,
                 n_pax=150, n_aisle=1, n_front=6,
                 wing_area=122, wing_aspect_ratio=10, wing_taper_ratio=0.25, wing_toc_ratio=0.12, wing_sweep25=unit.rad_deg(25), wing_dihedral=unit.rad_deg(5),
                 htp_area=30, htp_aspect_ratio=5, htp_taper_ratio=0.35, htp_toc_ratio=0.10, htp_sweep25=unit.rad_deg(30), htp_dihedral=unit.rad_deg(5),
                 vtp_area=25, vtp_aspect_ratio=1.7, vtp_taper_ratio=0.4, vtp_toc_ratio=0.10, vtp_sweep25=unit.rad_deg(30),
                 engine_slst=unit.N_kN(120.), engine_bpr=10,
                 hld_type=9,
                 hld_conf_to=0.3, kvs1g_to=1.13, s2_min_path_to=0.024,
                 hld_conf_ld=1., kvs1g_ld=1.23):

        self.design_range = design_range
        self.cruise_mach = cruise_mach
        self.cruise_altp = unit.m_ft(35000)

        # Physical components
        self.fuselage = Fuselage(self, n_pax, n_aisle, n_front)
        self.wing = Wing(self, wing_area, wing_aspect_ratio, wing_taper_ratio, wing_toc_ratio, wing_sweep25, wing_dihedral)
        self.htp = HTP(self, htp_area, htp_aspect_ratio, htp_taper_ratio, htp_toc_ratio, htp_sweep25, htp_dihedral)
        self.vtp = VTP(self, vtp_area, vtp_aspect_ratio, vtp_taper_ratio, vtp_toc_ratio, vtp_sweep25)
        self.nacelles = Nacelles(self, engine_slst, engine_bpr)
        self.landing_gears = LandingGears(self)

        # Logical components
        self.aerodynamics = Aerodynamics(self, hld_type, hld_conf_to, hld_conf_ld)
        self.propulsion = Propulsion(self)
        self.geometry = Geometry(self)
        self.mass = Mass(self)

        self.missions = Missions(self)
        self.operations = Operations(self, hld_conf_to, kvs1g_to, s2_min_path_to,
                                           hld_conf_ld, kvs1g_ld)

    def __iter__(self):
        public = [value for value in self.__dict__.values() if issubclass(type(value),Component)]
        return iter(public)


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


class Fuselage(Component):
    def __init__(self, airplane, n_pax, n_aisle, n_front):
        super(Fuselage, self).__init__(airplane)

        self.n_pax = n_pax
        self.n_aisle = n_aisle
        self.n_front = n_front

        self.width = None
        self.length = None
        self.cabin_center = None

        self.fuselage_mass = None
        self.m_furnishing = None
        self.m_op_item = None

        self.seat_pitch = unit.m_inch(32)
        self.seat_width = unit.m_inch(19)
        self.aisle_width = unit.m_inch(20)

        self.position = 0.
        self.wall_thickness = 0.20  # Overall fuselage wall thickness
        self.front_ratio = 1.2      # Cabin start over fuselage width
        self.rear_ratio = 1.5       # Cabin end to tail cone over fuselage width

        self.form_factor = 1.05 # Form factor for drag calculation

    def eval_geometry(self):
        self.width =  self.seat_width*self.n_front + self.aisle_width*self.n_aisle + 0.08 + self.wall_thickness
        cabin_front = self.front_ratio * self.width
        cabin_length = self.seat_pitch*(self.n_pax/self.n_front) + 2*self.width
        self.cabin_center = cabin_front + 0.5*cabin_length
        self.length = (self.front_ratio + self.rear_ratio)*self.width + cabin_length
        self.wet_area = 2.70*self.length*self.width
        self.aero_length = self.length

    def eval_mass(self):
        self.fuselage_mass = 5.80*(np.pi*self.length*self.width)**1.2                   # Statistical regression versus fuselage built surface
        self.m_furnishing = (0.063*self.n_pax**2 + 9.76*self.n_pax)                     # Furnishings mass
        self.m_op_item = max(160., 5.2*(self.n_pax*self.airplane.design_range*1e-6))    # Operator items mass
        self.mass = self.fuselage_mass + self.m_furnishing + self.m_op_item


class Wing(Component):
    def __init__(self, airplane, area, aspect_ratio, taper_ratio, toc_ratio, sweep25, dihedral):
        super(Wing, self).__init__(airplane)

        self.area = area
        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.toc_ratio = toc_ratio
        self.sweep25 = sweep25              # Sweep angle at 25% of chords
        self.dihedral = dihedral

        self.span = None
        self.root = None
        self.tip = None
        self.mac = None
        self.mac_position = None    # X wise MAC position versus wing root
        self.position = None        # X wise wing root position versus fuselage
        self.fuel_volume = None

        self.form_factor = 1.4

    def eval_geometry(self):
        fuselage = self.airplane.fuselage

        self.span = np.sqrt(self.area*self.aspect_ratio)
        self.root = self.area / (fuselage.width + 0.5*(1+self.taper_ratio)*(self.span - fuselage.width))
        self.tip = self.taper_ratio * self.root
        self.mac = (2/(3*self.area))*(3*(0.5*fuselage.width)*self.root**2 + 0.5*(self.span-fuselage.width)*(self.root**2 + self.tip**2 + self.root*self.tip))

        chord_gradient = 2 * (self.tip - self.root) / (self.span-fuselage.width)
        tan_sweep0 = np.tan(self.sweep25) - 0.25*chord_gradient
        tip_position = tan_sweep0 * (self.span-fuselage.width)/2
        self.mac_position = (1/(3*self.area))*tip_position*((self.span-fuselage.width)/2)*(2*self.tip + self.root)

        self.position = fuselage.cabin_center - (self.mac_position + 0.25*self.mac)    # Set wing root position

        self.fuel_volume = 0.4 * self.area * self.mac*self.toc_ratio

        self.wet_area = 2*(self.area - fuselage.width*self.root)
        self.aero_length = self.mac

    def eval_mass(self):
        mtow = self.airplane.mass.mtow
        mzfw = self.airplane.mass.mzfw

        aerodynamics = self.airplane.aerodynamics

        hld_conf_ld = self.airplane.aerodynamics.hld_conf_ld

        cz_max_ld,cz0 = aerodynamics.wing_high_lift(hld_conf_ld)

        A = 32*self.area**1.1
        B = 4.*self.span**2 * np.sqrt(mtow*mzfw)
        C = 1.1e-6*(1.+2.*self.aspect_ratio)/(1.+self.aspect_ratio)
        D = self.toc_ratio * (self.area/self.span)
        E = np.cos(self.sweep25)**2
        F = 1200.*max(0., cz_max_ld - 1.8)**1.5

        self.mass = (A + (B*C)/(D*E) + F)   # Shevell formula + high lift device regression


class HTP(Component):
    def __init__(self, airplane, area=30, aspect_ratio=5, taper_ratio=0.35, toc_ratio=0.10, sweep25=unit.rad_deg(30), dihedral=unit.rad_deg(5)):
        super(HTP, self).__init__(airplane)

        self.area = area
        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.toc_ratio = toc_ratio
        self.sweep25 = sweep25              # Sweep angle at 25% of chords
        self.dihedral = dihedral

        self.span = None
        self.axe = None
        self.tip = None
        self.mac = None
        self.mac_position = None    # X wise MAC position versus HTP root
        self.position = None        # X wise HTP axe position versus fuselage
        self.lever_arm = None

        self.form_factor = 1.4

    def eval_geometry(self):
        fuselage = self.airplane.fuselage
        wing = self.airplane.wing

        self.span = np.sqrt(self.area*self.aspect_ratio)
        self.axe = (2/self.span) * (self.area / (1+self.taper_ratio))
        self.tip = self.taper_ratio * self.axe
        self.mac = (2/3) * self.axe * (1+self.taper_ratio-self.taper_ratio/(1+self.taper_ratio))

        chord_gradient = 2 * (self.tip - self.axe) / self.span
        tan_sweep0 = np.tan(self.sweep25) - 0.25*chord_gradient
        self.mac_position = (1/3) * (self.span/2) * ((1+2*self.taper_ratio)/(1+self.taper_ratio)) * tan_sweep0

        self.position = fuselage.length - 1.05*self.axe
        self.lever_arm = (self.position + self.mac_position + 0.25*self.mac) - (wing.position + wing.mac_position + 0.25*wing.mac)

        self.wet_area = 1.63*self.area
        self.aero_length = self.mac

    def eval_mass(self):
        self.mass = 22. * self.area


class VTP(Component):
    def __init__(self, airplane, area=25, aspect_ratio=1.7, taper_ratio=0.4, toc_ratio=0.10, sweep25=unit.rad_deg(30)):
        super(VTP, self).__init__(airplane)

        self.area = area
        self.aspect_ratio = aspect_ratio
        self.taper_ratio = taper_ratio
        self.toc_ratio = toc_ratio
        self.sweep25 = sweep25              # Sweep angle at 25% of chords

        self.span = None
        self.root = None
        self.tip = None
        self.mac = None
        self.mac_position = None    # X wise MAC position versus HTP root
        self.position = None        # X wise HTP axe position versus fuselage
        self.lever_arm = None

        self.form_factor = 1.4

    def eval_geometry(self):
        fuselage = self.airplane.fuselage
        wing = self.airplane.wing

        self.span = np.sqrt(self.area*self.aspect_ratio)
        self.axe = (2/self.span) * (self.area / (1+self.taper_ratio))
        self.tip = self.taper_ratio * self.axe
        self.mac = (2/3) * self.axe * (1+self.taper_ratio-self.taper_ratio/(1+self.taper_ratio))

        chord_gradient = 2 * (self.tip - self.axe) / self.span
        tan_sweep0 = np.tan(self.sweep25) - 0.25*chord_gradient
        self.mac_position = (1/3) * (self.span/2) * ((1+2*self.taper_ratio)/(1+self.taper_ratio)) * tan_sweep0

        self.position = fuselage.length - 1.15*self.axe
        self.lever_arm = (self.position + self.mac_position + 0.25*self.mac) - (wing.position + wing.mac_position + 0.25*wing.mac)

        self.wet_area = 2.0*self.area
        self.aero_length = self.mac

    def eval_mass(self):
        self.mass = 25. * self.area


class Nacelles(Component):
    def __init__(self, airplane, engine_slst, engine_bpr):
        super(Nacelles, self).__init__(airplane)

        self.engine_slst = engine_slst
        self.engine_bpr = engine_bpr

        self.diameter = None
        self.length = None
        self.span_position = None
        self.ground_clearence = None

        self.form_factor = 1.15

    def eval_geometry(self):
        wing = self.airplane.wing
        landing_gears = self.airplane.landing_gears

        self.diameter = 0.5*self.engine_bpr**0.7 + 5.E-6*self.engine_slst
        self.length = 0.86*self.diameter + self.engine_bpr**0.37      # statistical regression
        self.span_position = 1.5 * self.diameter

        self.ground_clearence = landing_gears.leg_length + self.span_position*np.tan(wing.dihedral) - self.diameter

        knac = np.pi * self.diameter * self.length
        self.wet_area = knac*(1.48 - 0.0076*knac)       # statistical regression, all engines

        self.aero_length = self.length
        self.form_factor = 1.15

    def eval_mass(self):
        propulsion = self.airplane.propulsion

        engine_mass = (1250. + 0.021*self.engine_slst) * propulsion.n_engine
        pylon_mass = (0.0031*self.engine_slst) * propulsion.n_engine
        self.mass = engine_mass + pylon_mass


class LandingGears(Component):
    def __init__(self, airplane, leg_length=2.):
        super(LandingGears, self).__init__(airplane)

        self.leg_length = leg_length

    def eval_geometry(self):
        pass

    def eval_mass(self):
        mtow = self.airplane.mass.mtow
        mlw = self.airplane.mass.mlw

        self.mass = (0.015*mtow**1.03 + 0.012*mlw)


#-----------------------------------------------------------------------------------------------------------------------
# Logical components
#-----------------------------------------------------------------------------------------------------------------------

class Geometry(object):
    def __init__(self, airplane):
        self.airplane = airplane

        self.total_wet_area = None

    def eval_wet_area(self):
        """Compute global geometrical data
        """
        self.total_wet_area = 0.
        for comp in self.airplane:
            self.total_wet_area += comp.wet_area

    def eval(self):
        """Compute global geometrical data
        """
        self.total_wet_area = 0.
        for comp in self.airplane:
            comp.eval_geometry()
            self.total_wet_area += comp.wet_area


class Mass(object):
    def __init__(self, airplane):
        self.airplane = airplane

        mtow_init = 5. * 110. * airplane.fuselage.n_pax
        mzfw_init = 0.75 * mtow_init
        mlw_init = 1.07 * mzfw_init
        owe_init = 0.5 * mtow_init

        self.mtow = mtow_init
        self.mlw = mlw_init
        self.mzfw = mzfw_init
        self.owe = owe_init

        self.nominal_payload = None
        self.max_payload = None

    def eval_owe(self):
        """Mass computations
        """
        self.owe = 0.
        for comp in self.airplane:
            self.owe += comp.mass

    def eval_other_mass(self):
        self.nominal_payload = 105. * self.airplane.fuselage.n_pax
        self.max_payload = 120. * self.airplane.fuselage.n_pax
        self.mtow = self.owe + self.nominal_payload + self.airplane.missions.nominal.fuel_total
        self.mzfw = self.owe + self.max_payload
        self.mlw = 1.07 * self.mzfw
        self.mfw = 0.803 * self.airplane.wing.fuel_volume

    def eval(self):
        """Mass computations
        """
        self.owe = 0.
        for comp in self.airplane:
            comp.eval_mass()
            self.owe += comp.mass

        self.nominal_payload = 105. * self.airplane.fuselage.n_pax
        self.max_payload = 120. * self.airplane.fuselage.n_pax

        self.mtow = self.owe + self.nominal_payload + self.airplane.missions.nominal.fuel_total
        self.mzfw = self.owe + self.max_payload
        self.mlw = 1.07 * self.mzfw
        self.mfw = 0.803 * self.airplane.wing.fuel_volume


class Aerodynamics(object):
    def __init__(self, airplane, hld_type, hld_conf_to, hld_conf_ld):
        self.airplane = airplane

        self.hld_type = hld_type
        self.hld_conf_to = hld_conf_to
        self.hld_conf_ld = hld_conf_ld

    def wing_high_lift(self, hld_conf):
        """Retrieves max lift and zero aoa lift of a given (flap/slat) deflection (from 0 to 1).
            * 0 =< hld_type =< 10 : type of high lift device
            * 0 =< hld_conf =< 1  : (slat) flap deflection
        Typically:
            * hld_conf = 1 gives the :math:`C_{z,max}` for landind (`czmax_ld`)
            * hld_conf = 0.1 to 0.5 gives the :math:`C_{z,max}` for take-off(`czmax_to`)
        """
        wing = self.airplane.wing

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
            if (hld_conf==0): czmax_base = 1.45 # Clean
            else: czmax_base = 2.00             # Slat + Flap

        czmax_2d = (1.-hld_conf)*czmax_base + hld_conf*czmax_ld     # Setting effect

        if (hld_conf==0):
            cz0_2d = 0. # Clean
        else:
            cz0_2d = czmax_2d - czmax_base  # Assumed the Lift vs AoA is just translated upward and Cz0 clean equal to zero

        # Source : http://aerodesign.stanford.edu/aircraftdesign/highlift/clmaxest.html
        czmax = czmax_2d * (1.-0.08*np.cos(wing.sweep25)**2) * np.cos(wing.sweep25)**0.75
        cz0 = cz0_2d

        return czmax, cz0

    def drag_force(self,pamb,tamb,mach,cz):
        """Retrieves airplane drag and L/D in current flying conditions
        """
        fuselage = self.airplane.fuselage
        wing = self.airplane.wing
        geometry = self.airplane.geometry

        gam = 1.4

        # Form & friction drag
        #-----------------------------------------------------------------------------------------------------------
        re = reynolds_number(pamb, tamb, mach)

        fac = ( 1. + 0.126*mach**2 )

        cxf = 0.
        for comp in self.airplane:
            nwa = comp.wet_area
            ael = comp.aero_length
            frm = comp.form_factor
            if ael>0.:
                # Drag model is based on flat plane friction drag
                cxfi = frm * ((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) \
                           * (nwa/wing.area)
                cxf += frm * ((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) \
                           * (nwa/wing.area)
            else:
                # Drag model is based on drag area, in that case nwa is frontal area
                cxfi = frm * (nwa/wing.area)
                cxf += frm * (nwa/wing.area)

        # Parasitic drag (seals, antennas, sensors, ...)
        #-----------------------------------------------------------------------------------------------------------
        knwa = geometry.total_wet_area/1000.

        kp = (0.0247*knwa - 0.11)*knwa + 0.166       # Parasitic drag factor

        cx_par = cxf*kp

        # Total zero lift drag
        #-----------------------------------------------------------------------------------------------------------
        cx0 = cxf + cx_par

        # Induced drag
        #-----------------------------------------------------------------------------------------------------------
        ki_wing = (1.05 + (fuselage.width / wing.span)**2)  / (np.pi * wing.aspect_ratio)
        cxi = ki_wing*cz**2  # Induced drag

        # Compressibility drag
        #-----------------------------------------------------------------------------------------------------------
        # Freely inspired from Korn equation
        cz_design = 0.5
        mach_div = self.airplane.cruise_mach + (0.03 + 0.1*(cz_design-cz))

        cxc = 0.0025 * np.exp(40.*(mach - mach_div) )

        # Sum up
        #-----------------------------------------------------------------------------------------------------------
        cx = cx0 + cxi + cxc
        lod = cz/cx
        fd = (gam/2.)*pamb*mach**2*self.airplane.wing.area*cx

        return fd,cx,lod


class Propulsion(object):
    def __init__(self, airplane):
        self.airplane = airplane

        self.ratings = {"MTO":1.00, "MCN":0.86, "MCL":0.78, "MCR":0.70, "FID":0.05}
        self.n_engine = 2

    def unitary_thrust(self,pamb,tamb,mach,rating):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        nacelles = self.airplane.nacelles
        kth =  0.475*mach**2 + 0.091*(nacelles.engine_bpr/10.)**2 \
             - 0.283*mach*nacelles.engine_bpr/10. \
             - 0.633*mach - 0.081*nacelles.engine_bpr/10. + 1.192
        rho,sig = air_density(pamb, tamb)
        thrust = nacelles.engine_slst * kth * self.ratings[rating]* sig**0.75
        return thrust

    def unitary_sc(self,pamb,tamb,mach,thrust):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        nacelles = self.airplane.nacelles
        sfc = ( 0.4 + 1./nacelles.engine_bpr**0.895 )/36000.
        return sfc

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing_area = self.airplane.wing.area
        nacelle_diameter = self.airplane.nacelles.diameter
        dCx = 0.12*nacelle_diameter**2 / wing_area
        return dCx


class Flight(object):
    """Usefull methods for all simulation
    """
    def __init__(self, airplane):
        self.airplane = airplane

    def level_flight(self,pamb,tamb,mach,mass):
        """Level flight equilibrium
        """
        aerodynamics = self.airplane.aerodynamics
        propulsion = self.airplane.propulsion

        g = 9.80665
        gam = 1.4
        cz = (2.*mass*g)/(gam*pamb*mach**2*self.airplane.wing.area)
        thrust,cx,lod = aerodynamics.drag_force(pamb,tamb,mach,cz)
        sfc = propulsion.unitary_sc(pamb,tamb,mach,thrust)
        return sfc, lod

    def mach_from_vcas(self,pamb,Vcas):
        """Mach number from calibrated air speed, subsonic only
        """
        gam = 1.4
        P0 = 101325.
        vc0 = 340.29    # m/s
        fac = gam/(gam-1.)
        mach = np.sqrt(((((((gam-1.)/2.)*(Vcas/vc0)**2+1)**fac-1.)*P0/pamb+1.)**(1./fac)-1.)*(2./(gam-1.)))
        return mach

    def vcas_from_mach(self,pamb,mach):
        """Calibrated air speed from Mach number, subsonic only
        """
        gam = 1.4
        P0 = 101325.
        vc0 = 340.29    # m/s
        fac = gam/(gam-1.)
        vcas = vc0*np.sqrt((2./(gam-1.))*((((pamb/P0)*((1.+((gam-1.)/2.)*mach**2)**fac-1.))+1.)**(1./fac)-1.))
        return vcas

    def get_speed(self,pamb,speed_mode,mach):
        """retrieve CAS or Mach from mach depending on speed_mode
        """
        speed = {"cas" : self.vcas_from_mach(pamb, mach),  # CAS required
                 "mach" : mach  # mach required
                 }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return speed

    def get_mach(self,pamb,speed_mode,speed):
        """Retrieve Mach from CAS or mach depending on speed_mode
        """
        mach = {"cas" : self.mach_from_vcas(pamb, speed),  # Input is CAS
                "mach" : speed  # Input is mach
                }.get(speed_mode, "Erreur: select speed_mode equal to cas or mach")
        return mach

    def climb_mode(self,speed_mode,mach,dtodz,tstd,disa):
        """Acceleration factor depending on speed driver ('cas': constant CAS, 'mach': constant Mach)
        WARNING : input is mach number whatever speed_mode
        """
        g = 9.80665
        r = 287.053
        gam = 1.4
        if (speed_mode=="cas"):
            fac = (gam-1.)/2.
            acc_factor = 1. + (((1.+fac*mach**2)**(gam/(gam-1.))-1.)/(1.+fac*mach**2)**(1./(gam-1.))) \
                            + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
        elif (speed_mode=="mach"):
            acc_factor = 1. + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
        else:
            raise Exception("climb_mode key is unknown")
        return acc_factor

    def lift_from_speed(self,pamb,tamb,mach,mass):
        """Retrieve cz from mach using simplified lift equation
        """
        g = 9.80665
        gam = 1.4
        cz = (2.*mass*g)/(gam*pamb*mach**2*self.airplane.wing.area)
        return cz

    def speed_from_lift(self,pamb,tamb,cz,mass):
        """Retrieve mach from cz using simplified lift equation
        """
        g = 9.80665
        gam = 1.4
        mach = np.sqrt((mass*g)/(0.5*gam*pamb*self.airplane.wing.area*cz))
        return mach

    def air_path(self,nei,altp,disa,speed_mode,speed,mass,rating,kfn, full_output=False):
        """Retrieve air path in various conditions
        """
        aerodynamics = self.airplane.aerodynamics
        propulsion = self.airplane.propulsion

        g = 9.80665
        pamb,tamb,tstd,dtodz = atmosphere(altp, disa, full_output=True)
        mach = self.get_mach(pamb,speed_mode,speed)

        thrust = propulsion.unitary_thrust(pamb,tamb,mach,rating)
        sfc = propulsion.unitary_sc(pamb,tamb,mach,thrust)
        fn = kfn*thrust*propulsion.n_engine
        ff = sfc*fn
        if kfn!=1. and full_output:
            print("WARNING, air_path method, kfn is different from 1, fuel flow may not be accurate")
        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        fd,cx,lod = aerodynamics.drag_force(pamb,tamb,mach,cz)

        if(nei>0):
            dcx = propulsion.oei_drag(pamb,mach)
            cx = cx + dcx*nei
            lod = cz/cx

        acc_factor = self.climb_mode(speed_mode, mach, dtodz, tstd, disa)
        slope = ( fn/(mass*g) - 1./lod ) / acc_factor
        vz = slope * mach * sound_speed(tamb)
        acc = (acc_factor-1.)*g*slope
        if full_output:
            return slope,vz,fn,ff,acc,cz,cx,pamb,tamb
        else:
            return slope,vz


class Missions(Flight):
    def __init__(self, airplane):
        super(Missions, self).__init__(airplane)

        self.nominal = Breguet(airplane)
        self.max_payload = Breguet(airplane)
        self.max_fuel = Breguet(airplane)
        self.zero_payload = Breguet(airplane)

    def eval_nominal_mission(self):
        """Compute missions
        """
        disa = 0.
        altp = self.airplane.cruise_altp
        mach = self.airplane.cruise_mach

        range = self.airplane.design_range
        tow = self.airplane.mass.mtow
        self.nominal.eval(range,tow,altp,mach,disa)

    def eval_max_payload_mission(self):
        """Compute missions
        """
        disa = 0.
        altp = self.airplane.cruise_altp
        mach = self.airplane.cruise_mach

        range = self.max_payload.range
        tow = self.airplane.mass.mtow
        self.max_payload.eval(range,tow,altp,mach,disa)
        self.airplane.mass.max_payload = tow - self.airplane.mass.owe - self.max_payload.fuel_total

    def eval_max_fuel_mission(self):
        """Compute missions
        """
        disa = 0.
        altp = self.airplane.cruise_altp
        mach = self.airplane.cruise_mach

        range = self.max_fuel.range
        tow = self.airplane.mass.mtow
        self.max_fuel.eval(range,tow,altp,mach,disa)
        self.airplane.mass.mfw = self.max_fuel.fuel_total

    def eval_zero_payload_mission(self):
        """Compute missions
        """
        disa = 0.
        altp = self.airplane.cruise_altp
        mach = self.airplane.cruise_mach

        range = self.zero_payload.range
        tow = self.airplane.mass.owe + self.airplane.mass.mfw
        self.zero_payload.eval(range,tow,altp,mach,disa)
        self.airplane.mass.mfw = self.zero_payload.fuel_total

    def eval_payload_range(self):
        """Compute missions
        """
        self.eval_max_payload_mission()
        self.eval_max_fuel_mission()
        self.eval_zero_payload_mission()


class Breguet(Flight):
    def __init__(self, airplane):
        super(Breguet, self).__init__(airplane)

        self.disa = 0.                      # Mean cruise temperature shift
        self.altp = airplane.cruise_altp    # Mean cruise altitude
        self.mach = airplane.cruise_mach    # Cruise mach number

        range_init = airplane.design_range
        tow_init = 5. * 110. * airplane.fuselage.n_pax
        total_fuel_init = 0.2 * tow_init

        self.range = range_init             # Mission distance
        self.tow = tow_init                 # Take Off Weight
        self.time_block = None              # Mission block duration
        self.fuel_block = None              # Mission block fuel consumption
        self.fuel_reserve = None            # Mission reserve fuel
        self.fuel_total = total_fuel_init   # Mission total fuel

        self.holding_time = unit.s_min(30)
        self.reserve_fuel_ratio = 0.05
        self.diversion_range = unit.m_NM(200)

    def holding(self,time,mass,altp,mach,disa):
        """Holding fuel
        """
        g = 9.80665
        pamb,tamb = atmosphere(altp, disa)
        sfc, lod = self.level_flight(pamb,tamb,mach,mass)
        fuel = sfc*(mass*g/lod)*time
        return fuel

    def breguet_range(self,range,tow,altp,mach,disa):
        """Breguet range equation
        """
        g = 9.80665
        pamb,tamb = atmosphere(altp, disa)
        tas = mach * sound_speed(tamb)
        time = 1.09*(range/tas)
        sfc, lod = self.level_flight(pamb,tamb,mach,tow)
        val = tow*(1-np.exp(-(sfc*g*range)/(tas*lod)))
        return val,time

    def eval(self,range,tow,altp,mach,disa):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        g = 9.80665

        self.range = range
        self.tow = tow

        n_engine = self.airplane.propulsion.n_engine
        engine_slst = self.airplane.nacelles.engine_slst
        engine_bpr = self.airplane.nacelles.engine_bpr

        # Departure ground legs
        #-----------------------------------------------------------------------------------------------------------
        time_taxi_out = 540.
        fuel_taxi_out = (34. + 2.3e-4*engine_slst)*n_engine

        time_take_off = 220.*tow/(engine_slst*n_engine)
        fuel_take_off = 1e-4*(2.8+2.3/engine_bpr)*tow

        # Mission leg
        #-----------------------------------------------------------------------------------------------------------
        fuel_mission,time_mission = self.breguet_range(range,tow,altp,mach,disa)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)     # mass is not landing weight

        # Arrival ground legs
        #-----------------------------------------------------------------------------------------------------------
        time_landing = 180.
        fuel_landing = 1e-4*(0.5+2.3/engine_bpr)*mass

        time_taxi_in = 420.
        fuel_taxi_in = (26. + 1.8e-4*engine_slst)*n_engine

        # Block fuel and time
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_block = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        #-----------------------------------------------------------------------------------------------------------
        fuel_diversion,t = self.breguet_range(self.diversion_range,mass,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.65 * mach
        fuel_holding = self.holding(self.holding_time,mass,altp_holding,mach_holding,disa)

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission*self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.fuel_block + self.fuel_reserve

        #-----------------------------------------------------------------------------------------------------------
        return


class Operations(Flight):
    def __init__(self, airplane, hld_conf_to, kvs1g_to, s2_min_path_to,
                                 hld_conf_ld, kvs1g_ld):
        super(Operations, self).__init__(airplane)

        self.take_off = TakeOff(airplane, hld_conf_to, kvs1g_to, s2_min_path_to)
        self.approach = Approach(airplane, hld_conf_ld, kvs1g_ld)
        self.mcl_ceiling = ClimbCeiling(airplane, rating="MCL", speed_mode="cas")
        self.mcr_ceiling = ClimbCeiling(airplane, rating="MCR", speed_mode="mach")

    def eval_take_off(self):
        """Compute performances
        """
        disa = 15.
        altp = unit.m_ft(0.)
        tow = self.airplane.mass.mtow
        self.take_off.eval(disa,altp,tow)

    def eval_approach(self):
        """Compute performances
        """
        disa = 0.
        altp = unit.m_ft(0.)
        lw = self.airplane.mass.mlw
        self.approach.eval(disa,altp,lw)

    def eval_climb(self):
        """Compute performances
        """
        disa = 15.
        altp = self.airplane.cruise_altp
        mach = self.airplane.cruise_mach
        mass = 0.97 * self.airplane.mass.mtow
        self.mcl_ceiling.eval(disa,altp,mach,mass)
        self.mcr_ceiling.eval(disa,altp,mach,mass)

    def eval(self):
        """Compute performances
        """
        self.eval_take_off()
        self.eval_approach()
        self.eval_climb()


class TakeOff(Flight):
    """Take Off Field Length
    """
    def __init__(self, airplane, hld_conf, kvs1g, s2_min_path):
        super(TakeOff, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.tow = None

        self.tofl = None
        self.kvs1g_eff = None
        self.v2 = None
        self.mach2 = None
        self.s2_path = None
        self.limit = None

        self.hld_conf = hld_conf
        self.kvs1g = kvs1g
        self.s2_min_path = s2_min_path

    def eval(self,disa,altp,mass):
        """Take off field length and climb path with eventual kVs1g increase to recover min regulatory slope
        """
        self.disa = disa
        self.altp = altp
        self.tow = mass

        s2_min_path = self.s2_min_path
        kvs1g = self.kvs1g
        hld_conf = self.hld_conf
        rating = "MTO"
        kfn = 1.

        tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf,rating,kfn)

        if(s2_min_path<s2_path):
            limitation = "fl"   # field length
        else:
            dkvs1g = 0.005
            kvs1g_ = np.array([0.,0.])
            kvs1g_[0] = kvs1g
            kvs1g_[1] = kvs1g_[0] + dkvs1g

            s2_path_ = np.array([0.,0.])
            s2_path_[0] = s2_path
            tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf,rating,kfn)

            while(s2_path_[0]<s2_path_[1] and s2_path_[1]<s2_min_path):
                kvs1g_[0] = kvs1g_[1]
                kvs1g_[1] = kvs1g_[1] + dkvs1g
                tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf,rating,kfn)

            if(s2_min_path<s2_path_[1]):
                kvs1g = kvs1g_[0] + ((kvs1g_[1]-kvs1g_[0])/(s2_path_[1]-s2_path_[0]))*(s2_min_path-s2_path_[0])
                tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf,rating,kfn)
                s2_path = s2_min_path
                limitation = "s2"   # second segment
            else:
                tofl = np.nan
                kvs1g = np.nan
                s2_path = 0.
                limitation = None

        self.tofl = tofl
        self.kvs1g_eff = kvs1g
        self.v2 = cas
        self.mach2 = mach
        self.s2_path = s2_path
        self.limit = limitation
        return

    def take_off(self,kvs1g,altp,disa,mass,hld_conf,rating,kfn):
        """Take off field length and climb path at 35 ft depending on stall margin (kVs1g)
        """
        aerodynamics = self.airplane.aerodynamics
        propulsion = self.airplane.propulsion

        czmax,cz0 = aerodynamics.wing_high_lift(hld_conf)

        pamb,tamb = atmosphere(altp, disa)
        rho,sig = air_density(pamb, tamb)

        cz_to = czmax / kvs1g**2
        mach = self.speed_from_lift(pamb,tamb,cz_to,mass)
        speed_factor = 0.7

        nei = 0             # For tofl computation
        thrust = propulsion.unitary_thrust(pamb,tamb,speed_factor*mach,rating)
        fn = kfn*thrust*(propulsion.n_engine - nei)

        ml_factor = mass**2 / (cz_to*fn*self.airplane.wing.area*sig**0.8 )  # Magic Line factor
        tofl = 11.8*ml_factor + 100.    # Magic line

        nei = 1             # For 2nd segment computation
        speed_mode = "cas"  # Constant CAS
        speed = self.get_speed(pamb,speed_mode,mach)

        s2_path,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,"MTO",kfn)

        return tofl,s2_path,speed,mach


class Approach(Flight):
    """Approach speed
    """
    def __init__(self, airplane, hld_conf, kvs1g):
        super(Approach, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.lw = None
        self.app_speed = None

        self.hld_conf = hld_conf
        self.kvs1g = kvs1g

    def eval(self,disa,altp,mass):
        """Minimum approach speed (VLS)
        """
        aerodynamics = self.airplane.aerodynamics

        self.disa = disa
        self.altp = altp
        self.lw = mass

        hld_conf = self.hld_conf
        kvs1g = self.kvs1g

        pamb,tamb = atmosphere(altp, disa)
        czmax,cz0 = aerodynamics.wing_high_lift(hld_conf)
        cz = czmax / kvs1g**2
        mach = self.speed_from_lift(pamb,tamb,cz,mass)
        vapp = self.get_speed(pamb,"cas",mach)
        self.app_speed = vapp
        return


class ClimbCeiling(Flight):
    """Propulsion ceiling in MCL rating
    """
    def __init__(self, airplane, rating, speed_mode):
        super(ClimbCeiling, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.mach = None
        self.vz = None

        self.rating = rating
        self.speed_mode = speed_mode

    def eval(self,disa,altp,mach,mass):
        """Residual climb speed in MCL rating
        """
        self.disa = disa
        self.altp = altp
        self.mach = mach

        speed_mode = self.speed_mode
        rating = self.rating
        kfn = 1.
        nei = 0
        pamb,tamb = atmosphere(self.altp, self.disa)
        speed = self.get_speed(pamb,self.speed_mode,self.mach)
        slope,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
        self.vz = vz
        return



#-----------------------------------------------------------------------------------------------------------------------
# Exposed modules
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    ap = Airplane()

# High level
#-----------------------------------------
    ap.geometry.eval()

    ap.mass.eval()

    ap.missions.eval_nominal_mission()

    ap.missions.eval_payload_range()

    ap.operations.eval()


# Fine level
#-----------------------------------------
    ap.fuselage.eval_geometry()

    ap.wing.eval_geometry()

    ap.htp.eval_geometry()

    ap.vtp.eval_geometry()

    ap.nacelles.eval_geometry()

    ap.landing_gears.eval_geometry()

    ap.geometry.eval_wet_area()


    ap.fuselage.eval_mass()

    ap.wing.eval_mass()

    ap.htp.eval_mass()

    ap.vtp.eval_mass()

    ap.nacelles.eval_mass()

    ap.landing_gears.eval_mass()

    ap.mass.eval_owe()

    ap.mass.eval_other_mass()


    ap.missions.eval_nominal_mission()

    ap.missions.eval_max_payload_mission()

    ap.missions.eval_max_fuel_mission()

    ap.missions.eval_zero_payload_mission()


    ap.operations.eval_take_off()

    ap.operations.eval_approach()

    ap.operations.eval_climb()


