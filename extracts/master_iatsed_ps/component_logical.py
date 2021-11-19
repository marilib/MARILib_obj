#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import unit, util



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
        """Compute global geometrical data WITHOUT tail sizing
        """
        self.total_wet_area = 0.
        for comp in self.airplane:
            comp.eval_geometry()
            self.total_wet_area += comp.wet_area

    def solve(self):
        """Perform geometrical design
        """
        self.airplane.cabin.eval_geometry()
        self.airplane.fuselage.eval_geometry()
        self.airplane.wing.eval_geometry()
        self.airplane.tank.eval_geometry()
        self.airplane.nacelles.eval_geometry()

        self.airplane.htp.solve_area()                     # Solver inside
        self.airplane.vtp.solve_area()                     # Solver inside

        self.airplane.landing_gears.eval_geometry()
        self.airplane.systems.eval_geometry()
        self.airplane.geometry.eval_wet_area()


class Mass(object):
    def __init__(self, airplane, d_owe):
        self.airplane = airplane

        mtow_init = 5. * 110. * airplane.cabin.n_pax
        mzfw_init = 0.75 * mtow_init
        mlw_init = 1.07 * mzfw_init
        owe_init = 0.5 * mtow_init
        mwe_init = 0.4 * mtow_init

        self.mtow = mtow_init
        self.mlw = mlw_init
        self.mzfw = mzfw_init
        self.d_owe = d_owe
        self.owe = owe_init
        self.mwe = mwe_init
        self.mfw = None

        self.nominal_payload = None
        self.max_payload = None

    def eval_equiped_mass(self):
        """Mass computations
        """
        for comp in self.airplane:
            comp.eval_mass()

    def eval_characteristic_mass(self):
        cabin = self.airplane.cabin
        tank = self.airplane.tank
        missions = self.airplane.missions

        self.owe = self.d_owe
        for comp in self.airplane:
            self.owe += comp.mass
        self.nominal_payload = 105. * cabin.n_pax
        self.max_payload = 120. * cabin.n_pax
        self.mtow = self.owe + self.nominal_payload + missions.nominal.fuel_total
        self.mzfw = self.owe + self.max_payload
        self.mlw = min(self.mtow , (1.07*self.mzfw))
        self.mwe = self.owe - cabin.m_op_item
        self.mfw = max(0, min((803. * tank.fuel_volume), self.mtow - self.owe))


class Aerodynamics(object):
    def __init__(self, airplane, hld_type, hld_conf_to, hld_conf_ld):
        self.airplane = airplane

        self.hld_type = hld_type
        self.hld_conf_to = hld_conf_to
        self.hld_conf_ld = hld_conf_ld

        self.hld_data = { 0 : [1.45 , "Clean"],
                          1 : [2.25 , "Flap only, Rotation without slot"],
                          2 : [2.60 , "Flap only, Rotation single slot (ATR)"],
                          3 : [2.80 , "Flap only, Rotation double slot"],
                          4 : [2.80 , "Fowler Flap"],
                          5 : [2.00 , "Slat only"],
                          6 : [2.45 , "Slat + Flap rotation without slot"],
                          7 : [2.70 , "Slat + Flap rotation single slot"],
                          8 : [2.90 , "Slat + Flap rotation double slot"],
                          9 : [3.00 , "Slat + Fowler (A320)"],
                         10 : [3.20 , "Slat + Fowler + Fowler double slot (A321)"]}

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
        czmax_ld = self.hld_data.get(self.hld_type, "Erreur - high_lift_, HLDtype out of range")[0]    # 9 is default if x not found

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
        re = util.reynolds_number(pamb, tamb, mach)

        fac = ( 1. + 0.126*mach**2 )

        cxf = 0.
        for comp in self.airplane:
            nwa = comp.wet_area
            ael = comp.aero_length
            frm = comp.form_factor
            if ael>0.:
                # Drag model is based on flat plane friction drag
                cxf += frm * ((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) \
                           * (nwa/wing.area)
            else:
                # Drag model is based on drag area, in that case nwa is frontal area
                cxf += frm * (nwa/wing.area)

        # Parasitic drag (seals, antennas, sensors, ...)
        #-----------------------------------------------------------------------------------------------------------
        knwa = geometry.total_wet_area/1000.

        kp = (0.0247*knwa - 0.11)*knwa + 0.166       # Parasitic drag factor

        cx_par = cxf*kp

        # Fuselage tail cone drag
        #-----------------------------------------------------------------------------------------------------------
        cx_tap = 0.0020

        # Total zero lift drag
        #-----------------------------------------------------------------------------------------------------------
        cx0 = cxf + cx_par + cx_tap

        # Induced drag
        #-----------------------------------------------------------------------------------------------------------
        ki_wing = (1.05 + (fuselage.width / wing.span)**2)  / (np.pi * wing.aspect_ratio)
        cxi = ki_wing*cz**2  # Induced drag

        # Compressibility drag
        #-----------------------------------------------------------------------------------------------------------
        # Freely inspired from Korn equation
        cz_design = 0.5
        mach_div = self.airplane.cruise_mach + (0.03 + 0.1*(cz_design-cz))

        if 0.55 < mach:
            cxc = 0.0025 * np.exp(40.*(mach - mach_div) )
        else:
            cxc = 0.

        # Sum up
        #-----------------------------------------------------------------------------------------------------------
        cx = cx0 + cxi + cxc
        lod = cz/cx
        fd = (gam/2.)*pamb*mach**2*wing.area*cx

        return fd,cx,lod


class Propulsion(object):
    """Propulsion models
    """
    def __init__(self, airplane):
        self.airplane = airplane

        self.ratings = {"MTO":1.00, "MCN":0.86, "MCL":0.78, "MCR":0.70, "FID":0.05}
        self.n_engine = 2

    def unitary_thrust(self,pamb,tamb,mach,rating):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        nacelles = self.airplane.nacelles

        kth0 =  0.091*(nacelles.engine_bpr/10.)**2 \
              - 0.081*nacelles.engine_bpr/10. + 1.192
        kth =  0.475*mach**2 + 0.091*(nacelles.engine_bpr/10.)**2 \
             - 0.283*mach*nacelles.engine_bpr/10. \
             - 0.633*mach - 0.081*nacelles.engine_bpr/10. + 1.192
        rho,sig = util.air_density(pamb, tamb)
        thrust = nacelles.engine_slst * (kth/kth0) * self.ratings[rating] * sig**0.75
        return thrust

    def unitary_sc(self,pamb,tamb,mach,thrust):
        """Unitary thrust of a pure turbofan engine (semi-empirical model)
        """
        nacelles = self.airplane.nacelles

        sfc = 1.20 * ( 0.4 + 1./nacelles.engine_bpr**0.895 )/36000.
        return sfc

    def oei_drag(self,pamb,tamb):
        """Inoperative engine drag coefficient
        """
        wing = self.airplane.wing
        nacelles = self.airplane.nacelles

        dCx = 0.12*nacelles.diameter**2 / wing.area
        return dCx

