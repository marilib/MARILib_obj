#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

from marilib.utils import earth, unit

import numpy as np
from scipy.optimize import fsolve

from marilib.aircraft.performance import Flight
from marilib.aircraft.model_config import get_init

from marilib.utils.math import vander3, trinome, maximize_1d


class HandlingQuality(Flight):
    """
    Master class for all aircraft handling qualities
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.static_stab_margin = get_init(self,"static_stab_margin")

        self.forward_cg_mass = None
        self.forward_cg_req = None
        self.forward_cg_stall = None

        self.backward_cg_mass = None
        self.backward_cg_req = None
        self.backward_cg_stab = None

        self.backward_cg_oei_trim = None
        self.backward_cg_oei_mass = None

    def analysis(self):
        """Evaluate CG bounds according to HQ constraints
        """
        # Required forward CG position
        #------------------------------------------------------------------------------------------------------
        self.forward_cg_mass = (  self.aircraft.weight_cg.owe
                                + self.aircraft.airframe.cabin.pax_max_fwd_mass
                                + self.aircraft.airframe.cargo.freight_max_fwd_mass
                                + self.aircraft.airframe.tank.fuel_max_fwd_mass)
        self.forward_cg_req = (  self.aircraft.weight_cg.owe_cg * self.aircraft.weight_cg.owe
                               + self.aircraft.airframe.cabin.pax_max_fwd_cg * self.aircraft.airframe.cabin.pax_max_fwd_mass
                               + self.aircraft.airframe.cargo.freight_max_fwd_cg * self.aircraft.airframe.cargo.freight_max_fwd_mass
                               + self.aircraft.airframe.tank.fuel_max_fwd_cg * self.aircraft.airframe.cargo.fuel_max_fwd_mass ) \
                             /self.forward_cg_mass

        # Required backward CG position
        #------------------------------------------------------------------------------------------------------
        self.backward_cg_mass = (  self.aircraft.weight_cg.owe
                                 + self.aircraft.airframe.cabin.pax_max_bwd_mass
                                 + self.aircraft.airframe.cargo.freight_max_bwd_mass
                                 + self.aircraft.airframe.tank.fuel_max_bwd_mass)
        self.backward_cg_req = (  self.aircraft.weight_cg.owe_cg * self.aircraft.weight_cg.owe
                                + self.aircraft.airframe.cabin.pax_max_bwd_cg * self.aircraft.airframe.cabin.pax_max_bwd_mass
                                + self.aircraft.airframe.cargo.freight_max_bwd_cg * self.aircraft.airframe.cargo.freight_max_bwd_mass
                                + self.aircraft.airframe.tank.fuel_max_bwd_cg * self.aircraft.airframe.cargo.fuel_max_bwd_mass ) \
                              /self.backward_cg_mass

        # Forward limit : trim landing
        #------------------------------------------------------------------------------------------------------
        altp = unit.m_ft(0.)
        disa = 0.
        nei = 0
        speed_mode = "mach"
        hld_conf = self.aircraft.aerodynamics.hld_conf_ld
        mass = self.forward_cg_mass

        forward_cg_stall, speed, fn, aoa, ih, c_z, cx_trimmed = self.max_fwd_cg_stall(altp,disa,nei,hld_conf,speed_mode,mass)

        self.forward_cg_stall = forward_cg_stall    # Forward cg limit

        # Backward limit : static stability
        #------------------------------------------------------------------------------------------------------
        altp = unit.m_ft(0.)
        disa = 0.
        speed = 0.25
        speed_mode = "mach"
        hld_conf = self.aircraft.aerodynamics.hld_conf_clean

        self.backward_cg_stab = self.max_bwd_cg_stab(altp,disa,hld_conf,speed_mode,speed)

        # Backward limit : engine failure control
        #------------------------------------------------------------------------------------------------------
        altp = unit.m_ft(0.)
        disa = 15.

        backward_cg_oei_trim, backward_cg_oei_mass = self.max_bwd_cg_oei(altp, disa)

        self.backward_cg_oei_trim = backward_cg_oei_trim
        self.backward_cg_oei_mass = backward_cg_oei_mass

        return

    def max_fwd_cg_stall(self,altp,disa,nei,hld_conf,speed_mode,mass):
        """Computes max forward trimmable CG position at stall speed
        """
        htp_area = self.aircraft.airframe.horizontal_stab.area
        wing_area = self.aircraft.airframe.wing.area
        wing_setting = self.aircraft.airframe.wing.setting

        r,gam,Cp,Cv = earth.gas_data()
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)

        cz_max_wing,cz0 = self.aircraft.airframe.wing.high_lift(hld_conf)     # Wing maximum lift coefficient without margin
        cza_htp, xlc_htp, aoa_max_htp, ki_htp = self.aircraft.airframe.horizontal_stab.eval_aero_data()
        cz_max_htp = cza_htp*aoa_max_htp
        c_z = cz_max_wing - cz_max_htp      # Max forward Cg assumed, HTP has down lift
        mach = self.speed_from_lift(pamb, tamb, c_z, mass)
        cza_wo_htp, xlc_wo_htp, ki_wing = self.aircraft.airframe.wing.eval_aero_data(mach,hld_conf)

        if(nei>0):
            dcx_oei = self.aircraft.power_system.oei_drag(pamb,mach)
        else:
            dcx_oei = 0.

        dw_angle = self.aircraft.airframe.wing.downwash_angle(ki_wing,cz_max_wing)          # Downwash angle due to the wing
        cx_basic,lod_trash = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz_max_wing)    # By definition of the drag_ function
        cxi_htp = (ki_htp*cz_max_htp**2)*(htp_area/wing_area)   # Induced drag generated by HTP
        cx_inter = cz_max_htp*dw_angle                          # Interaction drag (due to downwash)
        cx_trimmed = cx_basic + cxi_htp + cx_inter + dcx_oei
        fn = 0.5*gam*pamb*mach**2*wing_area*cx_trimmed
        cm_prop = self.thrust_pitch_moment(fn,pamb,mach,dcx_oei)
        cg_max_fwd_stall = (cm_prop + xlc_wo_htp*cz_max_wing - xlc_htp*cz_max_htp)/(cz_max_wing - cz_max_htp)
        aoa_wing = (cz_max_wing-cz0) / cza_wo_htp   # Wing angle of attack
        aoa = aoa_wing - wing_setting               # Reference angle of attack (fuselage axis versus air speed)
        ih = - aoa + dw_angle - aoa_max_htp         # HTP trim setting
        speed = self.get_speed(pamb,speed_mode,mach)

        return cg_max_fwd_stall, speed, fn, aoa, ih, c_z, cx_trimmed

    def max_bwd_cg_stab(self, altp, disa, hld_conf, speed_mode, speed):
        """Computes max backward CG position according to static stability (neutral point position)
        """
        wing_mac = self.aircraft.airframe.wing.mac
        stability_margin = self.static_stab_margin

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        mach = self.get_mach(pamb,speed_mode,speed)

        cza_wo_htp, xlc_wo_htp, ki_wing = self.aircraft.airframe.wing.eval_aero_data(mach,hld_conf)
        cza_htp, xlc_htp, aoa_max_htp, ki_htp = self.aircraft.airframe.horizontal_stab.eval_aero_data()
        cg_max_bwd_stab =  (xlc_wo_htp*cza_wo_htp + xlc_htp*cza_htp*(1-ki_wing*cza_wo_htp)) \
                         / (cza_wo_htp + cza_htp*(1-ki_wing*cza_wo_htp)) \
                         - stability_margin*wing_mac

        return cg_max_bwd_stab

    def max_bwd_cg_oei(self, altp, disa):
        """Computes maximum backward CG position to meet engine failure case constraint
        WARNING : Influence of CG position is ignored
        """
        wing_mac = self.aircraft.airframe.wing.mac
        owe = self.aircraft.weight_cg.owe
        cyb_vtp, xlc_vtp, aoa_max_vtp, ki_vtp = self.aircraft.airframe.vertical_stab.eval_aero_data()

        payload = 0.5*self.aircraft.airframe.cabin.nominal_payload      # Light payload
        range = self.aircraft.requirement.design_range/15.              # Short mission
        altp = self.aircraft.requirement.cruise_altp                    # Nominal altitude
        mach = self.aircraft.requirement.cruise_mach                    # Nominal speed
        disa = self.aircraft.requirement.cruise_disa                    # Nominal temperature

        self.aircraft.performance.mission.toy.eval(owe,altp,mach,disa, range=range, payload=payload)
        tow = self.aircraft.performance.mission.toy.tow

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        stall_margin = self.aircraft.requirement.take_off.kvs1g
        czmax_to = self.aircraft.aerodynamics.cz_max_conf_to
        mach_s1g = self.speed_from_lift(pamb,tamb,czmax_to,tow)
        mach_35ft = stall_margin*mach_s1g       # V2 speed
        mach_mca = mach_35ft/1.1                # Approximation of required VMCA

        throttle = 1.
        nei = 1

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        dict = self.aircraft.power_system.thrust(pamb, tamb, mach_mca, "MTO", throttle, nei)
        fn = dict["fn"]
        dcx_oei = self.aircraft.power_system.oei_drag(pamb,mach)
        cn_prop = self.thrust_yaw_moment(fn,pamb,mach_mca,dcx_oei)
        backward_cg_oei = xlc_vtp - (cn_prop*wing_mac)/(cyb_vtp*aoa_max_vtp)

        return backward_cg_oei, tow

    def thrust_pitch_moment(aircraft,fn,pamb,mach,dcx_oei):

        propulsion = aircraft.propulsion
        wing = aircraft.wing

        r,gam,Cp,Cv = earth.gas_data()

        if (propulsion.architecture=="TF"):
            nacelle = aircraft.turbofan_nacelle
        elif (propulsion.architecture=="TP"):
            nacelle = aircraft.turboprop_nacelle
        elif (propulsion.architecture=="PTE1"):
            nacelle = aircraft.turbofan_nacelle
        elif (propulsion.architecture=="EF1"):
            nacelle = aircraft.electrofan_nacelle
        elif (propulsion.architecture=="EP1"):
            nacelle = aircraft.electroprop_nacelle
        else:
            raise Exception("propulsion.architecture index is out of range")

        if (nacelle.n_engine==2):
            cm_prop = nacelle.z_ext*(dcx_oei - fn/(0.5*gam*pamb*mach**2*wing.area))
        elif (nacelle.n_engine==4):
            cm_prop =   nacelle.z_ext*(dcx_oei - (fn/3.)/(0.5*gam*pamb*mach**2*wing.area)) \
                      - nacelle.z_int*(2.*fn/3.)/(0.5*gam*pamb*mach**2*wing.area)
        else:
            raise Exception("thrust_pitch_moment, Number of engine is not supported")

        return cm_prop

    def thrust_yaw_moment(aircraft,fn,pamb,mach,dcx_oei):
        """Computes the yaw moment due to most outboard engine failure
        WARNING : Assumed right engine inoperative
        """
        propulsion = aircraft.propulsion
        wing = aircraft.wing

        r,gam,Cp,Cv = earth.gas_data()

        cn_prop = (propulsion.y_ext_nacelle/wing.mac)*(fn/(0.5*gam*pamb*mach**2*wing.area) + dcx_oei)

        return cn_prop




#===========================================================================================================
def eval_hq0(aircraft):
    """
    Perform hq based empennage sizing without updating characteristic masses MTOW, MLW & MZFW
    """

    aircraft.center_of_gravity.cg_range_optimization = 1    # Start HQ optimization mode

    #===========================================================================================================
    def fct_hq_optim(x_in,aircraft):

        c_g = aircraft.center_of_gravity

        aircraft.wing.x_root = x_in[0]
        aircraft.horizontal_tail.area = x_in[1]
        aircraft.vertical_tail.area = x_in[2]

        eval_aircraft_pre_design(aircraft)   # Solves geometrical coupling without tails areas
        eval_mass_breakdown(aircraft)               # Just mass analysis without any solving
        eval_performance_analysis(aircraft)
        eval_handling_quality_analysis(aircraft)

        y_out = np.array([c_g.cg_constraint_1,
                          c_g.cg_constraint_2,
                          c_g.cg_constraint_3])
        return y_out
    #-----------------------------------------------------------------------------------------------------------

    x_ini = np.array([aircraft.wing.x_root,
                      aircraft.horizontal_tail.area,
                      aircraft.vertical_tail.area])

    fct_arg = aircraft

    output_dict = fsolve(fct_hq_optim, x0=x_ini, args=fct_arg, full_output=True)

    if (output_dict[2]!=1):
        print(output_dict[3])
        raise Exception("Convergence problem in HQ optimization")

    aircraft.wing.x_root = output_dict[0][0]
    aircraft.horizontal_tail.area = output_dict[0][1]
    aircraft.vertical_tail.area = output_dict[0][2]

    eval_mda0(aircraft)

    eval_handling_quality_analysis(aircraft)

    return



