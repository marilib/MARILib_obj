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

    def optimization(self):
        """Perform hq based empennage sizing and longitudinal wing positioning
        """
        def fct(x_in):
            self.aircraft.airframe.wing.x_root = x_in[0]
            self.aircraft.airframe.horizontal_stab.area = x_in[1]
            self.aircraft.airframe.vertical_stab.area = x_in[2]

            self.aircraft.airframe.geometry_analysis(hq_optim=True) # Recompute geometry
            self.aircraft.weight_cg.mass_pre_design()               # Recompute weights and CGs
            self.aircraft.aerodynamics.aerodynamic_analysis()       # Recompute aerodynamic characteristics
            self.aircraft.handling_quality.analysis()               # Recompute CG matching

            y_out = [self.forward_cg_stall - self.forward_cg_req[0],
                     self.backward_cg_stab - self.backward_cg_req[0],
                     self.backward_cg_oei_trim - self.backward_cg_req[0]]
            return y_out
        #-----------------------------------------------------------------------------------------------------------
        x_ini = [self.aircraft.airframe.wing.x_root,
                 self.aircraft.airframe.horizontal_stab.area,
                 self.aircraft.airframe.vertical_stab.area]

        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)

        if (output_dict[2]!=1):
            print(output_dict[3])
            raise Exception("Convergence problem in HQ optimization")

        self.aircraft.airframe.wing.x_root = output_dict[0][0]
        self.aircraft.airframe.horizontal_stab.area = output_dict[0][1]
        self.aircraft.airframe.vertical_stab.area = output_dict[0][2]

        self.aircraft.airframe.geometry_analysis(hq_optim=True) # Recompute geometry
        self.aircraft.weight_cg.mass_pre_design()               # Recompute weights and CGs
        self.aircraft.aerodynamics.aerodynamic_analysis()       # Recompute aerodynamic characteristics
        self.aircraft.handling_quality.analysis()               # Recompute CG matching

        return

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
                               + self.aircraft.airframe.tank.fuel_max_fwd_cg * self.aircraft.airframe.tank.fuel_max_fwd_mass ) \
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
                                + self.aircraft.airframe.tank.fuel_max_bwd_cg * self.aircraft.airframe.tank.fuel_max_bwd_mass ) \
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
        cz_max_htp = cza_htp*aoa_max_htp    # Assuming max down lift on HTP
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

        cm_prop = self.thrust_pitch_moment(fn,pamb,mach,dcx_oei,nei)

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

        cg_max_bwd_stab =  (xlc_wo_htp*cza_wo_htp + xlc_htp*cza_htp*(1.-ki_wing*cza_wo_htp)) \
                         / (cza_wo_htp + cza_htp*(1.-ki_wing*cza_wo_htp)) \
                         - stability_margin*wing_mac

        return cg_max_bwd_stab

    def max_bwd_cg_oei(self, altp, disa):
        """Computes maximum backward CG position to meet engine failure case constraint
        WARNING : Influence of CG position is ignored
        """
        wing_mac = self.aircraft.airframe.wing.mac
        owe = self.aircraft.weight_cg.owe
        cyb_vtp, xlc_vtp, aoa_max_vtp, ki_vtp = self.aircraft.airframe.vertical_stab.eval_aero_data()

        payload_miss = 0.5*self.aircraft.airframe.cabin.nominal_payload      # Light payload
        range_miss = self.aircraft.requirement.design_range/15.              # Short mission
        altp_miss = self.aircraft.requirement.cruise_altp                    # Nominal altitude
        mach_miss = self.aircraft.requirement.cruise_mach                    # Nominal speed
        disa_miss = self.aircraft.requirement.cruise_disa                    # Nominal temperature

        self.aircraft.performance.mission.toy.eval(owe,altp_miss,mach_miss,disa_miss, range=range_miss, payload=payload_miss)
        tow = self.aircraft.performance.mission.toy.tow

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        stall_margin = self.aircraft.requirement.take_off.kvs1g
        czmax_to = self.aircraft.aerodynamics.czmax_conf_to
        mach_s1g = self.speed_from_lift(pamb,tamb,czmax_to,tow) # Stall speed in Mach number
        mach_35ft = stall_margin*mach_s1g                       # V2 speed
        mach_mca = mach_35ft/1.1                                # Approximation of required VMCA

        throttle = 1.
        nei = 1

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        dict = self.aircraft.power_system.thrust(pamb, tamb, mach_mca, "MTO", throttle, nei)
        fn = dict["fn1"]    # Main engines only
        dcx_oei = self.aircraft.power_system.oei_drag(pamb,mach_mca)
        cn_prop = self.thrust_yaw_moment(fn,pamb,mach_mca,dcx_oei,nei)
        backward_cg_oei = xlc_vtp - (cn_prop*wing_mac)/(cyb_vtp*aoa_max_vtp)

        return backward_cg_oei, tow

    def thrust_pitch_moment(self,fn,pamb,mach,dcx_oei,nei):
        """Computes the pitch moment due to most outboard engine failure
        WARNING : Assumed right engine inoperative
        """
        r,gam,Cp,Cv = earth.gas_data()

        n_engine = self.aircraft.power_system.n_engine
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        qs = 0.5*gam*pamb*mach**2*wing_area
        ufn = fn / (n_engine - nei)         # Unitary thrust of main engines

        # Compute cm_prop as if all engines were running
        cm_prop = 0.
        for comp in self.aircraft.airframe:
            if comp.__class__.__name__ in self.aircraft.airframe.engine_analysis_order:
                cm_prop += (self.aircraft.weight_cg.owe_cg[2] - comp.frame_origin[2]) * ufn / (qs*wing_mac)

        # Replace thrust by oei drag for failed engines
        if nei==0:
            pass
        elif nei==1:
            cm_prop += (self.aircraft.weight_cg.owe_cg[2] - self.aircraft.airframe.nacelle.frame_origin[2]) * (-ufn/qs - dcx_oei) / wing_mac
        elif nei==2:
            cm_prop += (self.aircraft.weight_cg.owe_cg[2] - self.aircraft.airframe.nacelle.frame_origin[2]) * (-ufn/qs - dcx_oei) / wing_mac
            if n_engine==4:
                cm_prop += (self.aircraft.weight_cg.owe_cg[2] - self.aircraft.airframe.internal_nacelle.frame_origin[2]) * (-ufn/qs - dcx_oei) / wing_mac
            elif n_engine==6:
                cm_prop += (self.aircraft.weight_cg.owe_cg[2] - self.aircraft.airframe.median_nacelle.frame_origin[2]) * (-ufn/qs - dcx_oei) / wing_mac
        elif nei==3:
            cm_prop += (self.aircraft.weight_cg.owe_cg[2] - self.aircraft.airframe.nacelle.frame_origin[2]) * (-ufn/qs - dcx_oei) / wing_mac
            cm_prop += (self.aircraft.weight_cg.owe_cg[2] - self.aircraft.airframe.median_nacelle.frame_origin[2]) * (-ufn/qs - dcx_oei) / wing_mac
            cm_prop += (self.aircraft.weight_cg.owe_cg[2] - self.aircraft.airframe.internal_nacelle.frame_origin[2]) * (-ufn/qs - dcx_oei) / wing_mac

        return cm_prop

    def thrust_yaw_moment(self,fn,pamb,mach,dcx_oei,nei):
        """Computes the yaw moment due to most outboard engine failure
        WARNING : Assumed right engine inoperative
        """
        r,gam,Cp,Cv = earth.gas_data()

        n_engine = self.aircraft.power_system.n_engine
        wing_area = self.aircraft.airframe.wing.area
        wing_mac = self.aircraft.airframe.wing.mac

        qs = 0.5*gam*pamb*mach**2*wing_area
        ufn = fn / (n_engine - nei)         # Unitary thrust of main engines

        # cn_prop as if all engines were running
        cn_prop = 0.

        # Replace thrust by oei drag for failed engines
        if nei==0:
            pass
        elif nei==1:
            cn_prop += self.aircraft.airframe.nacelle.frame_origin[1] * (ufn/qs + dcx_oei) / wing_mac
        elif nei==2:
            cn_prop += self.aircraft.airframe.nacelle.frame_origin[1] * (ufn/qs + dcx_oei) / wing_mac
            if n_engine==4:
                cn_prop += self.aircraft.airframe.internal_nacelle.frame_origin[1] * (ufn/qs + dcx_oei) / wing_mac
            elif n_engine==6:
                cn_prop += self.aircraft.airframe.median_nacelle.frame_origin[1] * (ufn/qs + dcx_oei) / wing_mac
        elif nei==3:
            cn_prop += self.aircraft.airframe.nacelle.frame_origin[1] * (ufn/qs + dcx_oei) / wing_mac
            cn_prop += self.aircraft.airframe.median_nacelle.frame_origin[1] * (ufn/qs + dcx_oei) / wing_mac
            cn_prop += self.aircraft.airframe.internalnacelle.frame_origin[1] * (ufn/qs + dcx_oei) / wing_mac

        return cn_prop



