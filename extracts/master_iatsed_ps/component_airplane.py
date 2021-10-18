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

import unit, util, process

from component_physical import Component, Cabin, Fuselage, Wing, Tank, HTP, VTP, Nacelles, LandingGears, Systems

from component_logical import Aerodynamics, Propulsion, Geometry, Mass

from performance import Missions, Operations, Economics

#-----------------------------------------------------------------------------------------------------------------------
# Airplane component
#-----------------------------------------------------------------------------------------------------------------------

class Airplane(object):
    def __init__(self, cruise_mach=0.78, design_range=unit.m_NM(3000), cost_range=unit.m_NM(500),
                 n_pax=170, n_aisle=1, n_front=6,
                 wing_area=122, wing_aspect_ratio=9, wing_taper_ratio=0.20, wing_toc_ratio=0.12, wing_sweep25=unit.rad_deg(25), wing_dihedral=unit.rad_deg(5), hld_type=9,
                 htp_aspect_ratio=5, htp_taper_ratio=0.35, htp_toc_ratio=0.10, htp_sweep25=unit.rad_deg(30), htp_dihedral=unit.rad_deg(5), volume=0.94,
                 vtp_aspect_ratio=1.7, vtp_taper_ratio=0.4, vtp_toc_ratio=0.10, vtp_sweep25=unit.rad_deg(30), thrust_volume=0.4,
                 engine_slst=unit.N_kN(120.), engine_bpr=9, z_ratio = 0.55,
                 leg_length=3.7,
                 holding_time=unit.s_min(30), reserve_fuel_ratio=0.05, diversion_range=unit.m_NM(200),
                 hld_conf_to=0.3, kvs1g_req_to=1.13, s2_path_req_to=0.024, hld_conf_ld=1., kvs1g_ld=1.23,
                 tofl_req=2100, app_speed_req=unit.mps_kt(137), vz_mcl_req=unit.mps_ftpmin(300), vz_mcr_req=unit.mps_ftpmin(0),
                 oei_path_req=0.011, oei_altp_req=unit.m_ft(13000),
                 d_owe=0, d_cost=0):

        self.design_range = design_range
        self.cost_range = cost_range
        self.cruise_mach = cruise_mach
        self.cruise_altp = unit.m_ft(35000)

        # Physical components
        self.cabin = Cabin(self, n_pax, n_aisle, n_front)
        self.fuselage = Fuselage(self)
        self.wing = Wing(self, wing_area, wing_aspect_ratio, wing_taper_ratio, wing_toc_ratio, wing_sweep25, wing_dihedral)
        self.tank = Tank(self)
        self.htp = HTP(self, htp_aspect_ratio, htp_taper_ratio, htp_toc_ratio, htp_sweep25, htp_dihedral, volume)
        self.vtp = VTP(self, vtp_aspect_ratio, vtp_taper_ratio, vtp_toc_ratio, vtp_sweep25, thrust_volume)
        self.nacelles = Nacelles(self, engine_slst, engine_bpr, z_ratio)
        self.landing_gears = LandingGears(self, leg_length)
        self.systems = Systems(self)

        # Logical components
        self.aerodynamics = Aerodynamics(self, hld_type, hld_conf_to, hld_conf_ld)
        self.propulsion = Propulsion(self)
        self.geometry = Geometry(self)
        self.mass = Mass(self, d_owe)

        self.missions = Missions(self, holding_time, reserve_fuel_ratio, diversion_range)
        self.operations = Operations(self, hld_conf_to, kvs1g_req_to, s2_path_req_to, hld_conf_ld, kvs1g_ld,
                                           tofl_req, app_speed_req, vz_mcl_req, vz_mcr_req, oei_path_req, oei_altp_req)
        self.economics = Economics(self, d_cost)

    def __iter__(self):
        public = [value for value in self.__dict__.values() if issubclass(type(value),Component)]
        return iter(public)

    def compute_cabin(self, n_pax, range):
        self.design_range = range
        self.cabin.n_pax = n_pax
        self.cabin.eval_geometry()
        self.mass.nominal_payload = 105. * n_pax
        self.mass.max_payload = 120. * n_pax
        return self.cabin.width, self.cabin.length, self.mass.nominal_payload, self.mass.max_payload

    def compute_geometry(self, cabin_width, cabin_length, wing_area, engine_slst):
        self.cabin.width = cabin_width
        self.cabin.length = cabin_length
        self.wing.area = wing_area
        self.nacelles.engine_slst = engine_slst
        self.geometry.solve()
        geometry = {"dummy_object":None}
        return geometry

    def compute_owe(self, geometry, mtow_i, mzfw_i, mlw_i, d_owe):
        self.mass.mtow = mtow_i
        self.mass.mzfw = mzfw_i
        self.mass.mlw = mlw_i
        self.mass.d_owe = d_owe
        self.mass.eval_equiped_mass()
        self.mass.owe = self.mass.d_owe
        for comp in self:
            self.mass.owe += comp.mass
        return self.mass.owe

    def compute_characteristic_weights(self, owe, nominal_payload, max_payload, nominal_fuel):
        self.mass.mtow = owe + nominal_payload + nominal_fuel
        self.mass.mzfw = owe + max_payload
        self.mass.mlw = 1.07 * self.mass.mzfw
        self.mass.mfw = 803. * self.tank.fuel_volume
        return self.mass.mtow, self.mass.mzfw, self.mass.mlw, self.mass.mfw

    def compute_nominal_mission(self, mtow, range):
        self.design_range = range
        self.missions.nominal.altp = self.cruise_altp
        self.missions.nominal.mach = self.cruise_mach
        self.missions.nominal.range = self.design_range
        self.missions.nominal.tow = mtow
        self.missions.nominal.eval()
        return self.missions.nominal.fuel_total, self.missions.nominal.fuel_reserve

    def compute_other_missions(self, max_payload, mtow, mfw):
        self.mass.max_payload = max_payload
        self.mass.mtow = mtow
        self.mass.mfw = mfw
        self.missions.eval_payload_range_solver() # Solver inside
        return self.missions.cost.fuel_block, self.missions.cost.time_block

    def compute_other_performances(self, mtow, mlw, cost_fuel_block, cost_time_bloc):
        self.mass.mtow = mtow
        self.mass.mlw = mlw
        self.operations.eval()
        self.missions.cost.fuel_block = cost_fuel_block
        self.missions.cost.time_block = cost_time_bloc
        self.economics.eval()
        return self.economics.cash_op_cost, self.economics.direct_op_cost

    def print_airplane_data(self):
        # Print some relevant output
        #------------------------------------------------------------------------------------------------------
        print("")
        print("Design parameters")
        print("-------------------------------------------------------")
        print("Design range = "+"%.0f"%unit.NM_m(self.design_range)+" NM")
        print("Cost range = "+"%.0f"%unit.NM_m(self.cost_range)+" NM")
        print("Cruise Mach = "+"%.2f"%self.cruise_mach)
        print("Cruise pressure altitude = "+"%.0f"%unit.ft_m(self.cruise_altp)+" ft")
        print("")
        print("Component design")
        print("-------------------------------------------------------")
        print("Engine Sea Level Static Thrust = "+"%.1f"%unit.kN_N(self.nacelles.engine_slst)+" kN")
        print("Engine By Pass Ratio = "+"%.1f"%self.nacelles.engine_bpr)
        print("Nacelle diameter = "+"%.2f"%self.nacelles.diameter+" m")
        print("Fan cowl length = "+"%.2f"%self.nacelles.length+" m")
        print("Spanwise position = "+"%.2f"%self.nacelles.span_position+" m")
        print("Groun clearence = "+"%.2f"%self.nacelles.ground_clearence+" m")
        print("")
        print("Cabin length = "+"%.2f"%self.cabin.length+" m2")
        print("Cabin width = "+"%.2f"%self.cabin.width+" m")
        print("")
        print("Fuselage length = "+"%.2f"%self.fuselage.length+" m2")
        print("Fuselage height = "+"%.2f"%self.fuselage.height+" m")
        print("Fuselage width = "+"%.2f"%self.fuselage.width+" m")
        print("")
        print("Wing area = "+"%.2f"%self.wing.area+" m2")
        print("Wing aspect ratio = "+"%.2f"%self.wing.aspect_ratio)
        print("Wing taper ratio = "+"%.2f"%self.wing.taper_ratio)
        print("Wing sweep angle = "+"%.1f"%unit.deg_rad(self.wing.sweep25)+" deg")
        print("Wing spanwise kink position = "+"%.1f"%self.wing.kink_loc[1]+" m")
        print("")
        print("Fuel tank volume = "+"%.2f"%self.tank.fuel_volume+" m3")
        print("")
        print("HTP area = "+"%.2f"%self.htp.area+" m2")
        print("HTP lever arm = "+"%.2f"%self.htp.lever_arm+" m")
        print("")
        print("VTP area = "+"%.2f"%self.vtp.area+" m2")
        print("VTP lever arm = "+"%.2f"%self.vtp.lever_arm+" m")
        print("")
        print("Landing gear leg length = "+"%.2f"%self.landing_gears.leg_length+" m")
        print("Landing spanwise position = "+"%.2f"%self.landing_gears.attachment_loc[1]+" m")
        print("Landing longitudinal position = "+"%.2f"%self.landing_gears.attachment_loc[0]+" m")
        print("")
        print("Characteristic eights and breakdown")
        print("-------------------------------------------------------")
        print("MTOW = "+"%.0f"%self.mass.mtow+" kg")
        print("MLW = "+"%.0f"%self.mass.mlw+" kg")
        print("MZFW = "+"%.0f"%self.mass.mzfw+" kg")
        print("OWE = "+"%.0f"%self.mass.owe+" kg")
        print("MFW = "+"%.0f"%self.mass.mfw+" kg")
        print("Nacelles mass = "+"%.0f"%self.nacelles.mass+" kg")
        print("Fuselage mass = "+"%.0f"%self.fuselage.mass+" kg")
        print("Cabin mass = "+"%.0f"%self.cabin.mass+" kg")
        print("wing mass = "+"%.0f"%self.wing.mass+" kg")
        print("HTP mass = "+"%.0f"%self.htp.mass+" kg")
        print("VTP mass = "+"%.0f"%self.vtp.mass+" kg")
        print("Landing_gears mass = "+"%.0f"%self.landing_gears.mass+" kg")
        print("Systems mass = "+"%.0f"%self.systems.mass+" kg")
        print("")
        print("Start of cruise analysis")
        print("-------------------------------------------------------")
        print("Temperature shift vs ISA = "+"%.1f"%self.missions.crz_disa+" K")
        print("Cruise altitude = "+"%.0f"%unit.ft_m(self.missions.crz_altp)+" ft")
        print("Cruise Mach number = "+"%.2f"%self.missions.crz_mach)
        print("Cruise initial weight = "+"%.0f"%self.missions.crz_mass+" kg")
        print("Cruise true air speed = "+"%.0f"%unit.kt_mps(self.missions.crz_tas)+" kt")
        print("Cruise initial lift coefficient = "+"%.2f"%self.missions.crz_cz)
        print("Cruise initial L/D = "+"%.2f"%self.missions.crz_lod+" ")
        print("Cruise initial total thrust = "+"%.1f"%unit.kN_N(self.missions.crz_thrust)+" kN")
        print("Cruise initial propulsive power = "+"%.3f"%unit.MW_W(self.missions.crz_propulsive_power)+" MW")
        print("Cruise initial specific fuel consumption = "+"%.3f"%unit.convert_to("kg/daN/h",self.missions.crz_sfc)+" kg/daN/h")
        print("")
        print("All missions")
        print("-------------------------------------------------------")
        print("Nominal mission range = "+"%.0f"%unit.NM_m(self.missions.nominal.range)+" NM")
        print("Nominal mission payload = "+"%.0f"%self.missions.nominal.payload+" kg")
        print("Nominal mission time block = "+"%.0f"%unit.h_s(self.missions.nominal.time_block)+" h")
        print("Nominal mission fuel block = "+"%.0f"%self.missions.nominal.fuel_block+" kg")
        print("Nominal mission fuel total = "+"%.0f"%self.missions.nominal.fuel_total+" kg")
        print("")
        print("Max payload mission range = "+"%.0f"%unit.NM_m(self.missions.max_payload.range)+" NM")
        print("Max payload mission payload = "+"%.0f"%self.missions.max_payload.payload+" kg")
        print("Max payload mission residual = "+"%.4f"%self.missions.max_payload.residual)
        print("")
        print("Max fuel mission range = "+"%.0f"%unit.NM_m(self.missions.max_fuel.range)+" NM")
        print("Max fuel mission payload = "+"%.0f"%self.missions.max_fuel.payload+" kg")
        print("Max fuel mission residual = "+"%.4f"%self.missions.max_fuel.residual)
        print("")
        print("Zero payload mission range = "+"%.0f"%unit.NM_m(self.missions.zero_payload.range)+" NM")
        print("Zero payload mission payload = "+"%.0f"%self.missions.zero_payload.payload+" kg")
        print("Zero payload mission residual = "+"%.4f"%self.missions.zero_payload.residual)
        print("")
        print("Cost mission tow = "+"%.0f"%self.missions.cost.tow+" kg")
        print("Cost mission fuel_block = "+"%.0f"%self.missions.cost.fuel_block+" kg")
        print("Cost mission residual = "+"%.4f"%self.missions.cost.residual)
        print("")
        print("All performances")
        print("-------------------------------------------------------")
        print("Take off field length required = "+"%.1f"%self.operations.take_off.tofl_req+" m")
        print("Take off field length effective = "+"%.1f"%self.operations.take_off.tofl_eff+" m")
        print("Take off kvs1g effective = "+"%.3f"%self.operations.take_off.kvs1g_eff)
        print("Take off path effective = "+"%.3f"%self.operations.take_off.s2_path_eff)
        print("Take off limitation = "+self.operations.take_off.limit)
        print("")
        print("Approach speed required = "+"%.1f"%unit.kt_mps(self.operations.approach.app_speed_req)+" kt")
        print("Approach speed effective = "+"%.1f"%unit.kt_mps(self.operations.approach.app_speed_eff)+" kt")
        print("")
        print("Vertical speed required MCL = "+"%.1f"%unit.ftpmin_mps(self.operations.mcl_ceiling.vz_req)+" ft/min")
        print("Vertical speed effective MCL = "+"%.1f"%unit.ftpmin_mps(self.operations.mcl_ceiling.vz_eff)+" ft/min")
        print("")
        print("Vertical speed required MCR = "+"%.1f"%unit.ftpmin_mps(self.operations.mcr_ceiling.vz_req)+" ft/min")
        print("Vertical speed effective MCR = "+"%.1f"%unit.ftpmin_mps(self.operations.mcr_ceiling.vz_eff)+" ft/min")
        print("")
        print("One engine required altitude = "+"%.0f"%unit.ft_m(self.operations.oei_ceiling.altp_req)," ft")
        print("One engine path required = "+"%.3f"%self.operations.oei_ceiling.path_req)
        print("One engine path effective = "+"%.3f"%self.operations.oei_ceiling.path_eff)
        print("One engine Mach number = "+"%.3f"%self.operations.oei_ceiling.mach)
        print("")
        print("Economics")
        print("-------------------------------------------------------")
        print("Utilization = "+"%.1f"%self.economics.utilization+" trip/year")
        print("Cash Operating Cost = "+"%.1f"%self.economics.cash_op_cost+" $/trip")
        print("Direct Operating Cost = "+"%.1f"%self.economics.direct_op_cost+" $/trip")

    def view_3d(self):
        """
        Build a 3 views drawing of the airplane
        """
        window_title = "MARILib extract"
        plot_title = "3 View"

        # Drawing_ box
        #-----------------------------------------------------------------------------------------------------------
        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(window_title)
        fig.suptitle(plot_title, fontsize=14)
        axes.set_aspect('equal', 'box')
        plt.plot(np.array([0,100,100,0,0]), np.array([0,0,100,100,0]))      # Draw a square box of 100m side

        xTopView = 50 - (self.wing.mac_loc[0] + 0.25*self.wing.mac)
        yTopView = 50

        xSideView = 50 - (self.wing.mac_loc[0] + 0.25*self.wing.mac)
        ySideView = 82

        xFrontView = 50
        yFrontView = 10

        ref = {"xy":[xTopView,yTopView],"yz":[xFrontView,yFrontView],"xz":[xSideView,ySideView]}

        l0w, l1, l2, l3, l4, l5, high  =  0, 1, 2, 3, 4, 5, 6

        # Draw components
        #-----------------------------------------------------------------------------------------------------------
        zframe = {"xy":{"body":l2, "wing":l1, "htp":l1, "vtp":l3},      # top
                  "yz":{"body":l4, "wing":l3, "htp":l1, "vtp":l2},      # front
                  "xz":{"body":l2, "wing":l3, "htp":l3, "vtp":l1}}      # side

        for comp in self:
            typ,data = comp.sketch_3view()
            if data is not None:
                if typ in ["body","wing","htp","vtp"]:
                    for view in ["xy","yz","xz"]:
                        plt.fill(ref[view][0]+data[view][0:,0], ref[view][1]+data[view][0:,1], color="white", zorder=zframe[view][typ])    # draw mask
                        plt.plot(ref[view][0]+data[view][0:,0], ref[view][1]+data[view][0:,1], color="grey", zorder=zframe[view][typ])     # draw contour

        # Draw nacelles
        #-----------------------------------------------------------------------------------------------------------
        #                                  top        front     side
        znac = {"xy":l0w,  "yz":l4,  "xz":high}

        for comp in self:
            if issubclass(type(comp),Nacelles):
                typ,nacelle = comp.sketch_3view(side=1)
                for view in ["xy","yz","xz"]:
                    plt.fill(ref[view][0]+nacelle[view][0:,0], ref[view][1]+nacelle[view][0:,1], color="white", zorder=znac[view])    # draw mask
                    plt.plot(ref[view][0]+nacelle[view][0:,0], ref[view][1]+nacelle[view][0:,1], color="grey", zorder=znac[view])     # draw contour
                plt.plot(ref["yz"][0]+nacelle["disk"][0:,0], ref["yz"][1]+nacelle["disk"][0:,1], color="grey", zorder=znac["yz"])     # draw contour

                typ,nacelle = comp.sketch_3view(side=-1)
                for view in ["xy","yz","xz"]:
                    plt.fill(ref[view][0]+nacelle[view][0:,0], ref[view][1]+nacelle[view][0:,1], color="white", zorder=znac[view])    # draw mask
                    plt.plot(ref[view][0]+nacelle[view][0:,0], ref[view][1]+nacelle[view][0:,1], color="grey", zorder=znac[view])     # draw contour
                plt.plot(ref["yz"][0]+nacelle["disk"][0:,0], ref["yz"][1]+nacelle["disk"][0:,1], color="grey", zorder=znac["yz"])     # draw contour

        plt.ylabel('(m)')
        plt.xlabel('(m)')

        plt.show()
        return


