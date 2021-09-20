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

    def print(self):
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



#-----------------------------------------------------------------------------------------------------------------------
# Test sequence
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    ap = Airplane()


# MDA
#-----------------------------------------
    process.geometry_solver(ap)

    process.mass_mission_adaptation(ap)            # Solver inside

    # ap.mass.eval_equiped_mass()
    #
    # ap.mass.eval_characteristic_mass()
    #
    # ap.missions.eval_nominal_mission()

    ap.missions.eval_payload_range_solver() # Solver inside

    # ap.missions.eval_cost_mission_solver()

    ap.operations.eval()

    ap.economics.eval()


# MDF
#-----------------------------------------
    var = ["aircraft.nacelles.engine_slst",
           "aircraft.wing.area"]               # Main design variables

    var_bnd = [[unit.N_kN(80.), unit.N_kN(200.)],       # Design space area where to look for an optimum solution
               [100., 200.]]

    # Operational constraints definition
    cst = ["aircraft.operations.take_off.tofl_req - aircraft.operations.take_off.tofl_eff",
           "aircraft.operations.approach.app_speed_req - aircraft.operations.approach.app_speed_eff",
           "aircraft.operations.mcl_ceiling.vz_eff - aircraft.operations.mcl_ceiling.vz_req",
           "aircraft.operations.mcr_ceiling.vz_eff - aircraft.operations.mcr_ceiling.vz_req",
           "aircraft.operations.oei_ceiling.path_eff - aircraft.operations.oei_ceiling.path_req",
           "aircraft.mass.mfw - aircraft.missions.nominal.fuel_total"]

    # Magnitude used to scale constraints
    cst_mag = ["aircraft.operations.take_off.tofl_req",
               "aircraft.operations.approach.app_speed_req",
               "unit.mps_ftpmin(100.)",
               "unit.mps_ftpmin(100.)",
               "aircraft.operations.oei_ceiling.path_req",
               "aircraft.mass.mfw"]

    # Optimization criteria
    crt = "aircraft.mass.mtow"

    # Perform an MDF optimization process
    opt = process.Optimizer()
    # method = 'trust-constr'
    method = 'optim2d_poly'  # 'optim2d'
    opt.mdf(ap, var,var_bnd, cst,cst_mag, crt, method)
    algo_points = None

# Design space exploration
# ---------------------------------------------------------------------------------------------------------------------
    step = [0.05,
            0.05]    # Relative grid step

    data = [["Thrust", "daN", "%8.1f", var[0]+"/10."],
            ["Wing_area", "m2", "%8.1f", var[1]],
            ["Wing_span", "m", "%8.1f", "aircraft.wing.span"],
            ["MTOW", "kg", "%8.1f", "aircraft.mass.mtow"],
            ["MLW", "kg", "%8.1f", "aircraft.mass.mlw"],
            ["OWE", "kg", "%8.1f", "aircraft.mass.owe"],
            ["Cruise_LoD", "no_dim", "%8.1f", "aircraft.missions.crz_lod"],
            ["Cruise_SFC", "kg/daN/h", "%8.4f", "aircraft.missions.crz_sfc"],
            ["TOFL", "m", "%8.1f", "aircraft.operations.take_off.tofl_eff"],
            ["App_speed", "kt", "%8.1f", "unit.kt_mps(aircraft.operations.approach.app_speed_eff)"],
            ["OEI_path", "%", "%8.1f", "aircraft.operations.oei_ceiling.path_eff*100"],
            ["Vz_MCL", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.operations.mcl_ceiling.vz_eff)"],
            ["Vz_MCR", "ft/min", "%8.1f", "unit.ftpmin_mps(aircraft.operations.mcr_ceiling.vz_eff)"],
            ["FUEL", "kg", "%8.1f", "aircraft.mass.mfw"],
            ["Cost_Block_fuel", "kg", "%8.1f", "aircraft.missions.cost.fuel_block"],
            ["Std_op_cost", "$/trip", "%8.1f", "aircraft.economics.std_op_cost"],
            ["Cash_op_cost", "$/trip", "%8.1f", "aircraft.economics.cash_op_cost"],
            ["Direct_op_cost", "$/trip", "%8.1f", "aircraft.economics.direct_op_cost"]]

    file = "table.txt"
    proc = "process.mda"

    # res = process.eval_this(ac,var)                                  # This function allows to get the values of a list of addresses in the Aircraft
    res = util.explore_design_space(ap, var, step, data, file, proc)      # Build a set of experiments using above config data and store it in a file

    field = 'MTOW'                                                                  # Optimization criteria, keys are from data
    other = ['MLW']                                                                 # Additional useful data to show
    const = ['TOFL', 'App_speed', 'OEI_path', 'Vz_MCL', 'Vz_MCR', 'FUEL']    # Constrained performances, keys are from data
    bound = np.array(["ub", "ub", "lb", "lb", "lb", "lb"])                    # ub: upper bound, lb: lower bound
    color = ['red', 'blue', 'violet', 'orange', 'brown', 'black']         # Constraint color in the graph
    limit = [ap.operations.take_off.tofl_req,
             unit.kt_mps(ap.operations.approach.app_speed_req),
             unit.pc_no_dim(ap.operations.oei_ceiling.path_req),
             unit.ftpmin_mps(ap.operations.mcl_ceiling.vz_req),
             unit.ftpmin_mps(ap.operations.mcr_ceiling.vz_req),
             ap.missions.nominal.fuel_total]                        # Limit values

    util.draw_design_space(file, res, other, field, const, color, limit, bound,
                              optim_points=algo_points) # Used stored result to build a graph of the design space












# Utils
#-----------------------------------------
    ap.missions.payload_range()

    ap.print()

    ap.view_3d()



