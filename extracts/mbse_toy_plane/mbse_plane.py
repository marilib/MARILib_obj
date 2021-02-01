#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 16 17:18:19 2021
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
import matplotlib.pyplot as plt

import unit, util



#-----------------------------------------------------------------------------------------------------------------------
# Airplane component
#-----------------------------------------------------------------------------------------------------------------------

class Airplane(object):
    def __init__(self, cruise_mach=0.78, design_range=unit.m_NM(3000), cost_range=unit.m_NM(500),
                 n_pax=150, n_aisle=1, n_front=6,
                 wing_area=122, wing_aspect_ratio=9, wing_taper_ratio=0.20, wing_toc_ratio=0.12, wing_sweep25=unit.rad_deg(25), wing_dihedral=unit.rad_deg(5), hld_type=9,
                 htp_aspect_ratio=5, htp_taper_ratio=0.35, htp_toc_ratio=0.10, htp_sweep25=unit.rad_deg(30), htp_dihedral=unit.rad_deg(5), volume=0.94,
                 vtp_aspect_ratio=1.7, vtp_taper_ratio=0.4, vtp_toc_ratio=0.10, vtp_sweep25=unit.rad_deg(30), thrust_volume=0.4,
                 engine_slst=unit.N_kN(120.), engine_bpr=10, z_ratio = 0.55,
                 leg_length=3.7,
                 holding_time=unit.s_min(30), reserve_fuel_ratio=0.05, diversion_range=unit.m_NM(200),
                 hld_conf_to=0.3, kvs1g_req_to=1.13, s2_path_req_to=0.024, hld_conf_ld=1., kvs1g_ld=1.23,
                 tofl_req=2300, app_speed_req=unit.mps_kt(137), vz_mcl_req=unit.mps_ftpmin(300), vz_mcr_req=unit.mps_ftpmin(0),
                 oei_path_req=0.011, oei_altp_req=unit.m_ft(15000)):

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
        self.mass = Mass(self)

        self.missions = Missions(self, holding_time, reserve_fuel_ratio, diversion_range)
        self.operations = Operations(self, hld_conf_to, kvs1g_req_to, s2_path_req_to, hld_conf_ld, kvs1g_ld,
                                           tofl_req, app_speed_req, vz_mcl_req, vz_mcr_req, oei_path_req, oei_altp_req)
        self.economics = Economics(self)

    def __iter__(self):
        public = [value for value in self.__dict__.values() if issubclass(type(value),Component)]
        return iter(public)

    def view_3d(self, window_title):
        """
        Build a 3 views drawing of the airplane
        """
        plot_title = "MBSE toy plane"

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

        plt.show()
        return

    def print(self):
        # Print some relevant output
        #------------------------------------------------------------------------------------------------------
        print("")
        print("-------------------------------------------------------")
        print("Design range = "+"%.0f"%unit.NM_m(self.design_range)+" NM")
        print("Cost range = "+"%.0f"%unit.NM_m(self.cost_range)+" NM")
        print("Cruise Mach = "+"%.2f"%self.cruise_mach)
        print("Cruise pressure altitude = "+"%.0f"%unit.ft_m(self.cruise_altp)+" ft")
        print("")
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
        print("-------------------------------------------------------")
        print("MTOW = "+"%.0f"%self.mass.mtow+" kg")
        print("MLW = "+"%.0f"%self.mass.mlw+" kg")
        print("MZFW = "+"%.0f"%self.mass.mzfw+" kg")
        print("OWE = "+"%.0f"%self.mass.owe+" kg")
        print("MFW = "+"%.0f"%self.mass.mfw+" kg")
        print("Nacelles mass = "+"%.0f"%self.nacelles.mass+" kg")
        print("Fuselage mass = "+"%.0f"%self.fuselage.mass+" kg")
        print("wing mass = "+"%.0f"%self.wing.mass+" kg")
        print("HTP mass = "+"%.0f"%self.htp.mass+" kg")
        print("VTP mass = "+"%.0f"%self.vtp.mass+" kg")
        print("Landing_gears mass = "+"%.0f"%self.landing_gears.mass+" kg")
        print("Systems mass = "+"%.0f"%self.systems.mass+" kg")
        print("")
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
        print("Cost mission range = "+"%.0f"%self.missions.cost.tow+" kg")
        print("Cost mission fuel_block = "+"%.0f"%self.missions.cost.fuel_block+" kg")
        print("Cost mission residual = "+"%.4f"%self.missions.zero_payload.residual)
        print("")
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
        print("")
        print("-------------------------------------------------------")
        print("Cash Operating Cost = "+"%.1f"%self.economics.cash_op_cost+" $/trip")
        print("Direct Operating Cost = "+"%.1f"%self.economics.direct_op_cost+" $/trip")



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

    def get_this_shape(self, name): # TODO: is the docstring up to date ?
        """Contour curves for 3 view drawing
        nose1 : modern nose (A220, A350, 787)
        nose3 : classical Airbus nose
        nose2 : symetrical nose
        cone1 : classical tail cone
        cone2 : symetrical cone
        section : circle
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

        nose,cone,section = self.get_this_shape(["nose1","cone1","sec1"])

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
        mtow = self.airplane.mass.mtow
        mzfw = self.airplane.mass.mzfw
        aerodynamics = self.airplane.aerodynamics

        hld_conf_ld = aerodynamics.hld_conf_ld

        cz_max_ld,cz0 = aerodynamics.wing_high_lift(hld_conf_ld)

        A = 32*self.area**1.1
        B = 4.*self.span**2 * np.sqrt(mtow*mzfw)
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
        nacelles = self.airplane.nacelles

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

        section, = self.get_this_shape(["sec1"])

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
        mtow = self.airplane.mass.mtow
        mlw = self.airplane.mass.mlw
        self.mass = (0.015*mtow**1.03 + 0.012*mlw)

    def sketch_3view(self, side=None):
        return "ldg", {}


class Systems(Component):
    def __init__(self, airplane):
        super(Systems, self).__init__(airplane)

    def eval_geometry(self):
        pass

    def eval_mass(self):
        mtow = self.airplane.mass.mtow
        self.mass = 0.545*mtow**0.8    # global mass of all systems

    def sketch_3view(self, side=None):
        return "ldg", {}


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

    def eval_tail_areas(self):
        self.airplane.htp.eval_area()
        self.airplane.vtp.eval_area()

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

        mtow_init = 5. * 110. * airplane.cabin.n_pax
        mzfw_init = 0.75 * mtow_init
        mlw_init = 1.07 * mzfw_init
        owe_init = 0.5 * mtow_init
        mwe_init = 0.5 * mtow_init

        self.mtow = mtow_init
        self.mlw = mlw_init
        self.mzfw = mzfw_init
        self.owe = owe_init
        self.mwe = mwe_init
        self.mfw = None

        self.nominal_payload = None
        self.max_payload = None

    def eval(self):
        """Mass computations
        """
        for comp in self.airplane:
            comp.eval_mass()

    def eval_characteristic_mass(self):
        self.owe = 0.
        for comp in self.airplane:
            self.owe += comp.mass
        self.nominal_payload = 105. * self.airplane.cabin.n_pax
        self.max_payload = 120. * self.airplane.cabin.n_pax
        self.mtow = self.owe + self.nominal_payload + self.airplane.missions.nominal.fuel_total
        self.mzfw = self.owe + self.max_payload
        self.mlw = 1.07 * self.mzfw
        self.mwe = self.owe - self.airplane.cabin.m_op_item
        self.mfw = 803. * self.airplane.tank.fuel_volume


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
        re = util.reynolds_number(pamb, tamb, mach)

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
        pamb,tamb,tstd,dtodz = util.atmosphere(altp, disa, full_output=True)
        mach = self.get_mach(pamb,speed_mode,speed)

        thrust = propulsion.unitary_thrust(pamb,tamb,mach,rating)
        sfc = propulsion.unitary_sc(pamb,tamb,mach,thrust)
        fn = kfn * thrust * (propulsion.n_engine - nei)
        ff = sfc * fn
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
        vz = slope * mach * util.sound_speed(tamb)
        acc = (acc_factor-1.)*g*slope
        if full_output:
            return slope,vz,fn,ff,acc,cz,cx,pamb,tamb
        else:
            return slope,vz


class Missions(Flight):
    def __init__(self, airplane, holding_time, reserve_fuel_ratio, diversion_range):
        super(Missions, self).__init__(airplane)

        self.nominal = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.max_payload = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.max_fuel = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.zero_payload = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)
        self.cost = Breguet(airplane, holding_time, reserve_fuel_ratio, diversion_range)

    def eval_nominal_mission(self):
        """Compute missions
        """
        self.nominal.altp = self.airplane.cruise_altp
        self.nominal.mach = self.airplane.cruise_mach
        self.nominal.range = self.airplane.design_range
        self.nominal.tow = self.airplane.mass.mtow

        self.nominal.eval()

    def eval_max_payload_mission(self):
        """Compute missions
        """
        self.max_payload.altp = self.airplane.cruise_altp
        self.max_payload.mach = self.airplane.cruise_mach
        self.max_payload.tow = self.airplane.mass.mtow

        self.max_payload.eval()
        self.max_payload.residual = self.airplane.mass.max_payload - self.max_payload.payload       # INFO: range must drive residual to zero

    def eval_max_fuel_mission(self):
        """Compute missions
        """
        self.max_fuel.altp = self.airplane.cruise_altp
        self.max_fuel.mach = self.airplane.cruise_mach
        self.max_fuel.tow = self.airplane.mass.mtow

        self.max_fuel.eval()
        self.max_fuel.residual = self.airplane.mass.mfw - self.max_fuel.fuel_total       # INFO: range must drive residual to zero

    def eval_zero_payload_mission(self):
        """Compute missions
        """
        self.zero_payload.altp = self.airplane.cruise_altp
        self.zero_payload.mach = self.airplane.cruise_mach
        self.zero_payload.tow = self.airplane.mass.owe + self.airplane.mass.mfw

        self.zero_payload.eval()
        self.zero_payload.residual = self.airplane.mass.mfw - self.zero_payload.fuel_total       # INFO: range must drive residual to zero

    def eval_cost_mission(self):
        """Compute missions
        """
        self.cost.altp = self.airplane.cruise_altp
        self.cost.mach = self.airplane.cruise_mach
        self.cost.range = self.airplane.cost_range

        self.cost.eval()
        self.cost.residual = self.airplane.mass.nominal_payload - self.cost.payload       # INFO: tow must drive residual to zero

    def eval_payload_range(self):
        """Compute missions
        """
        self.eval_max_payload_mission()
        self.eval_max_fuel_mission()
        self.eval_zero_payload_mission()
        self.eval_cost_mission()


class Breguet(Flight):
    def __init__(self, airplane, holding_time, reserve_fuel_ratio, diversion_range):
        super(Breguet, self).__init__(airplane)

        self.disa = 0.      # Mean cruise temperature shift
        self.altp = None    # Mean cruise altitude
        self.mach = None    # Cruise mach number

        range_init = airplane.design_range
        tow_init = 5. * 110. * airplane.cabin.n_pax
        total_fuel_init = 0.2 * tow_init

        self.range = range_init             # Mission distance
        self.tow = tow_init                 # Take Off Weight
        self.payload = None                 # Payload Weight
        self.time_block = None              # Mission block duration
        self.fuel_block = None              # Mission block fuel consumption
        self.fuel_reserve = None            # Mission reserve fuel
        self.fuel_total = total_fuel_init   # Mission total fuel

        self.holding_time = holding_time
        self.reserve_fuel_ratio = reserve_fuel_ratio
        self.diversion_range = diversion_range

    def holding(self,time,mass,altp,mach,disa):
        """Holding fuel
        """
        g = 9.80665
        pamb,tamb = util.atmosphere(altp, disa)
        sfc, lod = self.level_flight(pamb,tamb,mach,mass)
        fuel = sfc*(mass*g/lod)*time
        return fuel

    def breguet_range(self,range,tow,altp,mach,disa):
        """Breguet range equation
        """
        g = 9.80665
        pamb,tamb = util.atmosphere(altp, disa)
        tas = mach * util.sound_speed(tamb)
        time = 1.09*(range/tas)
        sfc, lod = self.level_flight(pamb,tamb,mach,tow)
        val = tow*(1-np.exp(-(sfc*g*range)/(tas*lod)))
        return val,time

    def eval(self):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        g = 9.80665

        disa = self.disa
        altp = self.altp
        mach = self.mach

        range = self.range
        tow = self.tow

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
        self.payload = self.tow - self.airplane.mass.owe - self.fuel_total

        #-----------------------------------------------------------------------------------------------------------
        return


class Operations(Flight):
    def __init__(self, airplane, hld_conf_to, kvs1g_to, s2_min_path_to, hld_conf_ld, kvs1g_ld,
                                 tofl_req, app_speed_req, vz_mcl_req, vz_mcr_req, oei_path_req, oei_altp_req):
        super(Operations, self).__init__(airplane)

        self.take_off = TakeOff(airplane, hld_conf_to, kvs1g_to, s2_min_path_to, tofl_req)
        self.approach = Approach(airplane, hld_conf_ld, kvs1g_ld, app_speed_req)
        self.mcl_ceiling = ClimbCeiling(airplane, rating="MCL", speed_mode="cas", vz_req=vz_mcl_req)
        self.mcr_ceiling = ClimbCeiling(airplane, rating="MCR", speed_mode="mach", vz_req=vz_mcr_req)
        self.oei_ceiling = OeiCeiling(airplane, rating="MCN", speed_mode="mach", path_req=oei_path_req, altp_req=oei_altp_req)

    def eval_take_off(self):
        """Compute performances
        """
        self.take_off.disa = 15.
        self.take_off.altp = unit.m_ft(0.)
        self.take_off.tow = self.airplane.mass.mtow
        self.take_off.eval()

    def eval_approach(self):
        """Compute performances
        """
        self.approach.disa = 0.
        self.approach.altp = unit.m_ft(0.)
        self.approach.lw = self.airplane.mass.mlw
        self.approach.eval()

    def eval_climb_ceiling(self):
        """Compute performances
        """
        self.mcl_ceiling.disa = 15.
        self.mcl_ceiling.altp = self.airplane.cruise_altp
        self.mcl_ceiling.mach = self.airplane.cruise_mach
        self.mcl_ceiling.mass = 0.97 * self.airplane.mass.mtow
        self.mcl_ceiling.eval()

        self.mcr_ceiling.disa = 15.
        self.mcr_ceiling.altp = self.airplane.cruise_altp
        self.mcr_ceiling.mach = self.airplane.cruise_mach
        self.mcr_ceiling.mass = 0.97 * self.airplane.mass.mtow
        self.mcr_ceiling.eval()

    def eval_oei_ceiling(self):
        """Compute performances
        """
        self.oei_ceiling.disa = 15.
        self.oei_ceiling.mass = 0.97 * self.airplane.mass.mtow
        self.oei_ceiling.eval()

    def eval(self):
        """Compute performances
        """
        self.eval_take_off()
        self.eval_approach()
        self.eval_climb_ceiling()
        self.eval_oei_ceiling()


class TakeOff(Flight):
    """Take Off Field Length
    """
    def __init__(self, airplane, hld_conf, kvs1g_req, s2_path_req, tofl_req):
        super(TakeOff, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.tow = None
        self.tofl_eff = None      # INFO: tofl_eff must be lower or equal to tofl_req
        self.kvs1g_eff = None
        self.s2_path_eff = None
        self.limit = None

        self.v2 = None
        self.mach2 = None

        self.tofl_req = tofl_req
        self.kvs1g_req = kvs1g_req
        self.s2_path_req = s2_path_req
        self.hld_conf = hld_conf

    def eval(self):
        """Take off field length and climb path with eventual kVs1g increase to recover min regulatory slope
        """
        disa = self.disa
        altp = self.altp
        mass = self.tow

        s2_min_path = self.s2_path_req
        kvs1g = self.kvs1g_req
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

        self.tofl_eff = tofl
        self.kvs1g_eff = kvs1g
        self.s2_path_eff = s2_path
        self.limit = limitation
        self.v2 = cas
        self.mach2 = mach
        return

    def take_off(self,kvs1g,altp,disa,mass,hld_conf,rating,kfn):
        """Take off field length and climb path at 35 ft depending on stall margin (kVs1g)
        """
        aerodynamics = self.airplane.aerodynamics
        propulsion = self.airplane.propulsion

        czmax,cz0 = aerodynamics.wing_high_lift(hld_conf)

        pamb,tamb = util.atmosphere(altp, disa)
        rho,sig = util.air_density(pamb, tamb)

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
    def __init__(self, airplane, hld_conf, kvs1g, app_speed_req):
        super(Approach, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.lw = None
        self.app_speed_eff = None      # INFO: app_speed_eff must be lower or equal to app_speed_req

        self.app_speed_req = app_speed_req
        self.hld_conf = hld_conf
        self.kvs1g = kvs1g

    def eval(self):
        """Minimum approach speed (VLS)
        """
        aerodynamics = self.airplane.aerodynamics

        disa = self.disa
        altp = self.altp
        mass = self.lw

        hld_conf = self.hld_conf
        kvs1g = self.kvs1g

        pamb,tamb = util.atmosphere(altp, disa)
        czmax,cz0 = aerodynamics.wing_high_lift(hld_conf)
        cz = czmax / kvs1g**2
        mach = self.speed_from_lift(pamb,tamb,cz,mass)
        vapp = self.get_speed(pamb,"cas",mach)
        self.app_speed_eff = vapp
        return


class ClimbCeiling(Flight):
    """Propulsion ceiling in MCL rating
    """
    def __init__(self, airplane, rating, speed_mode, vz_req):
        super(ClimbCeiling, self).__init__(airplane)

        self.disa = None
        self.altp = None
        self.mach = None
        self.mass = None
        self.vz_eff = None      # INFO: vz_eff must be higher or equal to vz_req

        self.vz_req = vz_req
        self.rating = rating
        self.speed_mode = speed_mode

    def eval(self):
        """Residual climb speed in MCL rating
        """
        disa = self.disa
        altp = self.altp
        mach = self.mach
        mass = self.mass

        speed_mode = self.speed_mode
        rating = self.rating
        kfn = 1.
        nei = 0
        pamb,tamb = util.atmosphere(altp, disa)
        speed = self.get_speed(pamb,self.speed_mode,mach)
        path,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
        self.vz_eff = vz
        return


class OeiCeiling(Flight):
    """One engine ceiling in MCN rating
    """
    def __init__(self, airplane, rating, speed_mode, path_req, altp_req):
        super(OeiCeiling, self).__init__(airplane)

        self.disa = None
        self.altp_req = altp_req
        self.mach = 0.55*self.airplane.cruise_mach      # INFO: mach must maximize path_eff
        self.path_eff = None

        self.path_req = path_req
        self.rating = rating
        self.speed_mode = speed_mode

    def eval(self):
        """Residual climb speed in MCL rating
        """
        disa = self.disa
        altp = self.altp_req
        mach = self.mach
        mass = self.mass

        speed_mode = self.speed_mode
        rating = self.rating
        kfn = 1.
        nei = 1
        pamb,tamb = util.atmosphere(altp, disa)
        speed = self.get_speed(pamb,speed_mode,mach)
        path,vz = self.air_path(nei,altp,disa,speed_mode,speed,mass,rating,kfn)
        self.path_eff = path
        return


class Economics():

    def __init__(self, airplane):
        self.airplane = airplane

        cost_range = self.airplane.cost_range

        self.irp = unit.s_year(10)
        self.period = unit.s_year(15)
        self.interest_rate = 0.04
        self.labor_cost = 120.
        self.utilization = self.yearly_utilization(cost_range)

        self.fuel_price = 2./unit.m3_usgal(1)
        self.energy_price = 0.10/unit.W_kW(1)
        self.battery_price = 20.

        self.engine_price = None
        self.gear_price = None
        self.frame_price = None

        self.frame_cost = None
        self.engine_cost = None
        self.cockpit_crew_cost = None
        self.cabin_crew_cost = None
        self.landing_fees = None
        self.navigation_fees = None
        self.catering_cost = None
        self.pax_handling_cost = None
        self.ramp_handling_cost = None

        self.std_op_cost = None
        self.cash_op_cost = None
        self.direct_op_cost = None

    def yearly_utilization(self, mean_range):
        """Compute the yearly utilization from the average range

        :param mean_range: Average range
        :return:
        """
        range = unit.convert_from("NM",
                      [ 100.,  500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.])
        utilization = [2300., 2300., 1500., 1200.,  900.,  800.,  700.,  600.,  600.]
        return util.lin_interp_1d(mean_range, range, utilization)

    def landing_gear_price(self):
        """Typical value
        """
        landing_gear_mass = self.airplane.landing_gears.mass
        gear_price = 720. * landing_gear_mass
        return gear_price

# TODO   electrofan price

    def one_engine_price(self):
        """Regression on catalog prices
        """
        reference_thrust = self.airplane.nacelles.engine_slst
        engine_price = ((2.115e-4*reference_thrust + 78.85)*reference_thrust)
        return engine_price

# TODO  battery price

    def one_airframe_price(self):
        """Regression on catalog prices corrected with engine prices
        """
        mwe = self.airplane.mass.mwe
        airframe_price = 0.7e3*(9e4 + 1.15*mwe - 1.8e9/(2e4 + mwe**0.94))
        return airframe_price


    def eval(self):
        """Computes Cash and Direct Operating Costs per flight (based on AAE 451 Spring 2004)
        """
        n_pax_ref = self.airplane.cabin.n_pax

        nacelle_mass = self.airplane.nacelles.mass

        reference_thrust = self.airplane.nacelles.engine_slst
        n_engine = self.airplane.propulsion.n_engine

        mtow = self.airplane.mass.mtow
        mwe = self.airplane.mass.mwe

        cost_range = self.airplane.cost_range
        time_block = self.airplane.missions.cost.time_block

        # Cash Operating Cost
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        fuel_density = 803.
        fuel_block = self.airplane.missions.cost.fuel_block
        self.fuel_cost = fuel_block*self.fuel_price/fuel_density

        b_h = time_block/3600.
        t_t = b_h + 0.25

        w_f = (10000. + mwe - nacelle_mass)*1.e-5

        labor_frame = ((1.26+1.774*w_f-0.1071*w_f**2)*t_t + (1.614+0.7227*w_f+0.1204*w_f**2))*self.labor_cost
        matrl_frame = (12.39+29.8*w_f+0.1806*w_f**2)*t_t + (15.20+97.330*w_f-2.8620*w_f**2)
        self.frame_cost = labor_frame + matrl_frame

        t_h = 0.05*((reference_thrust)/4.4482198)*1e-4

        labor_engine = n_engine*(0.645*t_t+t_h*(0.566*t_t+0.434))*self.labor_cost
        matrl_engine = n_engine*(25.*t_t+t_h*(0.62*t_t+0.38))

        self.engine_cost = labor_engine + matrl_engine

        w_g = mtow*1e-3

        self.cockpit_crew_cost = b_h*2*(440-0.532*w_g)
        self.cabin_crew_cost = b_h*np.ceil(n_pax_ref/50.)*self.labor_cost
        self.landing_fees = 8.66*(mtow*1e-3)
        self.navigation_fees = 57.*(cost_range/185200.)*np.sqrt((mtow/1000.)/50.)
        self.catering_cost = 3.07 * n_pax_ref
        self.pax_handling_cost = 2. * n_pax_ref
        self.ramp_handling_cost = 8.70 * n_pax_ref
        self.std_op_cost = self.fuel_cost + self.frame_cost + self.engine_cost + self.cockpit_crew_cost + self.landing_fees + self.navigation_fees #+ self.elec_cost
        self.cash_op_cost = self.std_op_cost + self.cabin_crew_cost + self.catering_cost + self.pax_handling_cost + self.ramp_handling_cost

        # DirectOperating Cost
        #-----------------------------------------------------------------------------------------------------------------------------------------------
        self.engine_price = self.one_engine_price()
        self.gear_price = self.landing_gear_price()
        self.frame_price = self.one_airframe_price()

#        battery_price = eco.battery_mass_price*cost_mission.req_battery_mass

        self.utilization = self.yearly_utilization(cost_range)
        self.aircraft_price = self.frame_price + self.engine_price * n_engine + self.gear_price #+ battery_price
        self.total_investment = self.frame_price * 1.06 + n_engine * self.engine_price * 1.025
        irp_year = unit.year_s(self.irp)
        period_year = unit.year_s(self.period)
        self.interest = (self.total_investment/(self.utilization*period_year)) * (irp_year * 0.04 * (((1. + self.interest_rate)**irp_year)/((1. + self.interest_rate)**irp_year - 1.)) - 1.)
        self.insurance = 0.0035 * self.aircraft_price/self.utilization
        self.depreciation = 0.99 * (self.total_investment / (self.utilization * period_year))     # Depreciation
        self.direct_op_cost = self.cash_op_cost + self.interest + self.depreciation + self.insurance

        return



#-----------------------------------------------------------------------------------------------------------------------
# Exposed modules
#-----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    ap = Airplane()


# High level process view
#-----------------------------------------
    ap.geometry.eval()

    ap.geometry.eval_tail_areas()

    ap.mass.eval()

    ap.mass.eval_characteristic_mass()

    ap.missions.eval_nominal_mission()

    ap.missions.eval_payload_range()

    ap.operations.eval()

    ap.economics.eval()


# Low level process view
#-----------------------------------------
    ap.cabin.eval_geometry()

    ap.fuselage.eval_geometry()

    ap.wing.eval_geometry()         # Wing area to be optimized

    ap.tank.eval_geometry()

    ap.htp.eval_geometry()          # HTP area to be computed

    ap.htp.eval_area()          # HTP area to be computed

    ap.vtp.eval_geometry()          # VTP area to be computed

    ap.vtp.eval_area()          # VTP area to be computed

    ap.nacelles.eval_geometry()

    ap.landing_gears.eval_geometry()

    ap.systems.eval_geometry()

    ap.geometry.eval_wet_area()


    ap.cabin.eval_mass()

    ap.fuselage.eval_mass()

    ap.wing.eval_mass()

    ap.tank.eval_mass()

    ap.htp.eval_mass()

    ap.vtp.eval_mass()

    ap.nacelles.eval_mass()

    ap.landing_gears.eval_mass()

    ap.systems.eval_mass()

    ap.mass.eval_characteristic_mass()


    ap.missions.eval_nominal_mission()

    ap.missions.eval_max_payload_mission()

    ap.missions.eval_max_fuel_mission()

    ap.missions.eval_zero_payload_mission()

    ap.missions.eval_cost_mission()


    ap.operations.eval_take_off()

    ap.operations.eval_approach()

    ap.operations.eval_climb_ceiling()

    ap.operations.eval_oei_ceiling()


    ap.economics.eval()


    # Info
    #-----------------------------------------
    print("")
    print("-------------------------------------------------------")
    print("Warning : no solving achieved")
    print("   - All couplings must be solved")
    print("   - All residuals must be driven to zero")
    print("   - Local optimization of flight speed to be done for One Engine Ceiling")
    print("   - Wing_area and Engine_slst must minimize MTOW, COC or DOC")
    print("")
    print("For some more info, look for 'INFO:' labels")


# Utils
#-----------------------------------------
    ap.print()

    ap.view_3d("This_plane")


