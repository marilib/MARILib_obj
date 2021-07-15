#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

:author: DRUOT Thierry, PETEILH Nicolas and MONROLIN Nicolas
"""

import numpy as np

import matplotlib.pyplot as plt

from marilib.utils import unit

from marilib.aircraft.airframe.component import Nacelle, Pod


class Drawing(object):

    def __init__(self, aircraft):
        self.aircraft = aircraft

    def payload_range(self,window_title):
        """
        Print the payload - range diagram
        """
        plot_title = self.aircraft.name

        payload = [self.aircraft.performance.mission.max_payload.payload,
                   self.aircraft.performance.mission.max_payload.payload,
                   self.aircraft.performance.mission.max_fuel.payload,
                   0.]

        range = [0.,
                 unit.NM_m(self.aircraft.performance.mission.max_payload.range),
                 unit.NM_m(self.aircraft.performance.mission.max_fuel.range),
                 unit.NM_m(self.aircraft.performance.mission.zero_payload.range)]

        nominal = [self.aircraft.performance.mission.nominal.payload,
                   unit.NM_m(self.aircraft.performance.mission.nominal.range)]

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(window_title)
        fig.suptitle(plot_title, fontsize=14)

        plt.plot(range,payload,linewidth=2,color="blue")
        plt.scatter(range[1:],payload[1:],marker="+",c="orange",s=100)
        plt.scatter(nominal[1],nominal[0],marker="o",c="green",s=50)

        plt.grid(True)

        plt.ylabel('Payload (kg)')
        plt.xlabel('Range (NM)')

        plt.show()


    def get_3d_curves(self):
        """
        Build 3D curves to print the plane
        """
        component = {"name":self.aircraft.name, "surface":[], "body":[], "nacelle":[]}

        for comp in self.aircraft.airframe:
            data = comp.sketch_3view()
            if data is not None:
                typ = comp.get_component_type()
                if typ in ["wing","htp","vtp"]:
                    component["surface"].append({"le":data["le"], "te":data["te"], "toc":data["toc"]})
                elif typ in ["body","wing_pod_tank","piggyback_tank"]:
                    component["body"].append({"xz":data["body_xz"], "xy":data["body_xy"]})
                if typ == "piggyback_tank":                                                 # Treat Piggy Back exception
                    pyl_data = comp.pylon_sketch()
                    component["surface"].append({"le":pyl_data["fle"], "te":pyl_data["fte"], "toc":pyl_data["toc"]})
                    component["surface"].append({"le":pyl_data["ble"], "te":pyl_data["bte"], "toc":pyl_data["toc"]})

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Nacelle):
                data = comp.sketch_3view()
                component["nacelle"].append({"le":data["fle"], "te":data["fte"], "toc":data["toc"]})
                component["nacelle"].append({"le":data["cle"], "te":data["cte"], "toc":data["toc"]})
                if comp.get_component_type() in ["body_nacelle", "body_tail_nacelle", "pod_tail_nacelle", "piggyback_tail_nacelle"]:
                    component["surface"].append({"le":data["s1le"], "te":data["s1te"], "toc":data["toc"]})
                    component["surface"].append({"le":data["s2le"], "te":data["s2te"], "toc":data["toc"]})


        return component


    def view_3d(self, window_title):
        """
        Build a 3 views drawing of the airplane
        """
        plot_title = self.aircraft.name

        # Drawing_ box
        #-----------------------------------------------------------------------------------------------------------
        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(window_title)
        fig.suptitle(plot_title, fontsize=14)
        axes.set_aspect('equal', 'box')
        plt.plot(np.array([0,100,100,0,0]), np.array([0,0,100,100,0]))      # Draw a square box of 100m side

        xTopView = 50 - (self.aircraft.airframe.wing.mac_loc[0] + 0.25*self.aircraft.airframe.wing.mac)
        yTopView = 50

        xSideView = 50 - (self.aircraft.airframe.wing.mac_loc[0] + 0.25*self.aircraft.airframe.wing.mac)
        ySideView = 82

        xFrontView = 50
        yFrontView = 10

        ref = {"xy":[xTopView,yTopView],"yz":[xFrontView,yFrontView],"xz":[xSideView,ySideView]}

        low, l1, l2, l3, l4, l5, high  =  0, 1, 2, 3, 4, 5, 6

        # Draw components
        #-----------------------------------------------------------------------------------------------------------
        zframe = {"xy":{"body":l2, "wing":l1, "htp":l1, "vtp":l3},      # top
                  "yz":{"body":l4, "wing":l3, "htp":l1, "vtp":l2},      # front
                  "xz":{"body":l2, "wing":l3, "htp":l3, "vtp":l1}}      # side

        if self.aircraft.arrangement.wing_attachment=="high":
            zframe["xy"]["body"] = l1
            zframe["xy"]["wing"] = l2

        if self.aircraft.arrangement.stab_architecture=="t_tail":
            zframe["xy"]["htp"] = l3
            zframe["xy"]["vtp"] = l1

        if self.aircraft.arrangement.stab_architecture=="h_tail":
            zframe["xz"]["htp"] = l1
            zframe["xz"]["vtp"] = l3

        for comp in self.aircraft.airframe:
            data = comp.sketch_3view()
            if data is not None:
                typ = comp.get_component_type()
                if typ in ["body","wing","htp","vtp"]:
                    for view in ["xy","yz","xz"]:
                        plt.fill(ref[view][0]+data[view][0:,0], ref[view][1]+data[view][0:,1], color="white", zorder=zframe[view][typ])    # draw mask
                        plt.plot(ref[view][0]+data[view][0:,0], ref[view][1]+data[view][0:,1], color="grey", zorder=zframe[view][typ])     # draw contour

        # Draw tanks
        #-----------------------------------------------------------------------------------------------------------
        #                            top        front    side
        zpod = {   "wing_pod_tank":{"xy":l3,   "yz":l4, "xz":l4},
                  "piggyback_tank":{"xy":high, "yz":l4, "xz":l2}}

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Pod):
                pod = comp.sketch_3view()
                typ = comp.get_component_type()
                if typ=="wing_pod_tank" and self.aircraft.airframe.tank.frame_origin[2] < self.aircraft.airframe.tank.wing_axe_z:
                    zpod["wing_pod_tank"]["xy"] = low
                for view in ["xy","yz","xz"]:
                    plt.fill(ref[view][0]+pod[view][0:,0], ref[view][1]+pod[view][0:,1], color="white", zorder=zpod[typ][view])    # draw mask
                    plt.plot(ref[view][0]+pod[view][0:,0], ref[view][1]+pod[view][0:,1], color="grey", zorder=zpod[typ][view])     # draw contour
                # print(typ,zpod[typ]["xy"],zframe["xy"]["wing"])
                if typ=="wing_pod_tank" and zframe["xy"]["wing"] < zpod[typ]["xy"]:
                    data = self.aircraft.airframe.wing.sketch_3view()
                    view = "xz_tip"
                    plt.fill(ref["xz"][0]+data[view][0:,0], ref["xz"][1]+data[view][0:,1], color="white", zorder=high)    # draw mask
                    plt.plot(ref["xz"][0]+data[view][0:,0], ref["xz"][1]+data[view][0:,1], color="grey", zorder=high)     # draw contour

        # Draw nacelles
        #-----------------------------------------------------------------------------------------------------------
        #                                  top        front     side
        znac = {          "wing_nacelle":{"xy":low,  "yz":l4,  "xz":high},
                          "body_nacelle":{"xy":l3,   "yz":l1,  "xz":high},
                     "body_tail_nacelle":{"xy":l1,   "yz":low, "xz":low},
                      "pod_tail_nacelle":{"xy":l4,   "yz":low, "xz":l5},
                "piggyback_tail_nacelle":{"xy":high, "yz":low, "xz":l2}}

        for comp in self.aircraft.airframe:
            if issubclass(type(comp),Nacelle):
                nacelle = comp.sketch_3view()
                typ = comp.get_component_type()
                for view in ["xy","yz","xz"]:
                    plt.fill(ref[view][0]+nacelle[view][0:,0], ref[view][1]+nacelle[view][0:,1], color="white", zorder=znac[typ][view])    # draw mask
                    plt.plot(ref[view][0]+nacelle[view][0:,0], ref[view][1]+nacelle[view][0:,1], color="grey", zorder=znac[typ][view])     # draw contour
                # plt.fill(ref[view][0]+nacelle["disk"][0:,0], ref["disk"][1]+nacelle[view][0:,1], color="white", zorder=znac[typ]["yz"])    # draw mask
                plt.plot(ref["yz"][0]+nacelle["disk"][0:,0], ref["yz"][1]+nacelle["disk"][0:,1], color="grey", zorder=znac[typ]["yz"])     # draw contour

        plt.show()
        return

