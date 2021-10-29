#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot, Weichang Lyu
"""

import numpy as np
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

from marilib.utils import unit
from marilib.utils.math import lin_interp_1d

from analyse_data import coloration, read_db, lin_lst_reg, draw_reg, subplots_by_varname,\
    draw_colored_cloud_on_axis, get_error, do_regression


# ======================================================================================================
# Airplane object definition
# ------------------------------------------------------------------------------------------------------
class Aircraft(object):
    """Airplane object
    """
    def __init__(self,
                 npax=150.,
                 range=unit.m_NM(3000.),
                 mach=0.78):
        self.cruise_altp = unit.m_ft(35000.)    # Reference cruise altitude
        self.cruise_mach = mach     # Cruise Mach number
        self.cruise_speed = None    # Cruise speed
        self.range = range          # Range
        self.npax = npax            # Npax
        self.mpax = 130.            # Weight per passenger
        self.payload = None         # Design mission payload
        self.mtow = None            # Design mission Maximum Take Off Weight
        self.owe = None             # Design mission Operating Empty Weight
        self.ldw = None             # Design mission Landing Weight
        self.fuel_mission = None    # Design mission fuel
        self.fuel_reserve = None    # Design mission reserve fuel
        self.kr = 0.03              # fraction of mission fuel for reserve

        self.payload_max = None     # Maximum payload
        self.range_pl_max = None    # Range for maximum payload mission

        self.payload_fuel_max = None    # Payload for max fuel mission
        self.range_fuel_max = None      # Range for max fuel mission

        self.range_no_pl = None     # Range for zero payload mission

        self.eff_ratio = self.__eff_ratio(npax)     # Efficiency ratio for specific air range

        self.owe_coef = [-2.47547354e-07, 5.61453046e-01, 0.]   # Structural model from data base

        self.design_aircraft()

    def __eff_ratio(self, npax):
        """Ratio L/D over SFC for Breguet equation
        This ratio is related to the capacity segment of the aircraft so,
        it is computed according to the number of passenger
        """
        pax_list = [10., 60., 260., 360.]
        lod_list = [13., 15.,  21.,  21.]
        sfc_list = unit.convert_from("kg/daN/h", [0.7, 0.60,  0.54,  0.54])
        lod = lin_interp_1d(npax, pax_list, lod_list)
        sfc = lin_interp_1d(npax, pax_list, sfc_list)
        return lod/sfc

    def structure(self, mtow, coef=None):
        """Link between MTOW and OWE. This link implecitly represents the structural sizing
        """
        if coef is not None : self.owe_coef = coef
        owe = (self.owe_coef[0]*mtow + self.owe_coef[1]) * mtow + self.owe_coef[2]    # Structure design rule
        return owe

    def mission(self, tow, fuel_mission, effr=None):
        """Mission evaluation based on Breguet equation
        Warning : if given effr must be expressed in N.s/kg
        """
        if effr is not None: self.eff_ratio = effr
        pamb,tamb,vsnd,g = self.atmosphere(self.cruise_altp)
        range_factor = (self.cruise_mach*vsnd*self.eff_ratio)/g
        range = range_factor*np.log(tow/(tow-fuel_mission))       # Breguet equation
        return range

    def operation(self, n_pax, range):
        """Operational mission
        Compute mission data from passenger on board and range

        :param range: Distance to fly
        :param n_pax: Number of passengers
        :return:  mission_fuel,mission_time,tow
        """
        def fct(x_in):
            tow = x_in[0]
            fuel_mission = x_in[1]
            range_eff = self.mission(tow, fuel_mission)
            owe_eff = tow - (self.mpax*n_pax + (1.+self.kr)*fuel_mission)
            return np.array([self.owe-owe_eff, range-range_eff])

        x_ini = self.npax*self.mpax*np.array([4., 1.])
        dict = fsolve(fct, x0=x_ini, full_output=True)
        if (dict[2] != 1): raise Exception("Convergence problem")
        tow = dict[0][0]
        mission_fuel = dict[0][1]
        mission_time = 20.*60. + range/self.cruise_speed
        return mission_fuel,mission_time,tow

    def eval_design(self, X):
        """Evaluation function for design_aircraft
        """
        self.mtow = X[0]
        self.fuel_mission = X[1]

        owe_eff = self.structure(self.mtow)  # 1
        range_eff = self.mission(self.mtow, self.fuel_mission)  # 2

        self.fuel_reserve = self.kr*self.fuel_mission  # 3
        self.ldw = self.mtow - self.fuel_mission
        self.payload = self.npax * self.mpax  # 4
        self.owe = self.mtow - self.payload - self.fuel_mission - self.fuel_reserve  # 5
        return np.array([self.owe-owe_eff,self.range-range_eff])

    def design_aircraft(self, coef=None, kr=None, mpax=None, effr=None):
        """Design method (mass-mission adaptation only
        Warning : if given effr must be expressed in daN.h/kg
        """
        if coef is not None: self.owe_coef = coef
        if kr is not None: self.kr = kr
        if mpax is not None: self.mpax = mpax
        if effr is not None: self.eff_ratio = effr / unit.convert_from("kg/daN/h", 1.)

        Xini = self.npax*self.mpax*np.array([4., 1.])
        dict = fsolve(self.eval_design, x0=Xini, full_output=True)
        if (dict[2] != 1): raise Exception("Convergence problem")

        self.eval_design(np.array([dict[0][0], dict[0][1]]))

        pamb,tamb,vsnd,g = self.atmosphere(self.cruise_altp)
        self.cruise_speed = vsnd*self.cruise_mach

        self.payload_max = self.payload * 1.20
        fuel = (self.mtow - self.owe - self.payload_max) / (1.+self.kr)
        self.range_pl_max = self.mission(self.mtow, fuel)

        self.payload_fuel_max = self.payload * 0.60
        fuel_max = (self.mtow - self.owe - self.payload_fuel_max) / (1.+self.kr)
        self.range_fuel_max = self.mission(self.mtow, fuel_max)

        tow = self.owe + fuel_max * (1.+self.kr)
        self.range_no_pl = self.mission(tow, fuel_max)

    def is_in_plr(self, npax, range):
        """Assess if a mission is possible
        """
        payload = npax * self.mpax
        out_dict = {"capa":True, "dist":True}
        c1 = self.payload_max - payload                                                                 # Max payload limit
        c2 =  (payload-self.payload_fuel_max)*(self.range_pl_max-self.range_fuel_max) \
            - (self.payload_max-self.payload_fuel_max)*(range-self.range_fuel_max)                      # Max Take off weight limit
        c3 = payload*(self.range_fuel_max-self.range_no_pl) - self.payload_max*(range-self.range_no_pl) # Max fuel limit
        c4 = self.range_no_pl - range                                                                   # Max range limit
        if ((c1<0. or c2<0. or c3<0.) and c4>=0.):  # Out of PLR because of capacity
            out_dict["capa"] = False
        elif (c1>=0. and c4<0.):                    # Out of PLR because of range
            out_dict["dist"] = False
        elif (c1<0. and c4<0.):                     # Out of PLR because of range and capacity
            out_dict["capa"] = False
            out_dict["dist"] = False
        return out_dict

    def max_capacity(self, range):
        """Retrieve the maximum capacity for a given range

        :param range: Distance to fly
        :return:  capacity
        """
        if range<=self.range_pl_max:
            capacity = np.floor(self.payload_max/self.mpax)
        elif self.range_pl_max<range and range<=self.range_fuel_max:
            payload =    self.payload_fuel_max + (self.payload_max-self.payload_fuel_max) * (range-self.range_fuel_max) / (self.range_pl_max-self.range_fuel_max)
            capacity = np.floor(payload/self.mpax)
        elif self.range_fuel_max<range and range<=self.range_no_pl:
            payload =   self.payload_fuel_max*(range-self.range_no_pl) / (self.range_fuel_max-self.range_no_pl)
            capacity = np.floor(payload/self.mpax)
        else:
            capacity = 0.
        return capacity

    def max_range(self, npax):
        """Retrieve the maximum range for a given number of passenger

        :param npax: Number of passenger
        :return:  range
        """
        payload = self.mpax*npax
        if self.payload_max<payload:
            range = 0.
        elif self.payload_fuel_max<payload and payload<=self.payload_max:
            range = self.range_fuel_max + (payload - self.payload_fuel_max) * (self.range_pl_max-self.range_fuel_max) / (self.payload_max-self.payload_fuel_max)
        else:
            range = self.range_no_pl + payload * (self.range_fuel_max-self.range_no_pl) / self.payload_fuel_max
        return range

    def atmosphere(self, altp, disa=0.):
        """Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
        """
        g = 9.80665
        r = 287.053
        gam = 1.4

        Z = np.array([0., 11000., 20000., 32000., 47000., 50000.])
        dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])
        P = np.array([101325., 0., 0., 0., 0., 0.])
        T = np.array([288.15, 0., 0., 0., 0., 0.])

        if (Z[-1] < altp):
            raise Exception("atmosphere, altitude cannot exceed 50km")

        j = 0
        while (Z[1+j] <= altp):
            T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
            if (0. < np.abs(dtodz[j])):
                P[j+1] = P[j]*(1. + (dtodz[j]/T[j]) * (Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
            else:
                P[j+1] = P[j]*np.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
            j = j + 1

        if (0. < np.abs(dtodz[j])):
            pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
        else:
            pamb = P[j]*np.exp(-(g/r)*((altp-Z[j])/T[j]))
        tamb = T[j] + dtodz[j]*(altp-Z[j]) + disa
        vsnd = np.sqrt(gam*r*tamb)
        return pamb, tamb, vsnd, g

    def payload_range(self):
        """Print the payload - range diagram
        """
        payload = [self.payload_max,
                   self.payload_max,
                   self.payload_fuel_max,
                   0.]

        range = [0.,
                 unit.NM_m(self.range_pl_max),
                 unit.NM_m(self.range_fuel_max),
                 unit.NM_m(self.range_no_pl)]

        nominal = [self.payload,
                   unit.NM_m(self.range)]

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title("Pico Design")
        fig.suptitle("Payload - Range", fontsize=14)

        plt.plot(range,payload,linewidth=2,color="blue")
        plt.scatter(range[1:],payload[1:],marker="+",c="orange",s=100)
        plt.scatter(nominal[1],nominal[0],marker="o",c="green",s=50)

        plt.grid(True)

        plt.ylabel('Payload (kg)')
        plt.xlabel('Range (NM)')

        plt.show()



class Fleet(object):
    """Fleet object
    """
    def __init__(self, ac_list):
        self.aircraft = ac_list     # List of the airplanes of the fleet
        self.network = None
        self.dist_factor = 1.15     # Factor on great circle distance

        n = len(ac_list)

        self.fleet_trip = [0]*n
        self.fleet_npax = [0]*n
        self.fleet_capa = [0]*n
        self.fleet_dist = [0.]*n
        self.fleet_fuel = [0.]*n
        self.fleet_time = [0.]*n
        self.fleet_paxkm = [0.]*n
        self.fleet_tonkm = [0.]*n
        self.fleet_plane = [0.]*n

    def utilization(self, mean_range):
        """Compute the yearly utilization from the average range

        :param mean_range: Average range
        :return:
        """
        range = unit.convert_from("NM",
                      [ 100.,  500., 1000., 1500., 2000., 2500., 3000., 3500., 4000.])
        utilization = [2300., 2300., 1500., 1200.,  900.,  800.,  700.,  600.,  600.]
        return lin_interp_1d(mean_range, range, utilization)

    def fleet_analysis(self, data_matrix):
        cstep = data_matrix["npax_step"]
        rstep = data_matrix["range_step"]
        array = data_matrix["matrix"]

        mtow_list = [ac.mtow for ac in self.aircraft]           # List of MTOW of fleet airplanes
        mtow_index = np.argsort(mtow_list)                      # increaeing order of MTOWs

        range_list = [ac.range_fuel_max for ac in self.aircraft]    # List of MTOW of fleet airplanes
        range_index = np.argsort(range_list)                        # increaeing order of range

        capa_list = [ac.payload_max for ac in self.aircraft]    # List of MTOW of fleet airplanes
        capa_index = np.argsort(capa_list)                      # increaeing order of capacity

        nc,nr = array.shape

        def fly_it(i,nflight,capa,npax,dist,dist_eff):
            fuel,time,tow = self.aircraft[i].operation(npax,dist_eff)
            self.fleet_trip[i] += nflight
            self.fleet_npax[i] += npax*nflight
            self.fleet_capa[i] += capa*nflight
            self.fleet_dist[i] += dist*nflight
            self.fleet_fuel[i] += fuel*nflight
            self.fleet_time[i] += time*nflight
            self.fleet_paxkm[i] += npax*(dist*1.e-3)*nflight
            self.fleet_tonkm[i] += (dist*1.e-3)*(npax*self.aircraft[i].mpax*1.e-3)*nflight

        for c in range(nc):
            for r in range(nr):
                npax = cstep*(1.+c)
                dist = rstep*1000.*(1.+r)           # Great circle distance
                dist_eff = dist*self.dist_factor    # Operational distance is longer than great circle
                nflight = array[c,r]
                flag = False
                for i in mtow_index:
                    out_dict = self.aircraft[i].is_in_plr(npax,dist_eff)
                    if out_dict["capa"] and out_dict["dist"]:  # Mission can be done in one step with a single aircraft
                        capa = self.aircraft[i].max_capacity(dist_eff)
                        fly_it(i,nflight,capa,npax,dist,dist_eff)
                        flag = True
                        break
                if not flag:
                    for i in range_index:
                        out_dict = self.aircraft[i].is_in_plr(npax,dist_eff)
                        if (not out_dict["capa"]) and out_dict["dist"]:    # Mission can be done by spliting the payload into several flights
                            capa = self.aircraft[i].max_capacity(dist_eff)
                            if capa>=np.ceil(0.50*npax):
                                # print("Flight realized at max capacity: npax = ",npax," capa = ",capa," range = ","%.0f"%unit.km_m(dist_eff)," km")
                                nf = 0
                                while npax>0.:
                                    fly_it(i,nflight,capa,npax,dist,dist_eff)
                                    npax -= capa
                                    nf += 1
                                # print(nf," times")
                                flag = True
                                break
                if not flag:
                    for i in capa_index:
                        out_dict = self.aircraft[i].is_in_plr(npax,dist_eff)
                        if out_dict["capa"] or out_dict["dist"]:    # Mission can be done by a single aircraft in several steps
                            max_dist = self.aircraft[i].max_range(npax)
                            if max_dist>=(0.50*dist_eff):
                                capa = self.aircraft[i].max_capacity(max_dist)
                                # print("Flight realized at max range: npax = ",npax," capa = ",capa," max range = ","%.0f"%unit.km_m(max_dist)," km"," range = ","%.0f"%unit.km_m(dist_eff)," km")
                                ns = 0
                                while dist_eff>0.:
                                    dist = max_dist/self.dist_factor
                                    fly_it(i,nflight,capa,npax,dist,dist_eff)
                                    dist_eff -= max_dist
                                    ns += 1
                                # print(ns," steps")
                                flag = True
                                break
                if not flag:
                    print("This is embarrassing, this mission could not be flown : npax = ", npax," range = ","%.0f" % unit.km_m(dist_eff), " km")

        n = len(self.aircraft)
        for j in range(n):
            mean_range = self.fleet_dist[j]/(1+self.fleet_trip[j])
            utilisation = self.utilization(mean_range)
            self.fleet_plane[j] = np.ceil(self.fleet_trip[j]/utilisation)

        total_trip = sum(self.fleet_trip)
        total_npax = sum(self.fleet_npax)
        total_capa = sum(self.fleet_capa)
        total_dist = sum(self.fleet_dist)
        total_fuel = sum(self.fleet_fuel)
        total_time = sum(self.fleet_time)
        total_paxkm = sum(self.fleet_paxkm)
        total_tonkm = sum(self.fleet_tonkm)

        out_dict = {"trip":total_trip, "npax":total_npax, "capa":total_capa, "dist":total_dist,
                    "fuel":total_fuel, "time":total_time, "tonkm":total_tonkm, "paxkm":total_paxkm}

        return out_dict


if __name__ == '__main__':

    # Read data
    #-------------------------------------------------------------------------------------------------------------------
    path_to_data_base = "../../../data/All_Data_v5.xlsx"

    df,un = read_db(path_to_data_base)

    # Remove A380-800 row and reset index
    df = df[df['name']!='A380-800'].reset_index(drop=True)

    df1 = df[df['airplane_type']!='business'].reset_index(drop=True).copy()
    un1 = un.copy()


    # perform regressions
    #-------------------------------------------------------------------------------------------------------------------
    # abs = "MTOW"
    # ord = "OWE"
    #
    # # print(tabulate(df[[abs,ord]], headers='keys', tablefmt='psql'))
    # # df = df[df['MTOW']<6000].reset_index(drop=True)                     # Remove all airplane with MTOW > 6t
    #
    # # order = [1]
    # order = [2, 1]
    # dict_owe = do_regression(df1, un1, abs, ord, coloration, order)



    # Analysis
    #-------------------------------------------------------------------------------------------------------------------
    ac = Aircraft()

    df2 = df1[df1['cruise_speed']<1].reset_index(drop=True).copy()
    un2 = un1.copy()

    nap = df2.shape[0]

    df2["MTOW_2"] = df2["MTOW"]
    un2["MTOW_2"] = "kg"

    for n in range(nap):
        ac.npax = df2["n_pax"][n]
        ac.range = df2["nominal_range"][n]
        ac.cruise_mach = df2["cruise_speed"][n]
        ac.design_aircraft()
        df2["MTOW_2"][n] = ac.mtow

    abs = "MTOW"
    ord = "MTOW_2"

    # dict = draw_reg(df2, un2, abs, ord, [[],[]], coloration)

    order = [1]
    dict_owe = do_regression(df2, un2, abs, ord, coloration, order)
