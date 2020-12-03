#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Weichang LYU, Thierry DRUOT
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

from marilib.utils import unit


# ======================================================================================================
# Category definition
# ------------------------------------------------------------------------------------------------------
class AirplaneCategories(object):

    def __init__(self):
        """Wakevortex categories versus nominal range
        """
        self.design_capacity = {  "regional":[40., 80.],
                                     "short":[80., 125.],
                                    "medium":[125., 210.],
                                      "long":[250., 400.],
                                "ultra_long":[350., 850.]
                                }

        self.design_range = {  "regional":[unit.m_NM(0.), unit.m_NM(1000.)],
                                  "short":[unit.m_NM(1000.), unit.m_NM(2500.)],
                                 "medium":[unit.m_NM(2500.), unit.m_NM(4000.)],
                                   "long":[unit.m_NM(4500.), unit.m_NM(6500.)],
                             "ultra_long":[unit.m_NM(6500.), unit.m_NM(9000.)]
                             }

        self.design_mtow = {  "regional":[unit.kg_t(10.), unit.kg_t(25.)],
                                 "short":[unit.kg_t(18.), unit.kg_t(60.)],
                                "medium":[unit.kg_t(60.), unit.kg_t(100.)],
                                  "long":[unit.kg_t(100.), unit.kg_t(350.)],
                            "ultra_long":[unit.kg_t(400.), unit.kg_t(600.)]
                            }

        self.span = {  "regional":[20., 32.],
                          "short":[20., 32.],
                         "medium":[30., 40.],
                           "long":[60., 65.],
                     "ultra_long":[70., 80.]
                     }

        self.tofl = {  "regional":[1000., 1500.],
                          "short":[1500., 2000.],
                         "medium":[2000., 2500.],
                           "long":[2500., 3000.],
                     "ultra_long":[3000., 3500.]
                     }

        self.cruise_altp = {  "regional":[unit.m_ft(25000.), unit.m_ft(27000.)],
                                 "short":[unit.m_ft(35000.), unit.m_ft(41000.)],
                                "medium":[unit.m_ft(35000.), unit.m_ft(41000.)],
                                  "long":[unit.m_ft(35000.), unit.m_ft(41000.)],
                            "ultra_long":[unit.m_ft(35000.), unit.m_ft(41000.)]
                            }

        self.cruise_speed = {  "regional":[0.45, 0.55],
                                  "short":[0.76, 0.80],
                                 "medium":[0.76, 0.80],
                                   "long":[0.84, 0.86],
                             "ultra_long":[0.84, 0.86]
                             }

        self.app_speed = {  "regional":[unit.mps_kt(90.), unit.mps_kt(115.)],
                               "short":[unit.mps_kt(125.), unit.mps_kt(135.)],
                              "medium":[unit.mps_kt(135.), unit.mps_kt(155.)],
                                "long":[unit.mps_kt(135.), unit.mps_kt(155.)],
                          "ultra_long":[unit.mps_kt(135.), unit.mps_kt(155.)]
                          }

        self.wakevortex = {  "regional":"E",
                                "short":"E",
                               "medium":"D",
                                 "long":"C",
                           "ultra_long":"B"
                           }

        # runway occupation time (s)
        self.r_o_t = {  "regional":40.,
                           "short":40.,
                          "medium":60.,
                            "long":70.,
                      "ultra_long":80.
                      }

        # distance-based separation minimma on approach by pair of aircraft [LEADER][FOLLOWER]
        # The values are based on the RECAT-EU document, values in NM
        self.approach_separation = {"A":{"A":unit.m_NM(3. ), "B":unit.m_NM(4. ), "C":unit.m_NM(5. ), "D":unit.m_NM(5. ), "E":unit.m_NM(6. ), "F":unit.m_NM(8.)},
                                    "B":{"A":unit.m_NM(2.5), "B":unit.m_NM(3. ), "C":unit.m_NM(4. ), "D":unit.m_NM(4. ), "E":unit.m_NM(5. ), "F":unit.m_NM(7.)},
                                    "C":{"A":unit.m_NM(2.5), "B":unit.m_NM(2.5), "C":unit.m_NM(3. ), "D":unit.m_NM(3. ), "E":unit.m_NM(4. ), "F":unit.m_NM(6.)},
                                    "D":{"A":unit.m_NM(2.5), "B":unit.m_NM(2.5), "C":unit.m_NM(2.5), "D":unit.m_NM(2.5), "E":unit.m_NM(2.5), "F":unit.m_NM(5.)},
                                    "E":{"A":unit.m_NM(2.5), "B":unit.m_NM(2.5), "C":unit.m_NM(2.5), "D":unit.m_NM(2.5), "E":unit.m_NM(2.5), "F":unit.m_NM(4.)},
                                    "F":{"A":unit.m_NM(2.5), "B":unit.m_NM(2.5), "C":unit.m_NM(2.5), "D":unit.m_NM(2.5), "E":unit.m_NM(2.5), "F":unit.m_NM(3.)}
                                    }

        # time-based separation minimma on departure by pair of aircraft [LEADER][FOLLOWER]
        # The values are based on the RECAT-EU document, values in second
        self.takeoff_separation = {"A":{"A":120., "B":100., "C":120., "D":140., "E":160., "F":180.},
                                   "B":{"A":120., "B":120., "C":120., "D":100., "E":120., "F":140.},
                                   "C":{"A":120., "B":120., "C":120., "D":80. , "E":100., "F":120.},
                                   "D":{"A":120., "B":120., "C":120., "D":120., "E":120., "F":120.},
                                   "E":{"A":120., "B":120., "C":120., "D":120., "E":120., "F":100.},
                                   "F":{"A":120., "B":120., "C":120., "D":120., "E":120., "F":80.}
                                   }

        self.buffer_time = {"A":{"A":30., "B":30., "C":30., "D":30., "E":30., "F":30.},
                            "B":{"A":30., "B":30., "C":30., "D":30., "E":30., "F":30.},
                            "C":{"A":30., "B":30., "C":30., "D":30., "E":30., "F":30.},
                            "D":{"A":30., "B":30., "C":30., "D":30., "E":30., "F":30.},
                            "E":{"A":30., "B":30., "C":30., "D":30., "E":30., "F":30.},
                            "F":{"A":30., "B":30., "C":30., "D":30., "E":30., "F":30.}
                            }

    def get_data(self, type, segment):
        return getattr(self, type)[segment]

    def seg_from_pax(self, npax):
        """Retrieve range segment from design capacity
        """
        for k,r in self.design_capacity.items():
            if r[0]<=npax and npax<=r[1]:
                return k
        raise Exception("Cannot categorize number of passenger")

    def get_data_from_seg(self, type, segment, ratio=None):
        """Retrieve data type from range segment
        If ratio=None, min and max values are retrieved
        If 0<=ratio<=1, (1-ratio)*min + ratio*max is retrieved
        """
        if ratio is None:
            return self.get_data(type,segment)
        else:
            return (1.-ratio)*self.get_data(type,segment)[0] + ratio*self.get_data(type,segment)[1]

    def get_data_from_pax(self, type, npax, ratio=None):
        """Retrieve data type from design capacity
        If ratio=None, min and max values are retrieved
        If 0<=ratio<=1, (1-ratio)*min + ratio*max is retrieved
        """
        if type in ["design_range", "design_capacity", "design_mtow", "span", "tofl", "cruise_altp", "cruise_speed", "app_speed"]:
            segment = self.seg_from_pax(npax)
            if ratio is None:
                return self.get_data(type,segment)
            else:
                return (1.-ratio)*self.get_data(type,segment)[0] + ratio*self.get_data(type,segment)[1]
        else:
            raise Exception("Data type is unknown")

    def get_separation_data(self, type, cat_leader, cat_follower):
        """This function  determines the distance-based separation minimmal on approach by pair of aircraft (LEADER, FOLLOWER).
        The values are based on the RECAT-EU document"""
        if type in ["approach_separation", "takeoff_separation", "buffer_time"]:
            return getattr(self, type)[cat_leader][cat_follower]
        else:
            raise Exception("Data type is unknown")


# ======================================================================================================
# Airplane object
# ------------------------------------------------------------------------------------------------------
class Aircraft(object):
    """Airplane object
    """
    def __init__(self, phd, cat, npax=150., range=unit.m_NM(3000.), mach=0.78):
        self.phd = phd
        self.cat = cat

        self.cruise_altp = self.__altp(npax) # Reference cruise altitude
        self.cruise_mach = mach              # Cruise Mach number
        self.cruise_speed = None             # Cruise speed
        self.range = range                   # Range
        self.npax = npax                     # Npax
        self.mpax = self.__mpax(npax)        # Weight per passenger
        self.payload = None                  # Design mission payload
        self.mtow = None                     # Design mission Maximum Take Off Weight
        self.owe = None                      # Design mission Operating Empty Weight
        self.ldw = None                      # Design mission Landing Weight
        self.fuel_mission = None             # Design mission fuel
        self.fuel_reserve = None             # Design mission reserve fuel
        self.kr = self.__kr(range)           # fraction of mission fuel for reserve

        self.payload_max = None     # Maximum payload
        self.range_pl_max = None    # Range for maximum payload mission

        self.payload_fuel_max = None    # Payload for max fuel mission
        self.range_fuel_max = None      # Range for max fuel mission

        self.range_no_pl = None     # Range for zero payload mission

        self.eff_ratio = self.__eff_ratio(npax)     # Efficiency ratio for specific air range
        self.owe_coef = [-1.478e-07, 5.459e-01, 8.40e+02]   # "Structural model"

        self.design_aircraft()

    def __altp(self, npax):
        return self.cat.get_data_from_pax("cruise_altp", npax)[0]

    def __mpax(self, npax):
        """Weight allowance per passenger in kg
        """
        if   npax<=40.  : return 95.
        elif npax<=80.  : return 105.
        elif npax<=125. : return 115.
        elif npax<=250. : return 125.
        elif npax<=350. : return 135.
        else            : return 145.

    def __kr(self, dist):
        """Reserve fuel factor
        """
        if   dist<=unit.m_NM(3500.)  : return 0.05
        else: return 0.03

    def __eff_ratio(self, npax):
        """Ratio L/D over SFC for Breguet equation
        This ratio is related to the capacity segment of the aircraft so,
        it is computed according to the number of passenger
        """
        pax_list = [10., 60., 260., 360.]
        lod_list = [15., 15.,  19.,  19.]
        f_lod = interp1d(pax_list, lod_list, kind="linear", fill_value='extrapolate')
        lod = f_lod(npax)
        sfc = unit.convert_from("kg/daN/h", 0.60)  # Techno assumption
        return lod/sfc

    def structure(self, mtow, coef=None):
        """Link between MTOW and OWE. This link implecitly represents the structural sizing
        """
        if coef is not None: self.owe_coef = coef
        owe = (self.owe_coef[0]*mtow + self.owe_coef[1]) * mtow + self.owe_coef[2]    # Structure design rule
        return owe

    def mission(self, tow, fuel_mission, effr=None):
        """Mission evaluation based on Breguet equation
        Warning : if given effr must be expressed in N.s/kg
        """
        if effr is not None: self.eff_ratio = effr
        pamb,tamb,g = self.phd.atmosphere(self.cruise_altp)
        vsnd = self.phd.sound_speed(tamb)
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

        pamb,tamb,g = self.phd.atmosphere(self.cruise_altp)
        vsnd = self.phd.sound_speed(tamb)
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


# ======================================================================================================
# Fleet object
# ------------------------------------------------------------------------------------------------------
class Fleet(object):
    """Fleet object
    """
    def __init__(self, phd,cat,ac_def):

        self.aircraft = []      # List of the airplanes of the fleet
        self.segment = []       # List of the segment names of the airplanes
        self.ratio = []         # Proportion of each aircraft in the fleet
        for seg,dat in ac_def.items():
            self.segment.append(seg)
            self.ratio.append(dat["ratio"])
            self.aircraft.append(Aircraft(phd,cat, npax=dat["npax"] , range=dat["range"] , mach=dat["mach"]))

        self.network = None

        self.dist_factor = 1.15     # Factor on great circle distance

        n = len(self.aircraft)

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
        f_util = interp1d(range, utilization, kind="linear", fill_value='extrapolate')
        return f_util(mean_range)

    def fleet_analysis(self, route_network):
        cstep = route_network["npax_step"]
        rstep = route_network["range_step"]
        array = route_network["matrix"]

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
