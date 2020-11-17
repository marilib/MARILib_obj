#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
from marilib.utils.math import lin_interp_1d, maximize_1d

from marilib.utils import unit, earth


class Categories(object):

    def __init__(self):
        """Wakevortex categories versus nominal range
        """
        self.design_range = {  "regional":[unit.m_NM(0.), unit.m_NM(1000.)],
                                  "short":[unit.m_NM(1000.), unit.m_NM(2500.)],
                                 "medium":[unit.m_NM(2500.), unit.m_NM(4000.)],
                                   "long":[unit.m_NM(4500.), unit.m_NM(6500.)],
                             "ultra_long":[unit.m_NM(6500.), unit.m_NM(9000.)]
                             }

        self.capacity = {  "regional":[40., 80.],
                              "short":[80., 125.],
                             "medium":[125., 210.],
                               "long":[250., 400.],
                         "ultra_long":[350., 850.]
                         }

        self.mtow = {  "regional":[unit.kg_t(10.), unit.kg_t(25.)],
                          "short":[unit.kg_t(18.), unit.kg_t(60.)],
                         "medium":[unit.kg_t(60.), unit.kg_t(100.)],
                           "long":[unit.kg_t(100.), unit.kg_t(350.)],
                     "ultra_long":[unit.kg_t(400.), unit.kg_t(600.)]
                     }

        self.app_speed = {  "regional":[unit.mps_kt(90.), unit.mps_kt(115.)],
                               "short":[unit.mps_kt(125.), unit.mps_kt(135.)],
                              "medium":[unit.mps_kt(135.), unit.mps_kt(155.)],
                                "long":[unit.mps_kt(135.), unit.mps_kt(155.)],
                          "ultra_long":[unit.mps_kt(135.), unit.mps_kt(155.)]
                          }

        # runway occupation time (s)
        self.r_o_t = {  "regional":40.,
                           "short":40.,
                          "medium":60.,
                            "long":70.,
                      "ultra_long":80.
                      }

        self.wakevortex = {  "regional":"E",
                                "short":"E",
                               "medium":"D",
                                 "long":"C",
                           "ultra_long":"B"
                           }

        # distance-based separation minimma on approach by pair of aircraft [LEADER][FOLLOWER]
        # The values are based on the RECAT-EU document, values in NM
        self.approach_separation = {"A":{"A":3. , "B":4. , "C":5. , "D":5. , "E":6. , "F":8.},
                                    "B":{"A":2.5, "B":3. , "C":4. , "D":4. , "E":5. , "F":7.},
                                    "C":{"A":2.5, "B":2.5, "C":3. , "D":3. , "E":4. , "F":6.},
                                    "D":{"A":2.5, "B":2.5, "C":2.5, "D":2.5, "E":2.5, "F":5.},
                                    "E":{"A":2.5, "B":2.5, "C":2.5, "D":2.5, "E":2.5, "F":4.},
                                    "F":{"A":2.5, "B":2.5, "C":2.5, "D":2.5, "E":2.5, "F":3.}
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

        self.buffer_time = {"A":{"A":0., "B":0., "C":0., "D":0., "E":0., "F":0.},
                            "B":{"A":0., "B":0., "C":0., "D":0., "E":0., "F":0.},
                            "C":{"A":0., "B":0., "C":0., "D":0., "E":0., "F":0.},
                            "D":{"A":0., "B":0., "C":0., "D":0., "E":0., "F":0.},
                            "E":{"A":0., "B":0., "C":0., "D":0., "E":0., "F":0.},
                            "F":{"A":0., "B":0., "C":0., "D":0., "E":0., "F":0.}
                            }

    def get_app_dist_sep(self, cat_leader, cat_follower):
        """This function  determines the distance-based separation minimmal on approach by pair of aircraft (LEADER, FOLLOWER).
        The values are based on the RECAT-EU document"""
        return unit.m_NM(self.approach_separation[cat_leader][cat_follower])

    def get_app_speed(self, seg, index):
        """This function  determines the approach speed of a given segment
        index = 0 means that minimum speed is taken
        index = 1 means that maximum speed is taken
        index = x, 0<x<1, an intermediate value is taken
        """
        return (1.-index)*self.app_speed[seg][0] + index*self.app_speed[seg][1]

    def seg_from_pax(self,npax):
        for k,r in self.capacity.items():
            if r[0]<=npax and npax<=r[1]:
                return k
        raise Exception("Cannot categorize number of passenger")



class Airport(Categories):

    def __init__(self):
        super(Airport, self).__init__()

        self.n_runway = 2.
        self.open_slot = [6., 24.]
        self.approach_dist = unit.m_NM(5.)

    def insert_segment(self,ac_list):
        r = 0.
        for k in ac_list.keys():
            r += ac_list[k]["ratio"]
            seg = self.seg_from_pax(ac_list[k]["npax"])
            ac_list[k]["seg"] = self.seg_from_pax(ac_list[k]["npax"])
        if r != 1.:
            raise Exception("Sum of aircraft distribution ratios is different from 1")
        return len(ac_list)

    def get_mean_ot(self,ac_list):
        """Evaluate the mean occupation time of one runway according to the aircraft distribution
        """
        # Mean number of passenger by airplane movement
        mean_pax = 0.
        for ac in ac_list.keys():
            mean_pax += ac_list[ac]["npax"] * ac_list[ac]["ratio"]

        nac = self.insert_segment(ac_list)

        app_dist_separation = np.empty((nac,nac))
        to_time_separation = np.empty((nac,nac))
        probability = np.empty((nac,nac))
        buffer = np.empty((nac,nac))
        for jl,acl in enumerate(ac_list.keys()):
            catl = self.wakevortex[ac_list[acl]["seg"]]
            for jf,acf in enumerate(ac_list.keys()):
                catf = self.wakevortex[ac_list[acf]["seg"]]
                probability[jl,jf] = ac_list[acl]["ratio"]*ac_list[acf]["ratio"]
                app_dist_separation[jl,jf] = self.get_app_dist_sep(catl, catf)
                to_time_separation[jl,jf] = self.takeoff_separation[catl][catf]
                buffer[jl,jf] = self.buffer_time[catl][catf]

        # Compute the minimum time interval between 2 successive aircraft in approach
        # There are 2 cases: closing case (V_i <= V_j) and opening case (V_i > V_j)
        time_separation = np.empty((nac,nac))
        for jl,acl in enumerate(ac_list.keys()):
            vappl = self.get_app_speed(ac_list[acl]["seg"], 0.5)
            rotl = self.r_o_t[ac_list[acl]["seg"]]
            for jf,acf in enumerate(ac_list.keys()):
                vappf = self.get_app_speed(ac_list[acf]["seg"], 0.5)
                if vappl > vappf:
                    t = max((app_dist_separation[jl,jf]/vappf + self.approach_dist*(1./vappf - 1./vappl)), rotl) # Opening Case
                else:
                    t = max((app_dist_separation[jl,jf]/vappf), rotl) # Closing Case
                time_separation[jl,jf] = t

        mean_landing_ot = sum(sum((time_separation + buffer) * probability))
        mean_take_off_ot = sum(sum(to_time_separation * probability))

        return mean_pax, mean_landing_ot, mean_take_off_ot




ap = Airport()

ac_listribution = {"ac1":{"npax":70. , "ratio":0.30},
                   "ac2":{"npax":150., "ratio":0.50},
                   "ac3":{"npax":300., "ratio":0.15},
                   "ac4":{"npax":400., "ratio":0.05}
                  }

px, ld_ot, to_ot = ap.get_mean_ot(ac_listribution)

print(px)
print(ld_ot)
print(to_ot)


