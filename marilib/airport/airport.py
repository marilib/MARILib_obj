#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from scipy.optimize import fsolve
from marilib.utils.math import lin_interp_1d, maximize_1d

from marilib.utils import unit, earth


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
        if type in ["design_range", "design_capacity", "design_mtow", "span", "tofl", "app_speed"]:
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


class AirportComponent(object):

    def __init__(self):

        self.area = None
        self.area_width = None
        self.area_length = None

        self.peak_power = None
        self.nominal_power = None
        self.daily_consumption = None

    def get_power_data(self):
        return {"peak":self.peak_power, "nominal":self.nominal_power, "energy_day":self.daily_consumption}

    def get_area(self):
        return {"length":self.area_length, "width":self.area_width, "area":self.area}


class Runways(AirportComponent):

    def __init__(self, count, length, open_slot):
        super(Runways, self).__init__()

        self.count = count
        self.runway_length = length
        self.runway_width = 45.

        self.area_length = self.runway_length + 1000.
        self.area_width = (self.runway_width + 200.) * self.count
        self.area = self.area_length * self.area_width

        self.peak_power = 20. * 2000. * self.count
        self.nominal_power = self.peak_power
        self.daily_consumption = self.nominal_power * max(0., open_slot[1]-open_slot[0]-10.)*3600.


class TaxiWays(AirportComponent):

    def __init__(self, runway_length):
        super(TaxiWays, self).__init__()

        self.area_length = runway_length
        self.area_width = 400.
        self.area = self.area_length * self.area_width

        self.peak_power = 0.0125 * self.area  # 12.5 mW/m2
        self.nominal_power = self.peak_power
        self.daily_consumption = self.nominal_power * max(0., open_slot[1]-open_slot[0]-10.)*3600.


class AirParks(AirportComponent):

    def __init__(self, max_ac_flow, mean_ac_span, terminal_length, terminal_width):
        super(AirParks, self).__init__()

        self.area = max_ac_flow * mean_ac_span**2
        self.area_length = terminal_width + terminal_width
        self.area_width = self.area / self.area_length

        self.peak_power = 0.0125 * self.area  # 12.5 mW/m2
        self.nominal_power = self.peak_power
        self.daily_consumption = self.nominal_power * max(0., open_slot[1]-open_slot[0]-10.)*3600.


class Terminals(AirportComponent):

    def __init__(self, max_pax_flow_capacity):
        super(Terminals, self).__init__()

        self.departure_capacity = max_pax_flow_capacity
        self.arrival_capacity = max_pax_flow_capacity

        self.area = 4.5 * (self.departure_capacity + self.arrival_capacity)
        self.area_width = np.sqrt(self.area/2.)     # A rectangle w*(2w)
        self.area_length = 2. * self.area_width

        self.peak_power = 1000. * self.area
        self.nominal_power = 0.90 * self.peak_power
        self.daily_consumption = self.nominal_power * max(0., open_slot[1]-open_slot[0])*3600.


class CarParks(AirportComponent):

    def __init__(self, max_pax_flow):
        super(CarParks, self).__init__()

        self.space_count = 0.15 * max_pax_flow

        self.area = 2.5 * 5.0 * self.space_count
        self.area_length = np.sqrt(self.area)
        self.area_width = self.area_length

        self.peak_power = 0.005 * self.area  # 5 mW/m2
        self.nominal_power = self.peak_power
        self.daily_consumption = self.nominal_power * 14.*3600.


class Airport(object):
    """Airport object is build with airport components
    It is sized according to the characteristics of the fleet described in ac_list
    """

    def __init__(self, airplane_categories, ac_list, n_runway, open_slot, app_dist):
        self.cat = airplane_categories

        self.approach_dist = app_dist
        self.open_slot = open_slot
        self.car_ratio = 0.4
        self.bus_ratio = 0.1
        self.rail_ratio = (1.-self.car_ratio-self.bus_ratio)

        # Get max runway length according to ac_list
        max_rnw_length = 0.
        for ac in ac_list.keys():
            max_rnw_length = max(max_rnw_length, self.cat.get_data_from_pax("tofl", ac_list[ac]["npax"])[1])

        # Load runway component
        self.runway = Runways(n_runway, max_rnw_length, open_slot)
        # Load taxiway component
        self.taxiway = TaxiWays(max_rnw_length)

        # Get mean aircraft span according to ac_list
        mean_ac_span = 0.
        for ac in ac_list.keys():
            mean_ac_span += self.cat.get_data_from_pax("span", ac_list[ac]["npax"], 0.5) * ac_list[ac]["ratio"]
        print(mean_ac_span)

        # Compute flows for ac_list
        dict = self.get_flows(ac_list, 1.)

        max_pax_flow = dict["pax_flow"]
        # Load terminal component
        self.terminal = Terminals(max_pax_flow)

        max_ac_flow = dict["ac_flow"]
        terminal_length = self.terminal.area_length
        terminal_width = self.terminal.area_width

        # Load air parks component
        self.air_parks = AirParks(max_ac_flow, mean_ac_span, terminal_length, terminal_width)

        self.air_service = None

        self.car_parks = CarParks(max_pax_flow)

        self.bus_station = None

        self.railway_station = None

    def insert_segment(self,ac_list):
        r = 0.
        for k in ac_list.keys():
            r += ac_list[k]["ratio"]
            ac_list[k]["seg"] = self.cat.seg_from_pax(ac_list[k]["npax"])
        if r != 1.:
            raise Exception("Sum of aircraft distribution ratios is different from 1")
        return len(ac_list)

    def get_flows(self, ac_list, capacity_ratio):
        """Evaluate the mean occupation time of one runway according to the aircraft distribution
        """
        # Mean number of passenger by airplane
        mean_ac_capacity = 0.
        for ac in ac_list.keys():
            mean_ac_capacity += ac_list[ac]["npax"] * ac_list[ac]["ratio"]

        nac = self.insert_segment(ac_list)

        app_dist_separation = np.empty((nac,nac))
        to_time_separation = np.empty((nac,nac))
        probability = np.empty((nac,nac))
        buffer = np.empty((nac,nac))
        for jl,acl in enumerate(ac_list.keys()):
            catl = self.cat.get_data_from_seg("wakevortex", ac_list[acl]["seg"])
            for jf,acf in enumerate(ac_list.keys()):
                catf = self.cat.get_data_from_seg("wakevortex", ac_list[acf]["seg"])
                probability[jl,jf] = ac_list[acl]["ratio"]*ac_list[acf]["ratio"]
                app_dist_separation[jl,jf] = self.cat.get_separation_data("approach_separation", catl, catf)
                to_time_separation[jl,jf] = self.cat.get_separation_data("takeoff_separation", catl, catf)
                buffer[jl,jf] = self.cat.get_separation_data("buffer_time", catl, catf)

        # Compute the minimum time interval between 2 successive aircraft in approach
        # There are 2 cases: closing case (V_i <= V_j) and opening case (V_i > V_j)
        time_separation = np.empty((nac,nac))
        for jl,acl in enumerate(ac_list.keys()):
            vappl = self.cat.get_data_from_seg("app_speed", ac_list[acl]["seg"])[0]
            rotl = self.cat.r_o_t[ac_list[acl]["seg"]]
            for jf,acf in enumerate(ac_list.keys()):
                vappf = self.cat.get_data_from_seg("app_speed", ac_list[acf]["seg"])[0]
                if vappl > vappf:
                    t = max((app_dist_separation[jl,jf]/vappf + self.approach_dist*(1./vappf - 1./vappl)), rotl) # Opening Case
                else:
                    t = max((app_dist_separation[jl,jf]/vappf), rotl) # Closing Case
                time_separation[jl,jf] = t

        mean_landing_ot = sum(sum((time_separation + buffer) * probability))
        mean_take_off_ot = sum(sum(to_time_separation * probability))

        # Number of aircraft passing on the airport during one day supposing that passenger traffic is balanced over a day
        daily_ac_movements = capacity_ratio * self.runway.count * (self.open_slot[1] - self.open_slot[0]) / (mean_landing_ot + mean_take_off_ot)

        # Daily passenger flow taking an airplane
        daily_pax_flow = mean_ac_capacity * daily_ac_movements

        dict = {"pax_flow":daily_pax_flow,
                "ac_flow":daily_ac_movements,
                "ac_mean_cap":mean_ac_capacity,
                "mean_ld_ot":mean_landing_ot,
                "mean_to_ot":mean_take_off_ot}

        return dict



cat = AirplaneCategories()

ac_list = {"ac1":{"ratio":0.30, "npax":70.},
           "ac2":{"ratio":0.50, "npax":150.},
           "ac3":{"ratio":0.15, "npax":300.},
           "ac4":{"ratio":0.05, "npax":400.}
           }

runway_count = 3
app_dist = unit.m_NM(7.)
open_slot = [unit.s_h(6.), unit.s_h(23.)]


ap = Airport(cat, ac_list, runway_count, open_slot, app_dist)

ac_distribution = {"ac1":{"ratio":0.30, "npax":70.},
                   "ac2":{"ratio":0.50, "npax":150.},
                   "ac3":{"ratio":0.15, "npax":300.},
                   "ac4":{"ratio":0.05, "npax":400.}
                  }

capacity_ratio = 0.75

dict = ap.get_flows(ac_distribution, capacity_ratio)

print("Passenger flow (input or output) = ", "%.0f"%dict["pax_flow"])
print("Aircraft movements (landing or take off) = ", "%.0f"%dict["ac_flow"])
print("")
print("Mean airplane capacity = ", "%.0f"%dict["ac_mean_cap"])
print("Mean runway occupation time at landing = ", "%.1f"%dict["mean_ld_ot"])
print("Mean runway occupation time at take off = ", "%.1f"%dict["mean_to_ot"])


