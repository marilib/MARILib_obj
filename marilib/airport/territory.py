#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 20 20:20:20 2020
@author: Cong Tam DO, Thierry DRUOT
"""

import numpy as np
from scipy.optimize import fsolve
from marilib.utils.math import lin_interp_1d, maximize_1d

from marilib.utils import unit, earth

from marilib.airport.aircraft import AirplaneCategories, Aircraft

from marilib.airport.airport import Airport


class TaxiNetwork(object):
    """Bus station
    """
    def __init__(self):
        self.h2_ratio = 0.
        self.battery_ratio = 0.
        self.gasoline_ratio = 1. - (self.h2_ratio + self.battery_ratio)

        self.taxi_capacity = 1.    # passenger
        gas_efficiency = (7.)*1.e-8*earth.fuel_density("gasoline")*earth.fuel_heat("gasoline")  # 7. L/100km
        self.taxi_efficiency = {"hydrogene":0., "electricity":0., "gasoline":gas_efficiency}    # J/m

    def set_energy_ratio(self, h2_ratio, battery_ratio):
        self.h2_ratio = h2_ratio
        self.battery_ratio = battery_ratio
        self.gasoline_ratio = 1. - (self.h2_ratio + self.battery_ratio)


class BusNetwork(object):
    """Bus station
    """
    def __init__(self):
        self.h2_ratio = 0.
        self.battery_ratio = 0.
        self.gasoline_ratio = 1. - (self.h2_ratio + self.battery_ratio)

        self.bus_capacity = 64.    # passenger
        gas_efficiency = (30.)*1.e-8*earth.fuel_density("gasoline")*earth.fuel_heat("gasoline") # 30. L/100km
        self.bus_efficiency = {"hydrogene":0., "electricity":0., "gasoline":gas_efficiency}     # J/m

    def set_energy_ratio(self, h2_ratio, battery_ratio):
        self.h2_ratio = h2_ratio
        self.battery_ratio = battery_ratio
        self.gasoline_ratio = 1. - (self.h2_ratio + self.battery_ratio)


class RailNetwork(object):
    """Bus station
    """
    def __init__(self):
        self.train_set_capacity = 380.                      # passenger
        self.train_set_efficiency = unit.Jpm_kWhpkm(20.)    # J/m


class Territory(object):
    """Territory characteristics
    The shape of the territory is idealized as à disk with the town in its center
    A territory is defined by a radius and a given global population
    Four population areas are distinguished :
    - the town center of radius rt has a constant population density
    - the first ring of radius r1 has a constant population density
    - the second ring of diameter r2 has a constant population density
    The territory includes one airport which is located at a given distance from the center
    it also contains a road network that connect all habitations, a bus network and a rail network
    Each network connects a given ratio of the habitation of each ring to the town center
    """

    def __init__(self, airport, taxi_network, bus_network, rail_network):

        self.radius = 60.e3     # 50 km
        self.inhabitant = 2.e6  # 2 million

        self.area = np.pi * self.radius**2
        self.density = self.population / self.area

        # Set population distribution
        self.population = {"town": {"radius": 4.e3,
                                    "density": 1500./unit.m2_km2(1.)},
                           "first": {"radius": 10.e3,
                                    "density": 250./unit.m2_km2(1.)},
                           "second": {"radius": 20.e3,
                                      "density": 50./unit.m2_km2(1.)},
                           "third": {"radius": self.radius,
                                      "density": None}
                           }

        self.population["town"]["area"] = np.pi * self.population["town"]["radius"]**2
        self.population["town"]["inhab"] =  self.population["town"]["density"] / self.population["town"]["area"]
        self.population["town"]["dist"] = 0.

        self.population["first"]["area"] = np.pi * (self.population["first"]["radius"]**2 - self.population["town"]["radius"]**2)
        self.population["first"]["inhab"] = self.population["first"]["density"] / self.population["first"]["area"]
        self.population["first"]["dist"] = 0.5 * (self.population["first"]["radius"] + self.population["town"]["radius"])

        self.population["second"]["area"] = np.pi * (self.population["second"]["radius"]**2 - self.population["first"]["radius"]**2)
        self.population["second"]["inhab"] = self.population["second"]["density"] / self.population["second"]["area"]
        self.population["second"]["dist"] = 0.5 * (self.population["second"]["radius"] + self.population["first"]["radius"])

        self.population["third"]["area"] = np.pi * (self.population["third"]["radius"]**2 - self.population["second"]["radius"]**2)
        third_ring_inhab = self.inhabitant - self.population["town"]["inhab"] - self.population["first"]["inhab"] - self.population["second"]["inhab"]
        if third_ring_inhab<0.:
            raise Exception("inner rings population is not compatible with territory total population")
        self.population["third"]["density"] = third_ring_inhab / self.population["third"]["area"]
        self.population["third"]["inhab"] = third_ring_inhab
        self.population["third"]["dist"] = 0.5 * (self.population["third"]["radius"] + self.population["second"]["radius"])

        # Complete the territory
        self.airport = airport
        self.taxi_network = taxi_network
        self.bus_network = bus_network
        self.rail_network = rail_network

    def travel_distance(self, r0, r1, d):
        """Compute the mean distance between a focal point and a cloud of points arranged into a ring
        Travels are supposed direct
        :param r0: internal radius of the ring (can be zero)
        :param r1: external radius of the ring (must be greater than r0)
        :param n: Number of points within the radius
        :param m: Number of points in the circonference
        :param d: distance between the center of th edisk and the focal point
        :return: Mean distance per travel
        """
        n,m = 100,100
        dist = 0.
        for i in range(n):
            for j in range(m):
                dist += np.sqrt( ((r0+(r1-r0)*((1+i)/n))*np.cos(2*np.pi*(j/m)) - d)**2 + ((r0+(r1-r0)*((1+i)/n))*np.sin(2*np.pi*(j/m)))**2 )
        return dist/(m*n)



cat = AirplaneCategories()

# Only proportion and design capacity is required to design the airport
ac_list = [{"ratio":0.30, "npax":70. },
           {"ratio":0.50, "npax":150.},
           {"ratio":0.15, "npax":300.},
           {"ratio":0.05, "npax":400.}]

runway_count = 3
app_dist = unit.m_NM(7.)
town_dist = unit.m_km(20.)
open_slot = [unit.s_h(6.), unit.s_h(23.)]

ap = Airport(cat, ac_list, runway_count, open_slot, app_dist, town_dist)

ap.print_airport_design_data()
ap.print_component_design_data()



# For fuel evaluation, design range must be added
fleet = [Aircraft(cat, npax=70. , range=unit.m_NM(500.) , mach=0.50),
         Aircraft(cat, npax=150., range=unit.m_NM(3000.), mach=0.78),
         Aircraft(cat, npax=300., range=unit.m_NM(5000.), mach=0.85),
         Aircraft(cat, npax=400., range=unit.m_NM(7000.), mach=0.85)]

# Defines the load factor and the route distribution for each airplane
network = [{"ratio":0.30, "load_factor":0.85, "route":[[0.25, unit.m_NM(100.)], [0.5, unit.m_NM(200.)], [0.25, unit.m_NM(400.)]]},
           {"ratio":0.50, "load_factor":0.85, "route":[[0.50, unit.m_NM(400.)], [0.35, unit.m_NM(800)], [0.15, unit.m_NM(2000.)]]},
           {"ratio":0.15, "load_factor":0.85, "route":[[0.35, unit.m_NM(2000.)], [0.5, unit.m_NM(3500.)], [0.15, unit.m_NM(5500.)]]},
           {"ratio":0.05, "load_factor":0.85, "route":[[0.25, unit.m_NM(1500.)], [0.5, unit.m_NM(5000.)], [0.25, unit.m_NM(7500.)]]}]

capacity_ratio = 0.75   # Capacity ratio of the airport, 1. means that the airport is at full capacity

data_dict = ap.get_flows(capacity_ratio, fleet, network)

print("==============================================================================")
for j in range(len(fleet)):
    print("Daily movements (landing or take off) for airplane n°", 1+j, " = ", "%.0f"%data_dict["ac_count"][j])
    print("Daily passenger transported by airplane n°", 1+j, " = ", "%.0f"%(data_dict["ac_pax"][j]))
    print("Daily fuel delivered to airplane n°", 1+j, " = ", "%.0f"%(data_dict["ac_fuel"][j]/1000.), " t")
    print("")
print("Daily total passenger transported = ""%.0f"%(data_dict["total_pax"]))
print("Daily total fuel delivered = ""%.0f"%(data_dict["total_fuel"]/1000.), " t")
print("")




