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



class RoadNetwork(object):
    """Road network
    """
    def __init__(self):
        self.distance_ratio = 1.15     # Road distance over great circle distance ratio

    def road_distance(self, r0, r1, d):
        """Compute the mean distance between a focal point and a cloud of points arranged into a ring
        Travels are supposed direct
        :param r0: internal radius of the ring (can be zero)
        :param r1: external radius of the ring (must be greater than r0)
        :param d: distance between the center of the ring and the focal point
        :return: Mean distance per travel
        """
        # n: Number of points within the radius
        # m: Number of points in the circonference
        n,m = 100,100
        dist = 0.
        for i in range(n):
            for j in range(m):
                dist += np.sqrt( ((r0+(r1-r0)*((1+i)/n))*np.cos(2*np.pi*(j/m)) - d)**2 + ((r0+(r1-r0)*((1+i)/n))*np.sin(2*np.pi*(j/m)))**2 )
        return (dist*self.distance_ratio)/(m*n)


class RailNetwork(object):
    """Bus station
    """
    def __init__(self):
        pass

class TrainFleet(object):
    """Train fleet
    """
    def __init__(self):
        self.set_capacity = 380.                      # passenger
        self.load_factor = 0.5
        self.set_efficiency = unit.Jpm_kWhpkm(20.)    # J/m

    def get_travel_energy(self, pass_flow, distance):
        n_set = pass_flow / (self.set_capacity*self.load_factor)
        energy = n_set * distance * self.set_efficiency
        return energy


class TaxiFleet(object):
    """Taxi fleet
    """
    def __init__(self):

        h = 0.
        b = 0.
        g = 1.-h-b
        self.fleet = {"hydrogen":h, "battery":b, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 1.    # passenger

        gas_efficiency = (7.)*1.e-8*earth.fuel_density("gasoline")*earth.fuel_heat("gasoline")  # 7. L/100km
        self.efficiency = {"hydrogen":0., "battery":0., "gasoline":gas_efficiency}    # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["hydrogen"] = hydrogen
        self.fleet["battery"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        pk = pass_flow * distance
        energy = 0.
        for type in self.fleet.keys():
            energy += pk * self.efficiency[type] * self.fleet[type]
        return energy


class BusFleet(object):
    """Bus fleet
    """
    def __init__(self):
        h = 0.
        b = 0.
        g = 1.-h-b
        self.fleet = {"hydrogen":h, "battery":b, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 64.    # passenger
        gas_efficiency = (30.)*1.e-8*earth.fuel_density("gasoline")*earth.fuel_heat("gasoline") # 30. L/100km
        self.efficiency = {"hydrogen":0., "electricity":0., "gasoline":gas_efficiency}     # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["hydrogen"] = hydrogen
        self.fleet["battery"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        pk = pass_flow * distance
        energy = 0.
        for type in self.fleet.keys():
            energy += pk * self.efficiency[type] * self.fleet[type]
        return energy


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
                           "ring1": {"radius": 10.e3,
                                    "density": 250./unit.m2_km2(1.)},
                           "ring2": {"radius": 20.e3,
                                      "density": 50./unit.m2_km2(1.)},
                           "ring3": {"radius": self.radius,
                                      "density": None}
                           }

        self.population["town"]["area"] = np.pi * self.population["town"]["radius"]**2
        self.population["town"]["inhab"] =  self.population["town"]["density"] / self.population["town"]["area"]

        self.population["ring1"]["area"] = np.pi * (self.population["ring1"]["radius"]**2 - self.population["town"]["radius"]**2)
        self.population["ring1"]["inhab"] = self.population["ring1"]["density"] / self.population["ring1"]["area"]

        self.population["ring2"]["area"] = np.pi * (self.population["ring2"]["radius"]**2 - self.population["ring1"]["radius"]**2)
        self.population["ring2"]["inhab"] = self.population["ring2"]["density"] / self.population["ring2"]["area"]

        self.population["ring3"]["area"] = np.pi * (self.population["ring3"]["radius"]**2 - self.population["ring2"]["radius"]**2)
        third_ring_inhab = self.inhabitant - self.population["town"]["inhab"] - self.population["ring1"]["inhab"] - self.population["ring2"]["inhab"]
        if third_ring_inhab<0.:
            raise Exception("inner rings population is not compatible with territory total population")
        self.population["ring3"]["density"] = third_ring_inhab / self.population["ring3"]["area"]
        self.population["ring3"]["inhab"] = third_ring_inhab

        # Probability for a territory habitant to be in a given area
        for area in self.population.keys():
            self.population[area]["prob"] = self.population[area]["inhab"] / self.inhabitant

        # Complete the territory
        self.airport = Airport()
        self.rail_network = RailNetwork()
        self.road_network = RoadNetwork()
        self.train_fleet = TrainFleet()
        self.taxi_fleet = TaxiFleet()
        self.bus_fleet = BusFleet()

    def service_energy(self, pass_flow):

        r0 = 0.
        dist = 0.
        for area in self.population.keys():
            r1 = self.population[area]["radius"]
            dist +=   self.road_network.road_distance(self, r0, r1, self.airport.town_distance) \
                    * self.population[area]["prob"]
            r0 = r1




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




