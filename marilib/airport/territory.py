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
        # Coverage defines the proportion of destinations addressed into each area, this represents the transport offer of the medium
        # Note that road network can reach any destinations in all territory area
        self.coverage = {"town":1.,
                         "ring1":1.,
                         "ring2":1.,
                         "ring3":1.}

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
    """Train line network
    """
    def __init__(self):
        # Coverage defines the proportion of destinations addressed into each area, this represents the transport offer of the medium
        # Note that the sum of these proportions is not necessarely equal to 1
        self.coverage = {"town":0.90,
                         "ring1":0.10,
                         "ring2":0.01,
                         "ring3":0.01}  # TO BE CHECKED

    def rail_distance(self, r0, r1, d):
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
        dist = d            # WE SUPPOSE THAT ALL DSESTINATIONS IN THE TERRITORY IS CONNECTED TO THE TOWN, THGUS, PASSENGERS MUST REACH THE TOWN FIRST
        for i in range(n):
            for j in range(m):
                dist += np.sqrt( ((r0+(r1-r0)*((1+i)/n))*np.cos(2*np.pi*(j/m)) - d)**2 + ((r0+(r1-r0)*((1+i)/n))*np.sin(2*np.pi*(j/m)))**2 )
        return dist/(m*n)


class TrainFleet(object):
    """Train fleet
    """
    def __init__(self):
        self.capacity = 377                       # passengers
        self.load_factor = 0.50                   # Mean load factor within a day
        self.efficiency = unit.Jpm_kWhpkm(20.)    # J/m

    def get_travel_energy(self, pass_flow, distance):
        n_set = pass_flow / (self.capacity * self.load_factor)
        elec = n_set * distance * self.efficiency
        energy = {"hydrogen":0., "electric":elec, "gasoline":0.}
        return energy


class BusNetwork(object):
    """Bus mine network
    """
    def __init__(self):
        # Coverage defines the proportion of destinations addressed into each area, this represents the transport offer of the medium
        # Note that the sum of these proportions is not necessarely equal to 1
        # Even if busses are using the road network they do not address all destinations
        self.coverage = {"town":0.50,
                         "ring1":0.25,
                         "ring2":0.10,
                         "ring3":0.05}


class BusFleet(object):
    """Bus fleet
    """
    def __init__(self):
        h = 0.
        b = 0.
        g = 1.-h-b
        self.fleet = {"hydrogen":h, "electric":b, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 100.        # passengers
        self.load_factor = 0.50     # Mean load factor within a day
        gas_efficiency = (30.)*1.e-8*earth.fuel_density("gasoline")*earth.fuel_heat("gasoline") # 30. L/100km
        self.efficiency = {"hydrogen":0., "electric":0., "gasoline":gas_efficiency}     # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["hydrogen"] = hydrogen
        self.fleet["electric"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        n_bus = pass_flow / (self.capacity * self.load_factor)
        pk = n_bus * distance
        energy = {"hydrogen":0., "electric":0., "gasoline":0.}
        for type in self.fleet.keys():
            energy[type] = pk * self.efficiency[type] * self.fleet[type]
        return energy


class TaxiFleet(object):
    """Taxi fleet
    """
    def __init__(self):

        h = 0.
        b = 0.05
        g = 1.-h-b
        self.fleet = {"hydrogen":h, "electric":b, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 1.    # passenger

        # Efficiencies refer to internal energy, use fuel energy density to compute fuel flow
        gh2_eff = unit.J_kWh(34.*1.e-5)  # J/m, 34. kWh/100km
        bat_eff = unit.J_kWh(20.*1.e-5)  # J/m, 20. kWh/100km
        gas_eff = (7.)*1.e-8*earth.fuel_density("gasoline")*earth.fuel_heat("gasoline")  # 7. L/100km
        self.efficiency = {"hydrogen":gh2_eff, "electric":bat_eff, "gasoline":gas_eff}    # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["hydrogen"] = hydrogen
        self.fleet["electric"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        pk = pass_flow * distance
        energy = {"hydrogen":0., "electric":0., "gasoline":0.}
        for type in self.fleet.keys():
            energy[type] = pk * self.efficiency[type] * self.fleet[type]
        return energy


class CarFleet(object):
    """Car fleet
    """
    def __init__(self):

        h = 0.
        b = 0.01
        g = 1.-h-b
        self.fleet = {"hydrogen":h, "electric":b, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 1.1    # passenger

        # Efficiencies refer to internal energy, use fuel energy density to compute fuel flow
        gh2_eff = unit.J_kWh(34.*1.e-5)  # J/m, 34. kWh/100km
        bat_eff = unit.J_kWh(20.*1.e-5)  # J/m, 20. kWh/100km
        gas_eff = (7.)*1.e-8*earth.fuel_density("gasoline")*earth.fuel_heat("gasoline")  # 7. L/100km
        self.efficiency = {"hydrogen":gh2_eff, "electric":bat_eff, "gasoline":gas_eff}    # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["hydrogen"] = hydrogen
        self.fleet["electric"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        pk = pass_flow * distance
        energy = {"hydrogen":0., "electric":0., "gasoline":0.}
        for type in self.fleet.keys():
            energy[type] = pk * self.efficiency[type] * self.fleet[type]
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

    def __init__(self, airport):

        self.radius = 50.e3     # 50 km
        self.inhabitant = 2.e6  # 2 million

        self.area = np.pi * self.radius**2
        self.density = self.inhabitant / self.area

        # Set population distribution TO BE CHECKED
        self.population = {"town": {"radius": 4.e3,
                                    "density": 1500./unit.m2_km2(1.)},
                           "ring1": {"radius": 10.e3,
                                     "density": 1000./unit.m2_km2(1.)},
                           "ring2": {"radius": 20.e3,
                                     "density": 500./unit.m2_km2(1.)},
                           "ring3": {"radius": self.radius,
                                     "density": None}
                           }

        self.population["town"]["area"] = np.pi * self.population["town"]["radius"]**2
        self.population["town"]["inhab"] =  self.population["town"]["density"] * self.population["town"]["area"]

        self.population["ring1"]["area"] = np.pi * (self.population["ring1"]["radius"]**2 - self.population["town"]["radius"]**2)
        self.population["ring1"]["inhab"] = self.population["ring1"]["density"] * self.population["ring1"]["area"]

        self.population["ring2"]["area"] = np.pi * (self.population["ring2"]["radius"]**2 - self.population["ring1"]["radius"]**2)
        self.population["ring2"]["inhab"] = self.population["ring2"]["density"] * self.population["ring2"]["area"]

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
        self.rail_network = RailNetwork()
        self.train_fleet = TrainFleet()

        self.bus_network = BusNetwork()
        self.bus_fleet = BusFleet()

        self.road_network = RoadNetwork()
        self.taxi_fleet = TaxiFleet()
        self.car_fleet = TaxiFleet()

        self.airport = airport

    def get_transport_energy(self, capacity_ratio, fleet, air_network, technology):

        # Compute the passenger flow
        airport_flows = self.airport.get_flows(capacity_ratio, fleet, air_network)

        # Rough evaluation of the energy consumption versus propulsive technology
        fleet_fuel = airport_flows["fleet_fuel"]

        at_energy = {"battery":0., "hydrogen":0., "kerosene":1.}

        # Relative efficiency versus kerosene
        relative_efficiency = {"battery":0.36/0.95, "hydrogen":0.36/0.55, "kerosene":1.}

        for j,tech in enumerate(technology):
            for type in at_energy.keys():
                at_energy[type] = fleet_fuel[j] * earth.fuel_heat("kerosene") * tech[type] * relative_efficiency[type]

        # Number of passenger going to AND coming from the airport
        pass_flow = airport_flows["total_pax"] * 2

        # Compute mean distance between the airport and population areas
        # Of course rail distance will be applied only to reacheable destinations
        r0 = 0.
        road_dist = {"town":0., "ring1":0., "ring2":0., "ring3":0.}
        rail_dist = {"town":0., "ring1":0., "ring2":0., "ring3":0.}
        for area in self.population.keys():
            r1 = self.population[area]["radius"]
            road_dist[area] =   self.road_network.road_distance(r0, r1, self.airport.town_distance)
            rail_dist[area] =   self.rail_network.rail_distance(r0, r1, self.airport.town_distance)
            r0 = r1

        # Compute proportion of the total passenger flow that using taxi
        taxi_pass_flow = pass_flow * self.airport.passenger.taxi_ratio
        cbtr_pass_flow = pass_flow - taxi_pass_flow

        # Renormalize transport offer in each area between car, bus and train
        offer = {"town":{"car":0., "bus":0., "train":0.},
                 "ring1":{"car":0., "bus":0., "train":0.},
                 "ring2":{"car":0., "bus":0., "train":0.},
                 "ring3":{"car":0., "bus":0., "train":0.}}
        for area in offer.keys():
            total = self.road_network.coverage[area] + self.bus_network.coverage[area] + self.rail_network.coverage[area]
            offer[area]["car"] = self.road_network.coverage[area] / total
            offer[area]["bus"] = self.bus_network.coverage[area] / total
            offer[area]["train"] = self.rail_network.coverage[area] / total

        # All energies consumed by passengers coming from or going to the airport will be summed
        gt_energy = {"hydrogen":0., "electric":0., "gasoline":0.}

        # Energies consumed by taxis
        for area in offer.keys():
            pass_flow = taxi_pass_flow * self.population[area]["prob"]
            distance = road_dist[area]
            enrg = self.taxi_fleet.get_travel_energy(pass_flow, distance)
            for type in gt_energy.keys():
                gt_energy[type] += enrg[type]

        # Energies consumed by others transportation means
        for area in offer.keys():

            # Energy consumed by cars
            pass_flow = cbtr_pass_flow * self.population[area]["prob"] * offer[area]["car"]
            distance = road_dist[area]
            enrg = self.car_fleet.get_travel_energy(pass_flow, distance)
            for type in gt_energy.keys():
                gt_energy[type] += enrg[type]

            # Energy consumed by busses
            pass_flow = cbtr_pass_flow * self.population[area]["prob"] * offer[area]["bus"]
            distance = road_dist[area]
            enrg = self.bus_fleet.get_travel_energy(pass_flow, distance)
            for type in gt_energy.keys():
                gt_energy[type] += enrg[type]

            # Energy consumed by trains
            pass_flow = cbtr_pass_flow * self.population[area]["prob"] * offer[area]["train"]
            distance = rail_dist[area]
            enrg = self.train_fleet.get_travel_energy(pass_flow, distance)
            for type in gt_energy.keys():
                gt_energy[type] += enrg[type]

        return airport_flows, at_energy, gt_energy




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


tr = Territory(ap)

tr.airport.print_airport_design_data()
tr.airport.print_component_design_data()



# Fleet definition, all aircraft types are designed according to TLARs : npax, range, mach
fleet = [Aircraft(cat, npax=70. , range=unit.m_NM(500.) , mach=0.50),
         Aircraft(cat, npax=150., range=unit.m_NM(3000.), mach=0.78),
         Aircraft(cat, npax=300., range=unit.m_NM(5000.), mach=0.85),
         Aircraft(cat, npax=400., range=unit.m_NM(7000.), mach=0.85)]

# Defines the load factor and the route distribution for each airplane
air_network = [{"ratio":0.30, "load_factor":0.95, "route":[[0.25, unit.m_NM(100.)], [0.5, unit.m_NM(200.)], [0.25, unit.m_NM(400.)]]},
               {"ratio":0.50, "load_factor":0.85, "route":[[0.50, unit.m_NM(400.)], [0.35, unit.m_NM(800)], [0.15, unit.m_NM(2000.)]]},
               {"ratio":0.15, "load_factor":0.85, "route":[[0.35, unit.m_NM(2000.)], [0.5, unit.m_NM(3500.)], [0.15, unit.m_NM(5500.)]]},
               {"ratio":0.05, "load_factor":0.85, "route":[[0.25, unit.m_NM(1500.)], [0.5, unit.m_NM(5000.)], [0.25, unit.m_NM(7500.)]]}]

# Defines the proportion of each technology in each aircraft type of the fleet
technology = [{"battery":0., "hydrogen":0., "kerosene":1.},
              {"battery":0., "hydrogen":0., "kerosene":1.},
              {"battery":0., "hydrogen":0., "kerosene":1.},
              {"battery":0., "hydrogen":0., "kerosene":1.}]



capacity_ratio = 0.75   # Capacity ratio of the airport, 1. means that the airport is at full capacity

airport_flows, at_energy, gt_energy = tr.get_transport_energy(capacity_ratio, fleet, air_network, technology)

print("==============================================================================")
for j in range(len(fleet)):
    print("Daily movements (landing or take off) for airplane n°", 1+j, " = ", "%.0f"%airport_flows["fleet_count"][j])
    print("Daily passenger transported by airplane n°", 1+j, " = ", "%.0f"%(airport_flows["fleet_pax"][j]))
    print("Daily fuel delivered to airplane n°", 1+j, " = ", "%.0f"%(airport_flows["fleet_fuel"][j]/1000.), " t")
    print("")
print("Daily total passenger going to or coming from the airport = ""%.0f"%(airport_flows["total_pax"]))
print("Daily total fuel delivered = ""%.0f"%(airport_flows["total_fuel"]/1000.), " t")
print("")
print("Daily energy consumed by ground transport as hydrogen = ", "%.0f"%unit.MWh_J(gt_energy["hydrogen"]), " MWh")
print("Daily energy consumed by ground transport as electricity = ", "%.0f"%unit.MWh_J(gt_energy["electric"]), " MWh")
print("Daily energy consumed by ground transport as gasoline = ", "%.0f"%unit.MWh_J(gt_energy["gasoline"]), " MWh")
print("")
print("Daily energy consumed by air transport as hydrogen = ", "%.0f"%unit.MWh_J(at_energy["hydrogen"]), " MWh")
print("Daily energy consumed by air transport as electricity = ", "%.0f"%unit.MWh_J(at_energy["battery"]), " MWh")
print("Daily energy consumed by air transport as kerosene = ", "%.0f"%unit.MWh_J(at_energy["kerosene"]), " MWh")


