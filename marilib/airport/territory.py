#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 20 20:20:20 2020
@author: Cong Tam DO, Thierry DRUOT
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as pat

from marilib.utils import unit

from marilib.airport.physical_data import PhysicalData
from marilib.airport.aircraft import AirplaneCategories, Fleet
from marilib.airport.airport import Airport

from marilib.airport.power_plant import PvPowerPlant, CspPowerPlant, EolPowerPlant, NuclearPowerPlant, EnergyMix
from marilib.airport.power_plant import PowerToFuel, PowerToHydrogen, FuelMix



class RoadNetwork(object):
    """Road network
    """
    def __init__(self):
        # Coverage defines the proportion of destinations addressed into each area, this represents the transport offer of the medium
        # Note that road network can reach any destinations in the whole territory area
        self.coverage = {"town":1.,
                         "ring1":1.,
                         "ring2":1.,
                         "ring3":1.}

        self.distance_ratio = 1.15     # Road distance over great circle distance ratio

    def distance(self, r0, r1, d):
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
        self.coverage = {"town":0.60,
                         "ring1":0.40,
                         "ring2":0.10,
                         "ring3":0.05}  # TO BE CHECKED

    def distance(self, r0, r1, d):
        """Compute the mean distance between a focal point and a cloud of points arranged into a ring
        IMPORTANT NOTE : All travels from or to the airport are supposed to pass via the town center
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
                dist += np.sqrt( ((r0+(r1-r0)*((1+i)/n))*np.cos(2*np.pi*(j/m)))**2 + ((r0+(r1-r0)*((1+i)/n))*np.sin(2*np.pi*(j/m)))**2 )
        return dist/(m*n)


class TrainFleet(object):
    """Train fleet
    """
    def __init__(self, phd):
        self.phd = phd

        self.capacity = 380                       # passengers
        self.load_factor = 0.50                   # Mean load factor within a day
        self.efficiency = unit.Jpm_kWhpkm(20.)    # J/m

    def get_travel_energy(self, pass_flow, distance):
        n_set = pass_flow / (self.capacity * self.load_factor)
        elec = n_set * distance * self.efficiency
        energy = {"electricity":elec}
        return energy


class BusNetwork(object):
    """Bus mine network
    """
    def __init__(self):
        # Coverage defines the proportion of destinations addressed into each area, this represents the transport offer of the medium
        # Note that the sum of these proportions is not necessarely equal to 1
        # Even if busses are using the road network they do not address all destinations
        self.coverage = {"town":0.80,
                         "ring1":0.25,
                         "ring2":0.10,
                         "ring3":0.05}


class BusFleet(object):
    """Bus fleet
    """
    def __init__(self, phd):
        self.phd = phd

        h = 0.
        b = 0.
        g = 1.-h-b
        self.fleet = {"electricity":b, "compressed_h2":h, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 100.        # passengers
        self.load_factor = 0.50     # Mean load factor within a day

        # Efficiencies refer to internal energy, use fuel energy density to compute fuel flow
        gas_eff = (32.)*1.e-8*self.phd.fuel_density("gasoline")*self.phd.fuel_heat("gasoline")  # 32. L/100km
        gh2_eff = gas_eff * (0.30/0.45)
        bat_eff = gas_eff * (0.30/0.90)
        self.efficiency = {"electricity":bat_eff, "compressed_h2":gh2_eff, "gasoline":gas_eff}    # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["compressed_h2"] = hydrogen
        self.fleet["electricity"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        n_bus = pass_flow / (self.capacity * self.load_factor)
        pk = n_bus * distance
        energy = {"electricity":0., "compressed_h2":0., "gasoline":0.}
        for type in self.fleet.keys():
            energy[type] = pk * self.efficiency[type] * self.fleet[type]
        return energy


class TaxiFleet(object):
    """Taxi fleet
    """
    def __init__(self, phd):
        self.phd = phd

        h = 0.
        b = 0.05
        g = 1.-h-b
        self.fleet = {"electricity":b, "compressed_h2":h, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 1.    # passenger

        # Efficiencies refer to internal energy, use fuel energy density to compute fuel flow
        gas_eff = (7.)*1.e-8*self.phd.fuel_density("gasoline")*self.phd.fuel_heat("gasoline")  # 7. L/100km
        gh2_eff = gas_eff * (0.30/0.45)
        bat_eff = gas_eff * (0.30/0.90)
        self.efficiency = {"electricity":bat_eff, "compressed_h2":gh2_eff, "gasoline":gas_eff}    # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["compressed_h2"] = hydrogen
        self.fleet["electricity"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        pk = pass_flow * distance
        energy = {"electricity":0., "compressed_h2":0., "gasoline":0.}
        for type in self.fleet.keys():
            energy[type] = pk * self.efficiency[type] * self.fleet[type]
        return energy


class CarFleet(object):
    """Car fleet
    """
    def __init__(self, phd):
        self.phd = phd

        h = 0.
        b = 0.01
        g = 1.-h-b
        self.fleet = {"electricity":b, "compressed_h2":h, "gasoline":g}    # Proportion of each type in taxi fleet

        self.capacity = 1.1    # passenger

        # Efficiencies refer to internal energy, use fuel energy density to compute fuel flow
        gas_eff = (7.)*1.e-8*self.phd.fuel_density("gasoline")*self.phd.fuel_heat("gasoline")  # 7. L/100km
        gh2_eff = gas_eff * (0.30/0.45)
        bat_eff = gas_eff * (0.30/0.90)
        self.efficiency = {"electricity":bat_eff, "compressed_h2":gh2_eff, "gasoline":gas_eff}    # J/m

    def set_energy_ratio(self, hydrogen, battery):
        self.fleet["compressed_h2"] = hydrogen
        self.fleet["electricity"] = battery
        self.fleet["gasoline"] = 1. - hydrogen - battery

    def get_travel_energy(self, pass_flow, distance):
        pk = pass_flow * distance
        energy = {"electricity":0., "compressed_h2":0., "gasoline":0.}
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

    def __init__(self, phd, airport, fuel_mix, energy_mix):
        self.phd = phd

        # Maximum distance where people are likely to use town services
        self.influence_radius = unit.m_km(100.)

        # Set population distribution TO BE CHECKED
        self.population = {"town": {"radius": unit.m_km(6.1),
                                    "density": 4050./unit.m2_km2(1.)},
                           "ring1": {"radius": unit.m_km(12.1),
                                     "density": 850./unit.m2_km2(1.)},
                           "ring2": {"radius": unit.m_km(22.),
                                     "density": 200./unit.m2_km2(1.)},
                           "ring3": {"radius": unit.m_km(40.),
                                     "density": 100./unit.m2_km2(1.)}
                           }

        self.population["town"]["area"] = np.pi * self.population["town"]["radius"]**2
        self.population["town"]["inhab"] =  self.population["town"]["density"] * self.population["town"]["area"]

        self.population["ring1"]["area"] = np.pi * (self.population["ring1"]["radius"]**2 - self.population["town"]["radius"]**2)
        self.population["ring1"]["inhab"] = self.population["ring1"]["density"] * self.population["ring1"]["area"]

        self.population["ring2"]["area"] = np.pi * (self.population["ring2"]["radius"]**2 - self.population["ring1"]["radius"]**2)
        self.population["ring2"]["inhab"] = self.population["ring2"]["density"] * self.population["ring2"]["area"]

        self.population["ring3"]["area"] = np.pi * (self.population["ring3"]["radius"]**2 - self.population["ring2"]["radius"]**2)
        self.population["ring3"]["inhab"] = self.population["ring3"]["density"] * self.population["ring3"]["area"]

        self.area = np.pi * self.population["ring3"]["radius"]**2
        self.inhabitant = 0.
        for area in self.population.keys():
            self.inhabitant += self.population[area]["inhab"]
        self.density = self.inhabitant / self.area

        # Probability for a territory habitant to be in a given area
        for area in self.population.keys():
            self.population[area]["prob"] = self.population[area]["inhab"] / self.inhabitant

        # Add transport components to the territory
        #---------------------------------------------------------------------------------------------------------------
        self.rail_network = RailNetwork()
        self.train_fleet = TrainFleet(phd)

        self.bus_network = BusNetwork()
        self.bus_fleet = BusFleet(phd)

        self.road_network = RoadNetwork()
        self.taxi_fleet = TaxiFleet(phd)
        self.car_fleet = TaxiFleet(phd)

        self.airport = airport

        self.transport_fuel_mix = fuel_mix

        self.transport_energy_mix = energy_mix
        self.power_ratio = None

    def design(self, town_airport_dist, fleet, air_network, power_technology, capacity_ratio, elec_ratio, power_ratio):

        # Design the airport
        self.airport.design(town_airport_dist, fleet.profile)

        energy_dict = tr.get_transport_energy(capacity_ratio, fleet, air_network, power_technology)

        air_fuel = energy_dict["air_fuel"]
        ground_fuel = energy_dict["ground_fuel"]

        fuel_mix = {}
        for f in ground_fuel.keys():
            fuel_mix[f] = ground_fuel[f]
        for f in air_fuel.keys():
            fuel_mix[f] = air_fuel[f]

        data_dict = self.transport_fuel_mix.design(elec_ratio, fuel_mix)

        total_power = data_dict["electricity"] / (3600. * 24.)    # Energy per day must be converted into mean power

        self.power_ratio = power_ratio

        power_mix = {}
        for k,pr in self.power_ratio.items():
            power_mix[k] = total_power * pr

        load_factor = 1.

        self.transport_energy_mix.design(load_factor, power_mix)

    def get_air_transport_fuel_flow(self, capacity_ratio, fleet, air_network, power_technology):

        energy_dict = tr.get_transport_energy(capacity_ratio, fleet, air_network, power_technology)

        air_fuel = energy_dict["air_fuel"]
        ground_fuel = energy_dict["ground_fuel"]

        fuel_mix = {}
        for f in ground_fuel.keys():
            fuel_mix[f] = ground_fuel[f]
        for f in air_fuel.keys():
            fuel_mix[f] = air_fuel[f]

        data_dict = self.transport_fuel_mix.operate(fuel_mix)

        energy_dict["total_flows"] = data_dict

        return energy_dict


    def get_transport_energy(self, capacity_ratio, fleet, air_network, technology):

        # Compute the passenger flow
        airport_flows = self.airport.operate(capacity_ratio, fleet, air_network)

        # Rough evaluation of the energy consumption versus propulsive technology
        fleet_fuel = airport_flows["fleet_fuel"]

        # All energies consumed by airplanes
        at_energy = {"electricity":0., "compressed_h2":0., "liquid_h2":0., "kerosene":0.}

        # Relative efficiency versus kerosene
        relative_efficiency = {"electricity":0.36/0.95, "compressed_h2":0.36/0.55, "liquid_h2":0.36/0.55, "kerosene":1.}

        for seg,tech in technology.items():
            for type in tech.keys():
                at_energy[type] = fleet_fuel[seg] * self.phd.fuel_heat("kerosene") * tech[type] * relative_efficiency[type]

        at_fuel = {"compressed_h2":0., "liquid_h2":0., "kerosene":0.}
        for f in at_fuel.keys():
            at_fuel[f] = at_energy[f] / self.phd.fuel_heat(f)

        # Number of passenger going to AND coming from the airport
        pass_flow = airport_flows["total_pax"] * 2

        # Compute mean distance between the airport and population areas
        # Of course rail distance will be applied only to reacheable destinations
        r0 = 0.
        road_dist = {"town":0., "ring1":0., "ring2":0., "ring3":0.}
        rail_dist = {"town":0., "ring1":0., "ring2":0., "ring3":0.}
        for area in self.population.keys():
            r1 = self.population[area]["radius"]
            road_dist[area] =   self.road_network.distance(r0, r1, self.airport.town_distance)
            rail_dist[area] =   self.rail_network.distance(r0, r1, self.airport.town_distance)
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
        gt_energy = {"electricity":0., "compressed_h2":0., "liquid_h2":0., "gasoline":0.}

        # Energies consumed by taxis
        for area in offer.keys():
            pass_flow = taxi_pass_flow * self.population[area]["prob"]
            distance = road_dist[area]
            enrg = self.taxi_fleet.get_travel_energy(pass_flow, distance)
            for type in enrg.keys():
                gt_energy[type] += enrg[type]

        # Energies consumed by others transportation means
        for area in offer.keys():

            # Energy consumed by cars
            pass_flow = cbtr_pass_flow * self.population[area]["prob"] * offer[area]["car"]
            distance = road_dist[area]
            enrg = self.car_fleet.get_travel_energy(pass_flow, distance)
            for type in enrg.keys():
                gt_energy[type] += enrg[type]

            # Energy consumed by busses
            pass_flow = cbtr_pass_flow * self.population[area]["prob"] * offer[area]["bus"]
            distance = road_dist[area]
            enrg = self.bus_fleet.get_travel_energy(pass_flow, distance)
            for type in enrg.keys():
                gt_energy[type] += enrg[type]

            # Energy consumed by trains
            pass_flow = cbtr_pass_flow * self.population[area]["prob"] * offer[area]["train"]
            distance = rail_dist[area]
            enrg = self.train_fleet.get_travel_energy(pass_flow, distance)
            for type in enrg.keys():
                gt_energy[type] += enrg[type]

        gt_fuel = {"compressed_h2":0., "liquid_h2":0., "gasoline":0.}
        for f in gt_fuel.keys():
            gt_fuel[f] = gt_energy[f] / self.phd.fuel_heat(f)

        return {"airport_flows":airport_flows, "air_energy":at_energy, "air_fuel":at_fuel, "ground_energy":gt_energy, "ground_fuel":gt_fuel}

    def print(self):
        print("Territory")
        print("==============================================================================")
        print("Total population = ", "%.0f"%self.inhabitant)
        print("Total area = ", "%.0f"%unit.km2_m2(self.area))
        print("Total density = ", "%.0f"%(self.density/unit.km2_m2(1.)))
        print("")
        for area,info in self.population.items():
            print(area,", population = ","%.0f"%info["inhab"], ", area = " "%.0f"%unit.km2_m2(info["area"])," m2")
        print("")


    def draw(self):
        """Draw the territory
        """
        window_title = "My Territory"
        plot_title = "This territory"

        xmax = 120.
        ymax = 120.
        margin = 0.

        origin = (0., 0.)

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(window_title)
        fig.suptitle(plot_title, fontsize=14)
        axes.set_aspect('equal', 'box')
        axes.set_xbound(-xmax, xmax+margin)
        axes.set_ybound(-ymax, ymax)

        ring4 = plt.Circle(origin, unit.km_m(self.influence_radius), color="whitesmoke", label="Influence")
        ring3 = plt.Circle(origin, unit.km_m(self.population["ring3"]["radius"]), color="palegreen", label="%.0f"%(self.population["ring3"]["density"]/unit.km2_m2(1.))+" h/km2")
        ring2 = plt.Circle(origin, unit.km_m(self.population["ring2"]["radius"]), color="lightblue", label="%.0f"%(self.population["ring2"]["density"]/unit.km2_m2(1.))+" h/km2")
        ring1 = plt.Circle(origin, unit.km_m(self.population["ring1"]["radius"]), color="plum", label="%.0f"%(self.population["ring1"]["density"]/unit.km2_m2(1.))+" h/km2")
        town = plt.Circle(origin, unit.km_m(self.population["town"]["radius"]), color="fuchsia", label="%.0f"%(self.population["town"]["density"]/unit.km2_m2(1.))+" h/km2")

        airport_azimut = unit.rad_deg(65.)  # Angle between the North and the direction town center -> airport

        airport_loc = (origin[0]-unit.km_m(self.airport.town_distance*np.sin(airport_azimut)),
                       origin[1]+unit.km_m(self.airport.town_distance*np.cos(airport_azimut)))

        airport = plt.Circle(airport_loc, unit.km_m(0.5*self.airport.overall_width), color="red", label="Airport")


        area_pv = unit.km2_m2(tr.transport_energy_mix.tech_mix["photovoltaic"].total_footprint)
        rr = 0.50   # Proportion of the width of the influence area ring
        rad_pv = (1.-rr)*unit.km_m(self.population["ring3"]["radius"]) + rr*unit.km_m(self.influence_radius)
        wid_pv = 0.25*unit.km_m(self.influence_radius - self.population["ring3"]["radius"])
        tht_pv = [45., 135., 225., 315.]

        area_wt = unit.km2_m2(tr.transport_energy_mix.tech_mix["wind_turbine"].total_footprint)
        rr = 0.75   # Proportion of the width of the influence area ring
        rad_wt = (1.-rr)*unit.km_m(self.population["ring3"]["radius"]) + rr*unit.km_m(self.influence_radius)
        wid_wt = 0.25*unit.km_m(self.influence_radius - self.population["ring3"]["radius"])
        tht_wt = [0., 90., 180., 270.]

        axes.add_artist(ring4)

        for t in tht_pv:
            theta2 = t + (360.*(area_pv/4)) / (np.pi*wid_pv*(2.*rad_pv-wid_pv))
            wedge_pv = pat.Wedge(origin, rad_pv, t, theta2, width=wid_pv, color="cornflowerblue", label="PV farms")
            axes.add_artist(wedge_pv)

        for t in tht_wt:
            theta2 = t + (360.*(area_wt/4)) / (np.pi*wid_wt*(2.*rad_wt-wid_wt))
            wedge_wt = pat.Wedge(origin, rad_wt, t, theta2, width=wid_wt, color="palegoldenrod", label="Wind farms")
            axes.add_artist(wedge_wt)

        axes.add_artist(ring3)
        axes.add_artist(ring2)
        axes.add_artist(ring1)
        axes.add_artist(town)
        axes.add_artist(airport)

        axes.legend(handles=[town,ring1,ring2,ring3,airport,wedge_wt,wedge_pv], loc="upper right")

        plt.show()



if __name__ == "__main__":

    # Tool objects
    #-----------------------------------------------------------------------------------------------------------------------
    phd = PhysicalData()
    cat = AirplaneCategories()


    # Fleet definition
    #-----------------------------------------------------------------------------------------------------------------------
    fleet_profile = {    "regional":{"ratio":0.30, "npax":70. , "range":unit.m_NM(500.) , "mach":0.50},
                      "short_range":{"ratio":0.50, "npax":150., "range":unit.m_NM(3000.), "mach":0.78},
                     "medium_range":{"ratio":0.15, "npax":300., "range":unit.m_NM(5000.), "mach":0.85},
                       "long_range":{"ratio":0.05, "npax":400., "range":unit.m_NM(7000.), "mach":0.85}}

    fleet = Fleet(phd,cat,fleet_profile)

    # Defines the load factor and the route distribution for each airplane
    air_network = {    "regional":{"load_factor":0.95, "route":[[0.25, unit.m_NM(100.)], [0.5, unit.m_NM(200.)], [0.25, unit.m_NM(400.)]]},
                    "short_range":{"load_factor":0.85, "route":[[0.50, unit.m_NM(400.)], [0.35, unit.m_NM(800)], [0.15, unit.m_NM(2000.)]]},
                   "medium_range":{"load_factor":0.85, "route":[[0.35, unit.m_NM(2000.)], [0.5, unit.m_NM(3500.)], [0.15, unit.m_NM(5500.)]]},
                     "long_range":{"load_factor":0.85, "route":[[0.25, unit.m_NM(1500.)], [0.5, unit.m_NM(5000.)], [0.25, unit.m_NM(7500.)]]}}

    # Defines the proportion of each technology in each aircraft type of the fleet
    power_technology = {    "regional":{"electricity":0., "liquid_h2":0., "kerosene":1.},
                         "short_range":{"electricity":0., "liquid_h2":0., "kerosene":1.},
                        "medium_range":{"electricity":0., "liquid_h2":0., "kerosene":1.},
                          "long_range":{"electricity":0., "liquid_h2":0., "kerosene":1.}}


    # Airport
    #-----------------------------------------------------------------------------------------------------------------------
    runway_count = 3
    app_dist = unit.m_NM(7.)
    open_slot = [unit.s_h(6.), unit.s_h(23.)]

    # Instantiate an airport component
    ap = Airport(cat, runway_count, open_slot, app_dist)

    # ap is not design here, it will be design with the territory


    # Air transport fuel mix
    #-----------------------------------------------------------------------------------------------------------------------
    compressed_h2 = PowerToHydrogen(phd, "compressed_h2")

    liquid_h2 = PowerToHydrogen(phd, "liquid_h2")

    gasoline = PowerToFuel(phd, fuel_type="gasoline", co2_capture="air")

    kerosene = PowerToFuel(phd, fuel_type="kerosene", co2_capture="air")

    fuel_mix = {"compressed_h2":compressed_h2, "liquid_h2":liquid_h2, "gasoline":gasoline, "kerosene":kerosene}

    # Instantiate a fuel mix production system component
    fp = FuelMix(phd, fuel_mix)

    # fp is not design here because the amount of fuel required is not yet known


    # Air transport energy mix
    #-----------------------------------------------------------------------------------------------------------------------
    n_panel = 1.e6      # Number of panels of 2m2 in one plant (Cestas)
    sol_pw = 250.       # W/m2, Mean solar irradiation
    reg_factor = 0.5    # Regulation factor, 0.:no storage, 1.:regulation period is 24h
    pv = PvPowerPlant(n_panel, sol_pw, reg_factor=reg_factor)

    n_turbine = 20.             # Number of turbines in one plant
    peak_power = unit.W_MW(5.)  # MW
    wind = EolPowerPlant(n_turbine, "onshore", rotor_peak_power=peak_power)

    n_core = 4      # Number of reactor in one plant
    atom = NuclearPowerPlant(n_core)

    tech_mix =  {"photovoltaic":pv,   "wind_turbine":wind}

    # Instantiate an energy mix
    em = EnergyMix(tech_mix)

    # em is not designed here because the amount of energy required is not yet known


    # Territory
    #-----------------------------------------------------------------------------------------------------------------------
    tr = Territory(phd, ap, fp, em)

    # Distance between center town and airport
    town_airport_dist = unit.m_km(8.)

    # Capacity ratio of the airport, 1. means that the airport is at full capacity
    capacity_ratio = 0.65

    # Ratio of each fuel type produced with electricity as primary energy
    elec_ratio = {"compressed_h2":1., "liquid_h2":1., "gasoline":1., "kerosene":1.}

    # Ratio of the total electric energy demand delivered by each power plant type
    power_ratio = {"photovoltaic":0.50, "wind_turbine":0.50}

    # The design of tr will include the design of ap, fp and em
    tr.design(town_airport_dist, fleet, air_network, power_technology, capacity_ratio, elec_ratio, power_ratio)

    # Airport data
    tr.airport.print_airport_design_data()
    tr.airport.print_component_design_data()
    ap.draw()

    # Territory
    tr.print()
    tr.transport_energy_mix.print()
    tr.draw()








    # Print One day of air transport operation
    #-------------------------------------------------------------------------------------------------------------------
    capacity_ratio = 0.75   # Capacity ratio of the airport, 1. means that the airport is at full capacity

    data_dict = tr.get_air_transport_fuel_flow(capacity_ratio, fleet, air_network, power_technology)

    print("One day of air transport operation")
    print("==============================================================================")
    for seg in fleet.segment:
        print("Airplane movements (landing or take off) for ", seg, " = ", "%.0f"%data_dict["airport_flows"]["fleet_count"][seg])
        print("Passenger transported by ",seg, " = ", "%.0f"%(data_dict["airport_flows"]["fleet_pax"][seg]))
        print("Fuel equivalent kerosene delivered to ", seg, " = ", "%.0f"%(data_dict["airport_flows"]["fleet_fuel"][seg]/1000.), " t")
        print("")
    print("Total passenger going to or coming from the airport = ""%.0f"%(data_dict["airport_flows"]["total_pax"]))
    print("Total fuel delivered = ""%.0f"%(data_dict["airport_flows"]["total_fuel"]/1000.), " t")
    print("")
    print("Energy consumed by ground transport as electricity = ", "%.0f"%unit.MWh_J(data_dict["ground_energy"]["electricity"]), " MWh")
    print("Energy consumed by ground transport as compressed hydrogen = ", "%.0f"%unit.MWh_J(data_dict["ground_energy"]["compressed_h2"]), " MWh")
    print("Energy consumed by ground transport as liquid hydrogen = ", "%.0f"%unit.MWh_J(data_dict["ground_energy"]["liquid_h2"]), " MWh")
    print("Energy consumed by ground transport as gasoline = ", "%.0f"%unit.MWh_J(data_dict["ground_energy"]["gasoline"]), " MWh")
    print("Total hydrogen mass consumed by ground transport = ", "%.0f"%unit.t_kg(data_dict["ground_fuel"]["liquid_h2"]), " t")
    print("Total gasoline mass consumed by ground transport = ", "%.0f"%unit.t_kg(data_dict["ground_fuel"]["gasoline"]), " t")
    print("")
    print("Energy consumed by air transport as electricity = ", "%.0f"%unit.MWh_J(data_dict["air_energy"]["electricity"]), " MWh")
    print("Energy consumed by air transport as hydrogen = ", "%.0f"%unit.MWh_J(data_dict["air_energy"]["liquid_h2"]), " MWh")
    print("Energy consumed by air transport as kerosene = ", "%.0f"%unit.MWh_J(data_dict["air_energy"]["kerosene"]), " MWh")
    print("Total hydrogen mass consumed by air transport = ", "%.0f"%unit.t_kg(data_dict["air_fuel"]["liquid_h2"]), " t")
    print("Total kerosene mass consumed by air transport = ", "%.0f"%unit.t_kg(data_dict["air_fuel"]["kerosene"]), " t")
    print("")
    print("Total energy consumed by air transport as electricity = ", "%.0f"%unit.MWh_J(data_dict["total_flows"]["electricity"]), " MWh")
    print("Total compressed hydrogen mass consumed = ", "%.0f"%unit.t_kg(data_dict["total_flows"]["compressed_h2"]), " t")
    print("Total gasoline mass consumed = ", "%.0f"%unit.t_kg(data_dict["total_flows"]["gasoline"]), " t")
    print("Total kerosene mass consumed = ", "%.0f"%unit.t_kg(data_dict["total_flows"]["kerosene"]), " t")



