#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 20 20:20:20 2020
@author: Cong Tam DO, Thierry DRUOT
"""

import numpy as np

from marilib.utils import unit

import matplotlib.pyplot as plt

import marilib.airport.utils as tool

class AirportComponent(object):
    """Airport component skeleton
    All Airport components will inherit from this one
    """
    def __init__(self):

        self.label = None
        self.area = None
        self.area_width = None
        self.area_length = None

        self.peak_power = None
        self.nominal_power = None
        self.daily_energy = None

    def get_power_data(self):
        return {"peak":self.peak_power, "nominal":self.nominal_power, "daily_energy":self.daily_energy}

    def get_area_data(self):
        return {"length":self.area_length, "width":self.area_width, "area":self.area}

    def print_design_data(self):
        """Print component design characteristics
        """
        print(self.__class__.__name__)
        print("---------------------------------------------------------------")
        print("Area = ", "%.2f"%(self.area)," m2")
        print("Length = ", "%.2f"%(self.area_length)," m")
        print("Width = ", "%.2f"%(self.area_width)," m")
        print("")
        print("Peak power = ", "%.2f"%(self.peak_power*1.e-3)," kW")
        print("Nominal power = ", "%.2f"%(self.area_length*1.e-3)," kW")
        print("Daily consumption = ", "%.2f"%unit.kWh_J(self.daily_energy)," kWh")
        print("")

    def patch(self, s,o,a,c):
        # Airport componnent basic shape
        l = self.area_length
        w = self.area_width
        x = o[0] - s[1]*np.sin(a) + s[0]*np.cos(a)
        y = o[1] + s[1]*np.cos(a) + s[0]*np.sin(a)
        ptch = tool.rect(l, w, x, y, a, c, self.label)
        ptch_list = [ptch]
        return ptch_list


class Runways(AirportComponent):
    """Groups all the runways of the airport
    """
    def __init__(self, count, length, open_time):
        super(Runways, self).__init__()

        self.label = "Runway"
        self.count = count
        self.runway_length = length
        self.runway_width = 45.
        self.runnway_side = 75.

        self.area_length = self.runway_length + 1000.
        self.area_width = (self.runway_width + 2*self.runnway_side) * self.count
        self.area = self.area_length * self.area_width

        self.nominal_power = 20. * 2000. * self.count       # 2000 spots of 20W each
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_time - 10.)

    def patch(self, s,o,a,c):
        # Runway area
        color = c
        l = self.area_length
        w = self.area_width
        x = o[0] - s[1]*np.sin(a) + s[0]*np.cos(a)
        y = o[1] + s[1]*np.cos(a) + s[0]*np.sin(a)
        ptch = tool.rect(l, w, x, y, a, color, self.label)
        ptch_list = [ptch]

        # Runways
        color = "grey"
        l = self.runway_length
        w = self.runway_width
        side = self.runnway_side
        for n in range(self.count):
            xref = o[0] - (s[1] + side + n*(w+2*side))*np.sin(a) + s[0]*np.cos(a)
            yref = o[1] + (s[1] + side + n*(w+2*side))*np.cos(a) + s[0]*np.sin(a)
            ptch_list.append(tool.rect(l, w, xref, yref, a, color, "field"))

        return ptch_list


class RadarStation(AirportComponent):
    """Groups all the taxiways of the airport
    """
    def __init__(self, open_time):
        super(RadarStation, self).__init__()

        self.label = "Radar station"
        self.area_length = 200.
        self.area_width = 200.
        self.area = self.area_length * self.area_width

        self.nominal_power = 500.e3     # 500 kW
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * open_time


class TaxiWays(AirportComponent):
    """Groups all the taxiways of the airport
    """
    def __init__(self, runway_length, open_time):
        super(TaxiWays, self).__init__()

        self.label = "Taxiway"
        self.area_length = runway_length + 1000.
        self.area_width = 400.
        self.area = self.area_length * self.area_width

        self.nominal_power = 0.0125 * self.area  # 12.5 mW/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_time - 10.)


class AirParks(AirportComponent):
    """Stands for all the airplanes parks of the airport
    """
    def __init__(self, max_ac_flow, mean_ac_span, terminal_length, terminal_width, open_time):
        super(AirParks, self).__init__()

        self.label = "Air park"
        self.area = max_ac_flow * mean_ac_span**2
        self.area_length = terminal_length + terminal_width
        self.area_width = self.area / self.area_length

        self.nominal_power = 0.0125 * self.area  # 12.5 mW/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_time - 10.)


class AirService(AirportComponent):
    """ This component Represents all the hangars dedicated to services and maintenance
    """
    def __init__(self, max_ac_flow, mean_ac_span, open_time):
        super(AirService, self).__init__()

        self.label = "Air service"
        self.area = 0.75 * max_ac_flow * mean_ac_span**2
        self.area_length = np.sqrt(self.area)
        self.area_width = self.area / self.area_length

        self.nominal_power = 0.10 * self.area  # 100 mW/m2
        self.peak_power = 1.20 * self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_time - 10.)


class Terminals(AirportComponent):
    """Represents the passenger terminals grouped in one single area of one single level
    """
    def __init__(self, max_pax_flow_capacity, open_time):
        super(Terminals, self).__init__()

        self.departure_capacity = max_pax_flow_capacity
        self.arrival_capacity = max_pax_flow_capacity

        self.label = "Terminal"
        self.area = 4.0 * (self.departure_capacity + self.arrival_capacity)
        self.area_width = np.sqrt(0.5*self.area)     # A rectangle w*(2w)
        self.area_length = 2. * self.area_width

        self.nominal_power = 0.75 * self.area
        self.peak_power = 1.25 * self.nominal_power
        self.daily_energy = self.nominal_power * open_time


class CarParks(AirportComponent):
    """Stands for all car parks in one single level
    """
    def __init__(self, max_pax_flow):
        super(CarParks, self).__init__()

        self.space_count = 0.90 * max_pax_flow

        self.label = "Car parks"
        self.area = 1.5 * (2.5 * 5.0 * self.space_count)
        self.area_length = np.sqrt(self.area)
        self.area_width = self.area_length

        self.nominal_power = 0.005 * self.area  # 5 mW/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * (14.*3600.)


class TaxiStation(AirportComponent):
    """Taxi station
    """
    def __init__(self, max_pax_flow):

        self.label = "Taxi station"
        self.area = 0.5 * max_pax_flow
        self.area_length = np.sqrt(0.5*self.area)
        self.area_width = self.area / self.area_length

        self.nominal_power = 1. * self.area  # 1 W/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * (14.*3600.)


class BusStation(AirportComponent):
    """Bus station
    """
    def __init__(self, max_pax_flow):

        self.label = "Bus station"
        self.area = 0.5 * max_pax_flow
        self.area_length = np.sqrt(0.5*self.area)
        self.area_width = self.area / self.area_length

        self.nominal_power = 1. * self.area  # 1 W/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * (14.*3600.)


class TrainStation(AirportComponent):
    """Railway station
    """
    def __init__(self, max_pax_flow):

        self.label = "Train station"
        self.area_length = 50.
        self.area_width = 300.
        self.area = self.area_length * self.area_width

        self.nominal_power = 1. * self.area  # 1 W/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * (14.*3600.)


class Passenger(object):
    """Passenger profile
    for now, ground transport distribution profile only
    """
    def __init__(self):
        # Proportion of passengers that will prefer a taxi, others will distribute among train, bus and car
        self.taxi_ratio = 0.05

    def set_taxi_ratio(self, taxi_ratio):
        self.taxi_ratio = taxi_ratio


class Airport(object):
    """Airport object is build with airport components
    It is sized according to the characteristics of the ac_list
    """

    def __init__(self, categories, ac_list, n_runway, open_slot, app_dist, town_dist):
        self.cat = categories

        self.approach_dist = None
        self.open_slot = None
        self.open_time = None

        self.max_passenger_capacity = None
        self.max_airplane_capacity = None
        self.overall_width = None

        self.ref_mean_airplane_pax = None
        self.ref_mean_landing_time = None
        self.ref_mean_takeoff_time = None

        self.ref_peak_power = None
        self.ref_nominal_power = None
        self.ref_daily_energy = None
        self.ref_yearly_energy = None
        self.total_area = None

        self.design(ac_list, n_runway, open_slot, app_dist, town_dist)

    def design(self, ac_list, n_runway, open_slot, app_dist, town_dist):

        self.town_distance = town_dist
        self.approach_dist = app_dist
        self.open_slot = open_slot

        open_time = open_slot[1] - open_slot[0]
        self.open_time = open_time

        self.overall_width = unit.m_km(4.)

        # Build the airport
        #---------------------------------------------------------------------------------------------------------------
        # Get max runway length according to ac_list
        max_rnw_length = 0.
        for ac in ac_list:
            max_rnw_length = max(max_rnw_length, self.cat.get_data_from_pax("tofl", ac["npax"])[1])

        # Load runway component
        self.runway = Runways(n_runway, max_rnw_length, open_time)
        # Load taxiway component
        self.taxiway = TaxiWays(max_rnw_length, open_time)

        # Get mean aircraft span according to ac_list
        mean_ac_span = 0.
        for ac in ac_list:
            mean_ac_span += self.cat.get_data_from_pax("span", ac["npax"], 0.5) * ac["ratio"]

        # Compute flows from ac_list, these values will be considered as maximum
        data_dict = self.get_capacity(1., ac_list)

        max_pax_flow = data_dict["pax_flow"]
        # Load terminal component
        self.terminal = Terminals(max_pax_flow, open_time)

        max_ac_flow = data_dict["ac_flow"]
        terminal_length = self.terminal.area_length
        terminal_width = self.terminal.area_width

        # Load air parks component
        self.air_parks = AirParks(max_ac_flow, mean_ac_span, terminal_length, terminal_width, open_time)

        # Load radar service components
        self.radar_station = RadarStation(open_time)

        # Load air service components
        self.air_service = AirService(max_ac_flow, mean_ac_span, open_time)

        self.car_parks = CarParks(max_pax_flow)

        self.taxi_station = TaxiStation(max_pax_flow)

        self.bus_station = BusStation(max_pax_flow)

        self.train_station = TrainStation(max_pax_flow)

        self.passenger = Passenger()

        # Compute design characteristics
        #---------------------------------------------------------------------------------------------------------------
        self.max_passenger_capacity = data_dict["pax_flow"]
        self.max_airplane_capacity = data_dict["ac_flow"]

        self.ref_mean_airplane_pax = data_dict["ac_mean_pax"]
        self.ref_mean_landing_time = data_dict["mean_ld_ot"]
        self.ref_mean_takeoff_time = data_dict["mean_to_ot"]

        peak_power = 0.
        nominal_power = 0.
        daily_energy = 0.
        total_area = 0.
        for comp in self:
            dict_power = comp.get_power_data()
            peak_power += dict_power["peak"]
            nominal_power += dict_power["nominal"]
            daily_energy += dict_power["daily_energy"]

            dict_area = comp.get_area_data()
            total_area += dict_area["area"]

        self.ref_peak_power = peak_power
        self.ref_nominal_power = nominal_power
        self.ref_daily_energy = daily_energy
        self.ref_yearly_energy = daily_energy*365.
        self.total_area = total_area

    # Iterator to be able to loop over all airport components
    def __iter__(self):
        public = [value for value in self.__dict__.values() if issubclass(type(value),AirportComponent)]
        return iter(public)

    def get_capacity(self, capacity_ratio, ac_list):
        """Evaluate airplane, passenger flow and the mean occupation time of one runway according to the aircraft distribution
        """
        # Mean number of passenger by airplane
        mean_ac_capacity = 0.
        for ac in ac_list:
            mean_ac_capacity += ac["npax"] * ac["ratio"]

        # Insert range segment label in ac_list
        r = 0.
        for ac in ac_list:
            r += ac["ratio"]
            ac["seg"] = self.cat.seg_from_pax(ac["npax"])
        if r != 1.:
            raise Exception("Sum of aircraft distribution ratios is different from 1")
        nac = len(ac_list)

        # Prepare data for flow computations
        app_dist_separation = np.empty((nac,nac))
        to_time_separation = np.empty((nac,nac))
        probability = np.empty((nac,nac))
        buffer = np.empty((nac,nac))
        for jl,acl in enumerate(ac_list):
            catl = self.cat.get_data_from_seg("wakevortex", acl["seg"])
            for jf,acf in enumerate(ac_list):
                catf = self.cat.get_data_from_seg("wakevortex", acf["seg"])
                probability[jl,jf] = acl["ratio"]*acf["ratio"]
                app_dist_separation[jl,jf] = self.cat.get_separation_data("approach_separation", catl, catf)
                to_time_separation[jl,jf] = self.cat.get_separation_data("takeoff_separation", catl, catf)
                buffer[jl,jf] = self.cat.get_separation_data("buffer_time", catl, catf)

        # Compute the minimum time interval between 2 successive aircraft in approach
        # There are 2 cases: closing case (V_i <= V_j) and opening case (V_i > V_j)
        time_separation = np.empty((nac,nac))
        for jl,acl in enumerate(ac_list):
            vappl = self.cat.get_data_from_seg("app_speed", acl["seg"])[0]
            rotl = self.cat.r_o_t[acl["seg"]]
            for jf,acf in enumerate(ac_list):
                vappf = self.cat.get_data_from_seg("app_speed", acf["seg"])[0]
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
                "ac_mean_pax":mean_ac_capacity,
                "mean_ld_ot":mean_landing_ot,
                "mean_to_ot":mean_take_off_ot}

        return dict

    def get_flows(self, capacity_ratio, fleet, network):

        ac_list = []
        for seg,ac in fleet.items():
            ac_list.append({"ratio":network[seg]["ratio"], "npax":ac.npax})

        data_dict = self.get_capacity(capacity_ratio, ac_list)

        ac_count = {}
        for seg,ac in fleet.items():
            ac_count[seg] = data_dict["ac_flow"]*network[seg]["ratio"]

        total_fuel = 0.
        total_pax = 0.
        ac_fuel = {}
        ac_pax = {}
        for seg in fleet.keys():
            ac_fuel[seg] = 0.
            ac_pax[seg] = 0.
            for route in network[seg]["route"]:
                npax = network[seg]["load_factor"] * fleet[seg].npax    # Current number of passenger
                ac_pax[seg] += npax * (ac_count[seg] * route[0])       # pax on the route * Number of AC on this route
                dist = route[1]
                fuel,time,tow = fleet[seg].operation(npax,dist)
                ac_fuel[seg] += fuel * ac_count[seg] * route[0]        # Fuel on the route * Number of AC on this route
            total_fuel += ac_fuel[seg]
            total_pax += ac_pax[seg]

        data_dict["fleet_count"] = ac_count     # Number of aircraft of each category in the fleet
        data_dict["fleet_fuel"] = ac_fuel       # Fuel consumed by aircraft of each category in the fleet
        data_dict["fleet_pax"] = ac_pax         # Number of passengers taken by aircraft of each category in the fleet
        data_dict["total_fuel"] = total_fuel    # Total fuel delivered to the fleet
        data_dict["total_pax"] = total_pax      # Total number of passenger dropped or taken by the fleet

        return data_dict

    def print_airport_design_data(self):
        """Print airport characteristics
        """
        print("==============================================================================")
        print("Design daily passenger flow (input or output) = ", "%.0f"%self.max_passenger_capacity)
        print("Design daily aircraft movements (landing or take off) = ", "%.0f"%self.max_airplane_capacity)
        print("Total airport foot print", "%.1f"%(self.total_area*1.e-6)," km2")
        print("")
        print("Reference mean airplane capacity = ", "%.0f"%self.ref_mean_airplane_pax)
        print("Reference mean runway occupation time at landing = ", "%.1f"%self.ref_mean_landing_time," ss")
        print("Reference mean runway occupation time at take off = ", "%.1f"%self.ref_mean_takeoff_time," s")
        print("")
        print("reference peak power", "%.2f"%(self.ref_peak_power*1.e-6)," MW")
        print("reference nominal power", "%.2f"%(self.ref_nominal_power*1.e-6)," MW")
        print("reference daily consumption", "%.2f"%unit.MWh_J(self.ref_daily_energy)," MWh")
        print("reference yearly consumption", "%.2f"%unit.GWh_J(self.ref_yearly_energy)," GWh")
        print("")

    def print_component_design_data(self):
        """Print component characteristics
        """
        print("==============================================================================")
        for comp in self:
            comp.print_design_data()

    def draw(self):
        """Plot the airport
        """
        window_title = "My Airport"
        plot_title = "This airport"

        xmax = 3.
        ymax = 3.
        angle = unit.rad_deg(0.)

        origin = [0., 0.]

        fig,axes = plt.subplots(1,1)
        fig.canvas.set_window_title(window_title)
        fig.suptitle(plot_title, fontsize=14)
        axes.set_aspect('equal', 'box')
        axes.set_xbound(-xmax, xmax)
        axes.set_ybound(-ymax, ymax)
        plt.plot(np.array([-xmax,xmax,xmax,-xmax,-xmax]), np.array([-ymax,-ymax,ymax,ymax,-ymax]))      # Draw a square box of 20km x 20km

        patch_list = []

        shift = [0., 0.]
        patch_list += self.air_parks.patch(shift, origin, angle, "fuchsia")

        shift = [0., self.air_parks.area_width]
        patch_list += self.taxiway.patch(shift, origin, angle, "plum")

        shift = [0., self.air_parks.area_width + self.taxiway.area_width + self.runway.area_width + 1000.]
        patch_list += self.radar_station.patch(shift, origin, angle, "red")

        shift = [0., self.air_parks.area_width + self.taxiway.area_width]
        patch_list += self.runway.patch(shift, origin, angle, "palegreen")

        shift = [-0.5*(self.air_parks.area_length + self.air_service.area_length),
                 self.air_parks.area_width - self.air_service.area_width]
        patch_list += self.air_service.patch(shift, origin, angle, "khaki")

        shift = [0., -self.terminal.area_width]
        patch_list += self.terminal.patch(shift, origin, angle, "cornflowerblue")

        shift = [0.5*(self.air_parks.area_length + self.car_parks.area_length),
                 self.air_parks.area_width - self.car_parks.area_width]
        patch_list += self.car_parks.patch(shift, origin, angle, "lightgrey")

        shift = [0.5*(self.terminal.area_length - self.taxi_station.area_length),
                 -self.terminal.area_width - self.taxi_station.area_width]
        patch_list += self.taxi_station.patch(shift, origin, angle, "thistle")

        shift = [0.5*(-self.terminal.area_length + self.bus_station.area_length),
                 -self.terminal.area_width - self.bus_station.area_width]
        patch_list += self.bus_station.patch(shift, origin, angle, "yellowgreen")

        shift = [0.,
                 -self.terminal.area_width - self.train_station.area_width]
        patch_list += self.train_station.patch(shift, origin, angle, "orange")

        for ptch in patch_list:
            axes.add_patch(ptch)

        plt.show()
