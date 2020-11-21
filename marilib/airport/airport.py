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

from marilib.airport.aircraft import AirplaneCategories, Aircraft, Fleet



class AirportComponent(object):
    """Airport component skeleton
    All Airport components will inherit from this one
    """
    def __init__(self):

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


class Runways(AirportComponent):
    """Groups all the runways of the airport
    """
    def __init__(self, count, length, open_slot):
        super(Runways, self).__init__()

        self.count = count
        self.runway_length = length
        self.runway_width = 45.

        self.area_length = self.runway_length + 1000.
        self.area_width = (self.runway_width + 200.) * self.count
        self.area = self.area_length * self.area_width

        self.nominal_power = 20. * 2000. * self.count       # 2000 spots of 20W each
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_slot[1]-open_slot[0]-10.)


class TaxiWays(AirportComponent):
    """Groups all the taxiways of the airport
    """
    def __init__(self, runway_length):
        super(TaxiWays, self).__init__()

        self.area_length = runway_length
        self.area_width = 400.
        self.area = self.area_length * self.area_width

        self.nominal_power = 0.0125 * self.area  # 12.5 mW/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_slot[1]-open_slot[0]-10.)


class AirParks(AirportComponent):
    """Stands for all the airplanes parks of the airport
    """
    def __init__(self, max_ac_flow, mean_ac_span, terminal_length, terminal_width):
        super(AirParks, self).__init__()

        self.area = max_ac_flow * mean_ac_span**2
        self.area_length = terminal_width + terminal_width
        self.area_width = self.area / self.area_length

        self.nominal_power = 0.0125 * self.area  # 12.5 mW/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_slot[1]-open_slot[0]-10.)


class AirService(AirportComponent):
    """ This component Represents all the hangars dedicated to services and maintenance
    """
    def __init__(self, max_ac_flow, mean_ac_span):
        super(AirService, self).__init__()

        self.area = 0.20 * max_ac_flow * mean_ac_span**2
        self.area_length = np.sqrt(self.area)
        self.area_width = self.area / self.area_length

        self.nominal_power = 0.10 * self.area  # 100 mW/m2
        self.peak_power = 1.20 * self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_slot[1]-open_slot[0]-10.)


class Terminals(AirportComponent):
    """Represents the passenger terminals grouped in one single area of one single level
    """
    def __init__(self, max_pax_flow_capacity):
        super(Terminals, self).__init__()

        self.departure_capacity = max_pax_flow_capacity
        self.arrival_capacity = max_pax_flow_capacity

        self.area = 4.0 * (self.departure_capacity + self.arrival_capacity)
        self.area_width = np.sqrt(self.area/2.)     # A rectangle w*(2w)
        self.area_length = 2. * self.area_width

        self.nominal_power = 0.75 * self.area
        self.peak_power = 1.25 * self.nominal_power
        self.daily_energy = self.nominal_power * max(0., open_slot[1]-open_slot[0])


class CarParks(AirportComponent):
    """Stands for all car parks in one single level
    """
    def __init__(self, max_pax_flow):
        super(CarParks, self).__init__()

        self.space_count = 0.15 * max_pax_flow

        self.area = 2.5 * 5.0 * self.space_count
        self.area_length = np.sqrt(self.area)
        self.area_width = self.area_length

        self.nominal_power = 0.005 * self.area  # 5 mW/m2
        self.peak_power = self.nominal_power
        self.daily_energy = self.nominal_power * (14.*3600.)


class Parameter(object):
    """Container for all study variables
    """
    def __init__(self, car_ratio, bus_ratio):
        self.car_ratio = car_ratio
        self.bus_ratio = bus_ratio
        self.rail_ratio = (1.-self.car_ratio-self.bus_ratio)


class Airport(object):
    """Airport object is build with airport components
    It is sized according to the characteristics of the ac_list
    """

    def __init__(self, categories, ac_list, n_runway, open_slot, app_dist):
        self.cat = categories

        self.approach_dist = app_dist
        self.open_slot = open_slot

        # Build the airport
        #---------------------------------------------------------------------------------------------------------------
        # Get max runway length according to ac_list
        max_rnw_length = 0.
        for ac in ac_list:
            max_rnw_length = max(max_rnw_length, self.cat.get_data_from_pax("tofl", ac["npax"])[1])

        # Load runway component
        self.runway = Runways(n_runway, max_rnw_length, open_slot)
        # Load taxiway component
        self.taxiway = TaxiWays(max_rnw_length)

        # Get mean aircraft span according to ac_list
        mean_ac_span = 0.
        for ac in ac_list:
            mean_ac_span += self.cat.get_data_from_pax("span", ac["npax"], 0.5) * ac["ratio"]

        # Compute flows from ac_list, these values will be considered as maximum
        data_dict = self.get_capacity(1., ac_list)

        max_pax_flow = data_dict["pax_flow"]
        # Load terminal component
        self.terminal = Terminals(max_pax_flow)

        max_ac_flow = data_dict["ac_flow"]
        terminal_length = self.terminal.area_length
        terminal_width = self.terminal.area_width

        # Load air parks component
        self.air_parks = AirParks(max_ac_flow, mean_ac_span, terminal_length, terminal_width)

        # Load air service components
        self.air_service = AirService(max_ac_flow, mean_ac_span)

        self.car_parks = CarParks(max_pax_flow)

        self.bus_station = None

        self.railway_station = None

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

    def __iter__(self):
        public = [value for value in self.__dict__.values() if issubclass(type(value),AirportComponent)]
        return iter(public)

    def get_capacity(self, capacity_ratio, ac_list):
        """Evaluate the mean occupation time of one runway according to the aircraft distribution
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
        for j,ac in enumerate(fleet):
            ac_list.append({"ratio":network[j]["ratio"], "npax":ac.npax})

        data_dict = self.get_capacity(capacity_ratio, ac_list)

        ac_count = []
        for j,ac in enumerate(fleet):
            ac_count.append(data_dict["ac_flow"]*network[j]["ratio"])

        total_fuel = 0.
        total_pax = 0.
        ac_fuel = []
        ac_pax = []
        for j,ac in enumerate(fleet):
            ac_fuel.append(0.)
            ac_pax.append(0.)
            for route in network[j]["route"]:
                npax = network[j]["load_factor"] * fleet[j].npax    # Current number of passenger
                ac_pax[-1] += npax * ac_count[j] * route[0]         # pax on the route * Number of AC on this route
                dist = route[1]
                fuel,time,tow = fleet[j].operation(npax,dist)
                ac_fuel[-1] += fuel * ac_count[j] * route[0]        # Fuel on the route * Number of AC on this route
            total_fuel += ac_fuel[-1]
            total_pax += ac_pax[-1]

        data_dict["ac_count"] = ac_count
        data_dict["ac_fuel"] = ac_fuel
        data_dict["ac_pax"] = ac_pax
        data_dict["total_fuel"] = total_fuel
        data_dict["total_pax"] = total_pax

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



cat = AirplaneCategories()

# Only proportion and design capacity is required to design the airport
ac_list = [{"ratio":0.30, "npax":70. },
           {"ratio":0.50, "npax":150.},
           {"ratio":0.15, "npax":300.},
           {"ratio":0.05, "npax":400.}]

runway_count = 3
app_dist = unit.m_NM(7.)
open_slot = [unit.s_h(6.), unit.s_h(23.)]

ap = Airport(cat, ac_list, runway_count, open_slot, app_dist)

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

capacity_ratio = 0.75

data_dict = ap.get_flows(capacity_ratio, fleet, network)

print("==============================================================================")
for j in range(len(fleet)):
    print("Daily movements (landing or take off) for airplane n°", 1+j, " = ", "%.0f"%data_dict["ac_count"][j])
    print("Daily passenger transported by airplane n°", 1+j, " = ", "%.0f"%(data_dict["ac_pax"][j]))
    print("Daily fuel delivered to airplane n°", 1+j, " = ", "%.0f"%(data_dict["ac_fuel"][j]/1000.), " t")
    print("")
print("Total passenger transported = ""%.0f"%(data_dict["total_pax"]))
print("Total fuel delivered = ""%.0f"%(data_dict["total_fuel"]/1000.), " t")
print("")




