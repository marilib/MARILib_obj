#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from aircraft.tool import unit
import earth

import numpy as np
from scipy.optimize import fsolve

import aircraft.flight as flight


class Mission_fuel_from_range_and_tow(object):
    """Define common features for all mission types.

    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.range = None   # Mission distance
        self.disa = None    # Mean cruise temperature shift
        self.altp = None    # Mean cruise altitude
        self.mach = None    # Cruise mach number
        self.tow = None     # Take Off Weight
        self.payload = None         # Mission payload
        self.time_block = None      # Mission block duration
        self.fuel_block = None      # Mission block fuel consumption
        self.fuel_reserve = None    # Mission reserve fuel
        self.fuel_total = None      # Mission total fuel

        self.holding_time = unit.s_min(30)  # Holding duration
        self.reserve_fuel_ratio = self.__reserve_fuel_ratio__() # Ratio of mission fuel to account into reserve
        self.diversion_range = self.__diversion_range__()       # Diversion leg

    def simulate(self,range,tow,owe,altp,mach,disa):
        self.range = range  # Mission distance
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.eval_breguet(self,range,tow,altp,mach,disa)
        self.eval_payload(owe)

    def __reserve_fuel_ratio__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(6500.)):
            reserve_fuel_ratio = 0.03
        else:
            reserve_fuel_ratio = 0.05
        return reserve_fuel_ratio

    def __diversion_range__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(200.)):
            diversion_range = unit.m_NM(200.)
        else:
            diversion_range = design_range
        return diversion_range

    def eval_payload(self,owe):
        """
        Computing resulting payload
        """
        self.payload = self.tow - self.fuel_total - owe

    def eval_breguet(self,range,tow,altp,mach,disa):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        g = earth.gravity()
        fhv = self.aircraft.power_system.fuel_heat
        n_engine = self.aircraft.airframe.nacelle.n_engine
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        engine_bpr = self.aircraft.airframe.nacelle.bpr

        # Departure ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_taxi_out = (34. + 2.3e-4*reference_thrust)*n_engine
        time_taxi_out = 540.

        fuel_take_off = 1e-4*(2.8+2.3/engine_bpr)*tow
        time_take_off = 220.*tow/(reference_thrust*n_engine)

        # Mission leg
        #-----------------------------------------------------------------------------------------------------------
        fuel_mission,time_mission = self.aircraft.power_system.breguet_range(self.aircraft,range,tow,altp,mach,disa)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)

        # Arrival ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_landing = 1e-4*(0.5+2.3/engine_bpr)*mass
        time_landing = 180.

        fuel_taxi_in = (26. + 1.8e-4*reference_thrust)*n_engine
        time_taxi_in = 420.

        # Block fuel and time
        #-----------------------------------------------------------------------------------------------------------
        self.block_fuel = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        #-----------------------------------------------------------------------------------------------------------
        fuel_diversion,t = self.aircraft.power_system.breguet_range(self.aircraft,self.diversion_range,tow,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.50 * mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp_holding,disa)
        cz,cx,lod,fn,sfc,throttle = flight.level_flight(self.aircraft,pamb,tamb,mach_holding,mass)
        fuel_holding = sfc*(mass*g/lod)*self.holding_time()

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission*self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.block_fuel + self.fuel_reserve

        #-----------------------------------------------------------------------------------------------------------
        return


class Mission_range_from_payload_and_tow(Mission_fuel_from_range_and_tow):
    """Define common features for all mission types.

    """
    def __init__(self, aircraft):
        super(Mission_range_from_payload_and_tow, self).__init__(aircraft)

    def simulate(self,payload,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return payload - (tow-owe-self.fuel_total)

        range_ini = self.aircraft.requirement.design_range
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self,self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_tow(Mission_fuel_from_range_and_tow):
    """Define common features for all mission types.

    """
    def __init__(self, aircraft):
        super(Mission_range_from_fuel_and_tow, self).__init__(aircraft)

    def simulate(self,fuel_total,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = tow-owe-fuel_total  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return self.payload - (tow-owe-self.fuel_total)

        range_ini = self.aircraft.requirement.design_range
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self,self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_payload(Mission_fuel_from_range_and_tow):
    """Define common features for all mission types.

    """
    def __init__(self, aircraft):
        super(Mission_range_from_fuel_and_payload, self).__init__(aircraft)

    def simulate(self,fuel_total,payload,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = owe+payload+fuel_total     # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,self.tow,altp,mach,disa)
            return self.tow-owe-self.payload-self.fuel_total

        range_ini = self.aircraft.requirement.design_range
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self,self.range,self.tow,altp,mach,disa)


class Mission_fuel_from_range_and_payload(Mission_fuel_from_range_and_tow):
    """Define common features for all mission types.

    """
    def __init__(self, aircraft):
        super(Mission_fuel_from_range_and_payload, self).__init__(aircraft)

    def simulate(self,range,payload,owe,altp,mach,disa):
        self.range = range     # Take Off Weight
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.payload = payload  # Mission payload

        def fct(tow):
            self.eval_breguet(range,self.tow,altp,mach,disa)
            return tow-owe-self.payload-self.fuel_total

        range_ini = self.aircraft.requirement.design_range
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.tow = output_dict[0][0]
        self.eval_breguet(self,self.range,self.tow,altp,mach,disa)


class Mission(object):
    """Define common features for all mission types.

    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.max_payload = Mission_range_from_payload_and_tow(aircraft)
        self.nominal = Mission_fuel_from_range_and_tow(aircraft)
        self.max_fuel = Mission_range_from_fuel_and_tow(aircraft)
        self.zero_payload = Mission_range_from_fuel_and_payload(aircraft)
        self.cost = Mission_fuel_from_range_and_payload(aircraft)

    def payload_range(self):
        payload_max = self.aircraft.airframe.cabin.maximum_payload
        mtow = self.aircraft.weight_cg.mtow
        owe = self.aircraft.weight_cg.owe

        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        disa = 0.

        self.max_payload.simulate(payload_max,mtow,owe,altp,mach,disa)

        range = self.aircraft.requirement.design_range
        self.nominal.simulate(range,mtow,owe,altp,mach,disa)

        fuel_max = self.aircraft.weight_cg.mfw
        self.max_fuel.simulate(fuel_max,mtow,owe,altp,mach,disa)

        payload = 0.
        self.zero_payload.simulate(fuel_max,payload,owe,altp,mach,disa)

        range = self.aircraft.requirement.cost_range
        payload = self.aircraft.airframe.cabin.nominal_payload
        self.cost.simulate(range,payload,owe,altp,mach,disa)

        return self.aircraft
