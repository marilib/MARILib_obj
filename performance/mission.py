#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
from scipy.optimize import fsolve

import unit
import earth

import aircraft.flight as flight

class Breguet_range(object):
    """Define common features for all mission types.

    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.range = None   # Mission distance
        self.altp = None    # Mean cruise altitude
        self.mach = None    # Cruise mach number
        self.tow = None     # Take Off Weight
        self.time_block = None      # Mission block duration
        self.fuel_block = None      # Mission block fuel consumption
        self.fuel_reserve = None    # Mission reserve fuel
        self.fuel_total = None      # Mission total fuel

        self.holding_time = unit.s_min(30)  # Holding duration
        self.reserve_fuel_ratio = self.__reserve_fuel_ratio__() # Ratio of mission fuel to account into reserve
        self.diversion_range = self.__diversion_range__()       # Diversion leg

    def __reserve_fuel_ratio__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range>unit.m_NM(6500.)):
            reserve_fuel_ratio = 0.03
        else:
            reserve_fuel_ratio = 0.05
        return reserve_fuel_ratio

    def __diversion_range__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range>unit.m_NM(200.)):
            diversion_range = unit.m_NM(200.)
        else:
            diversion_range = design_range
        return diversion_range

    def mission(self,range,tow,altp,mach,disa):
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
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        tas = mach*earth.sound_speed(tamb)


        fn,ff,sfc = self.aircraft.power_system.thrust(pamb,tamb,mach,"MCR")

        mass = 0.95 * tow
        cz_cruise = flight.lift_from_speed(self.aircraft,pamb,tamb,mach,mass)
        cx_cruise,lod_cruise = self.aircraft.aerodynamics.drag()






        if (propulsion.architecture=="TF"):
            fuel_mission = tow*(1-numpy.exp(-(sfc*g*range)/(tas*lod_cruise)))
        elif (propulsion.architecture=="TP"):
            fuel_mission = tow*(1-numpy.exp(-(sfc*g*range)/(tas*lod_cruise)))
        elif (propulsion.architecture=="PTE1"):
            if (aircraft.pte1_battery.strategy>0):
                fuel_mission = tow*(1-numpy.exp(-(sfc*g*range)/(tas*lod_cruise))) \
                                - (sfc/sec)*aircraft.pte1_battery.energy_cruise
            else:
                fuel_mission = tow*(1-numpy.exp(-(sfc*g*range)/(tas*lod_cruise)))
        else:
            raise Exception("propulsion.architecture index is out of range")

        time_mission = 1.09*(range/tas)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)

        # Arrival ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_landing = 1e-4*(0.5+2.3/engine.bpr)*mass
        time_landing = 180.

        fuel_taxi_in = (26. + 1.8e-4*propulsion.reference_thrust)*propulsion.n_engine
        time_taxi_in = 420.

        # Block fuel and time
        #-----------------------------------------------------------------------------------------------------------
        block_fuel = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        #-----------------------------------------------------------------------------------------------------------
        fuel_diversion = mass*(1.-numpy.exp(-(sfc*g*regul.diversion_range(design_range))/(tas*lod_cruise)))

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.50 * mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp_holding,disa)
        lod_max, cz_lod_max = airplane_aero.lod_max(aircraft, pamb, tamb, mach_holding)
        fuel_holding = sfc*(mass*g/lod_max)*regul.holding_time()

        # Total
        #-----------------------------------------------------------------------------------------------------------
        fuel_total = fuel_mission*(1.+self.reserve_fuel_ratio) + fuel_diversion + fuel_holding

        #-----------------------------------------------------------------------------------------------------------
        return






