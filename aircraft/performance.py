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
import aircraft.requirement as requirement


class Performance(object):
    """
    Master class for all aircraft performances
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.mission = Mission(aircraft)
        self.take_off = Take_off(aircraft)
        self.landing = None
        self.mcr_ceiling = None
        self.mcl_ceiling = None
        self.oei_ceiling = None
        self.time_to_climb = None

    def analysis(self):
        hld_conf = self.aircraft.aerodynamics.hld_conf_to
        self.take_off.simulate(hld_conf)




    def get_speed(self,pamb,speed_mode,mach):
        """
        retrieves CAS or Mach from mach depending on speed_mode
        """
        speed = {1 : earth.vcas_from_mach(pamb,mach),   # CAS required
                 2 : mach                               # mach required
                 }.get(speed_mode, "Erreur: select speed_mode equal to 1 or 2")
        return speed

    def get_mach(self,pamb,speed_mode,speed):
        """
        Retrieves Mach from CAS or mach depending on speed_mode
        """
        mach = {1 : earth.mach_from_vcas(pamb,speed),   # Input is CAS
                2 : speed                               # Input is mach
                }.get(speed_mode, "Erreur: select speed_mode equal to 1 or 2")
        return mach

    def speed_from_lift(self,pamb,tamb,cz,mass):
        """
        Retrieves mach from cz using simplified lift equation
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()
        mach = np.sqrt((mass*g)/(0.5*gam*pamb*self.aircraft.airframe.wing.area*cz))
        return mach

    def lift_from_speed(self,pamb,tamb,mach,mass):
        """
        Retrieves cz from mach using simplified lift equation
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()
        cz = (2.*mass*g)/(gam*pamb*mach**2*self.aircraft.airframe.wing.area)
        return cz

    def level_flight(self,pamb,tamb,mach,mass):
        """
        Level flight equilibrium
        """
        g = earth.gravity()
        r,gam,Cp,Cv = earth.gas_data()

        cz = (2.*mass*g)/(gam*pamb*mach**2*self.aircraft.airframe.wing.area)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        fn = (gam/2.)*pamb*mach**2*self.aircraft.airframe.wing.area*cx
        sfc,throttle = self.aircraft.power_system.sc(pamb,tamb,mach,"MCR",fn)
        if (throttle>1.): print("level_flight, throttle is higher than 1, throttle = ",throttle)

        return cz,cx,lod,fn,sfc,throttle

    def air_path(self,nei,altp,disa,speed_mode,speed,mass,rating):
        """
        Retrieves air path in various conditions
        """
        g = earth.gravity()
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        mach = self.get_mach(pamb,speed_mode,speed)

        fn,ff,sfc = self.aircraft.power_system.thrust(pamb,tamb,mach,rating)
        cz = self.lift_from_speed(pamb,tamb,mach,mass)
        cx,lod = self.aircraft.aerodynamics.drag(pamb,tamb,mach,cz)

        if(nei>0):
            dcx = self.aircraft.power_system.oei_drag(pamb,mach)
            cx = cx + dcx*nei
            lod = cz/cx

        acc_factor = earth.climb_mode(speed_mode,dtodz,tstd,disa,mach)
        slope = ( fn/(mass*g) - 1/lod ) / acc_factor
        vz = mach*slope*earth.sound_speed(tamb)

        return slope,vz


class Take_off(requirement.Take_off_req):
    """
    Definition of all mission types
    """
    def __init__(self, aircraft):
        super(Take_off, self).__init__(aircraft.arrangement, aircraft.requirement)
        self.aircraft = aircraft

        self.hld_conf = None
        self.tofl_eff = None
        self.kvs1g_eff = None
        self.v2 = None
        self.s2_path = None
        self.limit = None

    def simulate(self,hld_conf):
        """
        Take off field length and climb path with eventual kVs1g increase to recover min regulatory slope
        """
        kvs1g = self.kvs1g
        altp = self.altp
        disa = self.disa
        mass = self.kmtow*self.aircraft.weight_cg.mtow

        tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf)

        if(self.s2_min_path<s2_path):
            limitation = "fl"   # field length
        else:
            dkvs1g = 0.005
            kvs1g_ = np.array([0.,0.])
            kvs1g_[0] = self.kvs1g
            kvs1g_[1] = kvs1g_[0] + dkvs1g

            s2_path_ = np.array([0.,0.])
            s2_path_[0] = s2_path
            tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf)

            while(s2_path_[0]<s2_path_[1] and s2_path_[1]<self.s2_min_path):
                kvs1g_[0] = kvs1g_[1]
                kvs1g_[1] = kvs1g_[1] + dkvs1g
                tofl,s2_path_[1],cas,mach = self.take_off(kvs1g_[1],altp,disa,mass,hld_conf)

            if(self.s2_min_path<s2_path_[1]):
                kvs1g = kvs1g_[0] + ((kvs1g_[1]-kvs1g_[0])/(s2_path_[1]-s2_path_[0]))*(self.s2_min_path-s2_path_[0])
                tofl,s2_path,cas,mach = self.take_off(kvs1g,altp,disa,mass,hld_conf)
                s2_path = self.s2_min_path
                limitation = "s2"   # second segment
            else:
                tofl = np.nan
                kvs1g = np.nan
                s2_path = 0.
                limitation = None

            self.tofl_eff = tofl
            self.kvs1g_eff = kvs1g
            self.s2_path = s2_path
            self.v2 = cas
            self.limit = limitation

        return

    def take_off(self,kvs1g,altp,disa,mass,hld_conf):
        """
        Take off field length and climb path at 35 ft depending on stall margin (kVs1g)
        """
        czmax,cz_0 = self.aircraft.airframe.wing.high_lift(hld_conf)

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
        rho,sig = earth.air_density(pamb,tamb)

        cz_to = czmax / kvs1g**2
        mach = self.aircraft.performance.speed_from_lift(pamb,tamb,cz_to,mass)

        nei = 0    # For Magic Line factor computation
        throttle = 1.
        fn,ff,sfc = self.aircraft.power_system.thrust(pamb,tamb,mach,"MTO",throttle,nei)

        ml_factor = mass**2 / (cz_to*fn*self.aircraft.airframe.wing.area*sig**0.8 )  # Magic Line factor
        tofl = 15.5*ml_factor + 100.    # Magic line

        nei = 1         # For 2nd segment computation
        speed_mode = 1  # Constant CAS
        speed = self.aircraft.performance.get_speed(pamb,speed_mode,mach)

        s2_path,vz = self.aircraft.performance.air_path(nei,altp,disa,speed_mode,speed,mass,"MTO")

        return tofl,s2_path,speed,mach












class Mission(object):
    """
    Definition of all mission types
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

        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach

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


class Mission_fuel_from_range_and_tow(object):
    """
    Define common features for all mission types.
    """
    def __init__(self, aircraft):
        self.aircraft = aircraft

        self.disa = None    # Mean cruise temperature shift
        self.altp = None    # Mean cruise altitude
        self.mach = None    # Cruise mach number
        self.range = None   # Mission distance
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
        self.eval_breguet(range,tow,altp,mach,disa)
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
        engine_bpr = self.aircraft.airframe.nacelle.engine_bpr

        # Departure ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_taxi_out = (34. + 2.3e-4*reference_thrust)*n_engine
        time_taxi_out = 540.

        fuel_take_off = 1e-4*(2.8+2.3/engine_bpr)*tow
        time_take_off = 220.*tow/(reference_thrust*n_engine)

        # Mission leg
        #-----------------------------------------------------------------------------------------------------------
        fuel_mission,time_mission = self.aircraft.power_system.breguet_range(range,tow,altp,mach,disa)

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
        fuel_diversion,t = self.aircraft.power_system.breguet_range(self.diversion_range,tow,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.50 * mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp_holding,disa)
        cz,cx,lod,fn,sfc,throttle = self.aircraft.performance.level_flight(pamb,tamb,mach_holding,mass)
        fuel_holding = sfc*(mass*g/lod)*self.holding_time

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission*self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.block_fuel + self.fuel_reserve

        #-----------------------------------------------------------------------------------------------------------
        return


class Mission_range_from_payload_and_tow(Mission_fuel_from_range_and_tow):
    """
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and total_fuel from payload and tow
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

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_tow(Mission_fuel_from_range_and_tow):
    """
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and payload from total_fuel and tow
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

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_payload(Mission_fuel_from_range_and_tow):
    """
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and tow from total_fuel and payload
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

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,self.tow,altp,mach,disa)


class Mission_fuel_from_range_and_payload(Mission_fuel_from_range_and_tow):
    """
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes total_fuel and tow from range and payload
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
            self.eval_breguet(range,tow,altp,mach,disa)
            return tow-owe-payload-self.fuel_total

        tow_ini = [self.aircraft.weight_cg.mtow]
        output_dict = fsolve(fct, x0=tow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.tow = output_dict[0][0]
        self.eval_breguet(self.range,self.tow,altp,mach,disa)

