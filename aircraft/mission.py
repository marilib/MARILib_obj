#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from context import earth, unit

import numpy as np
from scipy.optimize import fsolve

from aircraft.performance import Flight

class Basic_mission(Flight):
    """Definition of all mission types for fuel powered airplanes
    """
    def __init__(self, aircraft):
        super(Basic_mission, self).__init__(aircraft)
        self.aircraft = aircraft

        self.max_payload = None
        self.nominal = None
        self.max_fuel = None
        self.zero_payload = None
        self.cost = None

        self.disa = None
        self.altp = None
        self.mach = None
        self.mass = None

    def eval_cruise_point(self):
        raise NotImplementedError

    def payload_range(self):
        payload_max = self.aircraft.airframe.cabin.maximum_payload
        mtow = self.aircraft.weight_cg.mtow
        owe = self.aircraft.weight_cg.owe

        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach

        self.max_payload.eval(payload_max,mtow,owe,altp,mach,disa)
        #self.max_payload2.eval(owe,altp,mach,disa, payload=payload_max,tow=mtow)

        range = self.aircraft.requirement.design_range
        self.nominal.eval(range,mtow,owe,altp,mach,disa)

        fuel_max = self.aircraft.weight_cg.mfw
        self.max_fuel.eval(fuel_max,mtow,owe,altp,mach,disa)

        payload = 0.
        self.zero_payload.eval(fuel_max,payload,owe,altp,mach,disa)

        range = self.aircraft.requirement.cost_range
        payload = self.aircraft.airframe.cabin.nominal_payload
        self.cost.eval(range,payload,owe,altp,mach,disa)


class E_mission(Basic_mission):
    """Definition of all mission types for battery powered airplanes
    """
    def __init__(self, aircraft):
        super(E_mission, self).__init__(aircraft)
        self.aircraft = aircraft

        self.max_payload = E_mission_range_from_payload_and_tow(aircraft)
        self.nominal = E_mission_batt_from_range_and_tow(aircraft)
        self.max_fuel = E_mission_range_from_batt_and_tow(aircraft)
        self.zero_payload = E_mission_range_from_batt_and_payload(aircraft)
        self.cost = E_mission_batt_from_range_and_payload(aircraft)

        self.crz_esar = None
        self.crz_cz = None
        self.crz_lod = None
        self.crz_thrust = None
        self.crz_throttle = None
        self.crz_sec = None

        self.max_esar_altp = None
        self.max_esar = None
        self.max_esar_cz = None
        self.max_esar_lod = None
        self.max_esar_thrust = None
        self.max_esar_throttle = None
        self.max_esar_sec = None

    def eval_cruise_point(self):
        """Evaluate cruise point characteristics
        """
        self.disa = self.aircraft.requirement.cruise_disa
        self.altp = self.aircraft.requirement.cruise_altp
        self.mach = self.aircraft.requirement.cruise_mach
        self.mass = self.aircraft.weight_cg.mtow

        pamb,tamb,tstd,dtodz = earth.atmosphere(self.altp, self.disa)

        lf_dict = self.level_flight(pamb,tamb,self.mach,self.mass)
        sm_dict = self.eval_max_sar(self.mass,self.mach,self.disa)

        self.crz_esar = lf_dict["sar"]
        self.crz_cz = lf_dict["cz"]
        self.crz_lod = lf_dict["lod"]
        self.crz_thrust = lf_dict["fn"]
        self.crz_throttle = lf_dict["thtl"]
        self.crz_sec = lf_dict["sec"]

        self.max_esar_altp = sm_dict["altp"]
        self.max_esar = sm_dict["sar"]
        self.max_esar_cz = sm_dict["cz"]
        self.max_esar_lod = sm_dict["lod"]
        self.max_esar_thrust = sm_dict["fn"]
        self.max_esar_throttle = sm_dict["thtl"]
        self.max_esar_sec = sm_dict["sec"]

    def mass_mission_adaptation(self):
        """Solves coupling between MTOW and OWE
        """
        range = self.aircraft.requirement.design_range
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        disa = self.aircraft.requirement.cruise_disa

        payload = self.aircraft.airframe.cabin.nominal_payload

        def fct(mtow):
            self.aircraft.weight_cg.mtow = mtow[0]
            self.aircraft.weight_cg.mass_pre_design()
            owe = self.aircraft.weight_cg.owe
            self.nominal.eval(range,mtow,owe,altp,mach,disa)
            battery_mass = self.nominal.battery_mass
            return mtow - (owe + payload + battery_mass)

        mtow_ini = [self.aircraft.weight_cg.mtow]
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.aircraft.weight_cg.mtow = output_dict[0][0]
        self.aircraft.weight_cg.mass_pre_design()
        self.aircraft.performance.mission.payload_range()


class E_mission_batt_from_range_and_tow(Flight):
    """Define common features for all mission types.
    """
    def __init__(self, aircraft):
        super(E_mission_batt_from_range_and_tow, self).__init__(aircraft)
        self.aircraft = aircraft

        self.disa = None    # Mean cruise temperature shift
        self.altp = None    # Mean cruise altitude
        self.mach = None    # Cruise mach number
        self.range = None   # Mission distance
        self.tow = None     # Take Off Weight
        self.payload = None         # Mission payload
        self.time_block = None      # Mission block duration
        self.enrg_block = None      # Mission block energy consumption
        self.enrg_reserve = None    # Mission reserve energy
        self.enrg_total = None      # Mission total energy
        self.battery_mass = None    # Mission battery mass

        self.holding_time = unit.s_min(30)  # Holding duration
        self.reserve_enrg_ratio = self.__reserve_enrg_ratio__() # Ratio of mission fuel to account into reserve
        self.diversion_range = self.__diversion_range__()       # Diversion leg

    def eval(self,range,tow,owe,altp,mach,disa):
        """Evaluate mission and store results in object attributes
        """
        self.range = range  # Mission distance
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.eval_breguet(range,tow,altp,mach,disa)
        self.eval_payload(owe)

    def __reserve_enrg_ratio__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(6500.)):
            reserve_enrg_ratio = 0.03
        else:
            reserve_enrg_ratio = 0.05
        return reserve_enrg_ratio

    def __diversion_range__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(200.)):
            diversion_range = unit.m_NM(200.)
        else:
            diversion_range = design_range
        return diversion_range

    def eval_payload(self,owe):
        """Computing resulting payload
        """
        self.payload = self.tow - self.battery_mass - owe

    def breguet_range(self,range,tow,altp,mach,disa):
        """Breguet range equation is dependant from power source : fuel or battery
        """
        g = earth.gravity()

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        tas = mach * earth.sound_speed(tamb)

        dict = self.level_flight(pamb,tamb,mach,tow)
        enrg = tow*g*range*dict["sec"] / (tas*dict["lod"])
        time = 1.09*(range/tas)

        return enrg,time

    def eval_breguet(self,range,tow,altp,mach,disa):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        g = earth.gravity()
        n_engine = self.aircraft.airframe.nacelle.n_engine
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust

        # Departure ground phases
        #-----------------------------------------------------------------------------------------------------------
        enrg_taxi_out = (0.25*43.1e6)*(34. + 2.3e-4*reference_thrust)*n_engine
        time_taxi_out = 540.

        enrg_take_off = (0.25*43.1e6)*3.e-4*tow
        time_take_off = 220.*tow/(reference_thrust*n_engine)

        # Mission leg
        #-----------------------------------------------------------------------------------------------------------
        enrg_mission,time_mission = self.breguet_range(range,tow,altp,mach,disa)

        # Arrival ground phases
        #-----------------------------------------------------------------------------------------------------------
        enrg_landing = (0.25*43.1e6)*0.75e-4*tow
        time_landing = 180.

        enrg_taxi_in = (0.25*43.1e6)*(26. + 1.8e-4*reference_thrust)*n_engine
        time_taxi_in = 420.

        # Block fuel and time
        #-----------------------------------------------------------------------------------------------------------
        self.enrg_block = enrg_taxi_out + enrg_take_off + enrg_mission + enrg_landing + enrg_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        #-----------------------------------------------------------------------------------------------------------
        enrg_diversion,t = self.breguet_range(self.diversion_range,tow,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.65 * mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp_holding, disa)
        dict = self.level_flight(pamb,tamb,mach_holding,tow)
        enrg_holding = dict["sec"]*(tow*g/dict["lod"])*self.holding_time

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.enrg_reserve = enrg_mission*self.reserve_enrg_ratio + enrg_diversion + enrg_holding
        self.enrg_total = self.enrg_block + self.enrg_reserve
        self.battery_mass = self.enrg_total / self.aircraft.airframe.system.battery_energy_density

        #-----------------------------------------------------------------------------------------------------------
        return


class E_mission_range_from_payload_and_tow(E_mission_batt_from_range_and_tow):
    """Specific mission evaluation from payload and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and total_fuel from payload and tow
    """
    def __init__(self, aircraft):
        super(E_mission_range_from_payload_and_tow, self).__init__(aircraft)

    def eval(self,payload,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.battery_mass)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class E_mission_range_from_batt_and_tow(E_mission_batt_from_range_and_tow):
    """Specific mission evaluation from total fuel and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and payload from total_fuel and tow
    """
    def __init__(self, aircraft):
        super(E_mission_range_from_batt_and_tow, self).__init__(aircraft)

    def eval(self,battery_mass,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = tow-owe-battery_mass  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.battery_mass)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class E_mission_range_from_batt_and_payload(E_mission_batt_from_range_and_tow):
    """Specific mission evaluation from payload and total fuel
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and tow from total_fuel and payload
    """
    def __init__(self, aircraft):
        super(E_mission_range_from_batt_and_payload, self).__init__(aircraft)

    def eval(self,battery_mass,payload,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = owe+payload+battery_mass     # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,self.tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.battery_mass)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,self.tow,altp,mach,disa)


class E_mission_batt_from_range_and_payload(E_mission_batt_from_range_and_tow):
    """Specific mission evaluation from range and payload
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes total_fuel and tow from range and payload
    """
    def __init__(self, aircraft):
        super(E_mission_batt_from_range_and_payload, self).__init__(aircraft)

    def eval(self,range,payload,owe,altp,mach,disa):
        self.range = range     # Take Off Weight
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.payload = payload  # Mission payload

        def fct(tow):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  tow - (owe+self.payload+self.battery_mass)

        tow_ini = [self.aircraft.weight_cg.mtow]
        output_dict = fsolve(fct, x0=tow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.tow = output_dict[0][0]
        self.eval_breguet(self.range,self.tow,altp,mach,disa)


class Mission(Basic_mission):
    """Definition of all mission types for fuel powered airplanes
    """
    def __init__(self, aircraft):
        super(Mission, self).__init__(aircraft)
        self.aircraft = aircraft

        self.max_payload = Mission_range_from_payload_and_tow(aircraft)
        #self.max_payload2 = Mission_generic(aircraft)
        self.nominal = Mission_fuel_from_range_and_tow(aircraft)
        self.max_fuel = Mission_range_from_fuel_and_tow(aircraft)
        self.zero_payload = Mission_range_from_fuel_and_payload(aircraft)
        self.cost = Mission_fuel_from_range_and_payload(aircraft)

        self.ktow = 0.90    # TOW ratio at which cruise mean consumption is computed for fueled airplanes only

        self.crz_sar = None
        self.crz_cz = None
        self.crz_lod = None
        self.crz_thrust = None
        self.crz_throttle = None
        self.crz_sfc = None

        self.max_sar_altp = None
        self.max_sar = None
        self.max_sar_cz = None
        self.max_sar_lod = None
        self.max_sar_thrust = None
        self.max_sar_throttle = None
        self.max_sar_sfc = None

    def eval_cruise_point(self):
        """Evaluate cruise point characteristics
        """
        self.disa = self.aircraft.requirement.cruise_disa
        self.altp = self.aircraft.requirement.cruise_altp
        self.mach = self.aircraft.requirement.cruise_mach
        self.mass = self.ktow*self.aircraft.weight_cg.mtow

        pamb,tamb,tstd,dtodz = earth.atmosphere(self.altp, self.disa)

        lf_dict = self.level_flight(pamb,tamb,self.mach,self.mass)
        sm_dict = self.eval_max_sar(self.mass,self.mach,self.disa)

        self.crz_sar = lf_dict["sar"]
        self.crz_cz = lf_dict["cz"]
        self.crz_lod = lf_dict["lod"]
        self.crz_thrust = lf_dict["fn"]
        self.crz_throttle = lf_dict["thtl"]
        self.crz_sfc = lf_dict["sfc"]

        self.max_sar_altp = sm_dict["altp"]
        self.max_sar = sm_dict["sar"]
        self.max_sar_cz = sm_dict["cz"]
        self.max_sar_lod = sm_dict["lod"]
        self.max_sar_thrust = sm_dict["fn"]
        self.max_sar_throttle = sm_dict["thtl"]
        self.max_sar_sfc = sm_dict["sfc"]

    def mass_mission_adaptation(self):
        """Solves coupling between MTOW and OWE
        """
        range = self.aircraft.requirement.design_range
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach
        disa = self.aircraft.requirement.cruise_disa

        payload = self.aircraft.airframe.cabin.nominal_payload

        def fct(mtow):
            self.aircraft.weight_cg.mtow = mtow[0]
            self.aircraft.weight_cg.mass_pre_design()
            owe = self.aircraft.weight_cg.owe
            self.nominal.eval(range,mtow,owe,altp,mach,disa)
            fuel_total = self.nominal.fuel_total
            return mtow - (owe + payload + fuel_total)

        mtow_ini = [self.aircraft.weight_cg.mtow]
        output_dict = fsolve(fct, x0=mtow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.aircraft.weight_cg.mtow = output_dict[0][0]
        self.aircraft.weight_cg.mass_pre_design()
        self.aircraft.performance.mission.payload_range()


class Mission_fuel_from_range_and_tow(Flight):
    """Define common features for all mission types.
    """
    def __init__(self, aircraft):
        super(Mission_fuel_from_range_and_tow, self).__init__(aircraft)
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

    def eval(self,range,tow,owe,altp,mach,disa):
        """Evaluate mission and store results in object attributes
        """
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
        """Computing resulting payload
        """
        self.payload = self.tow - self.fuel_total - owe

    def breguet_range(self,range,tow,altp,mach,disa):
        """Breguet range equation is dependant from power source : fuel or battery
        """
        g = earth.gravity()

        pamb,tamb,tstd,dtodz = earth.atmosphere(altp, disa)
        tas = mach * earth.sound_speed(tamb)

        mass = self.aircraft.performance.mission.ktow*tow

        dict = self.level_flight(pamb,tamb,mach,mass)
        fuel = tow*(1-np.exp(-(dict["sfc"]*g*range)/(tas*dict["lod"])))
        time = 1.09*(range/tas)

        return fuel,time

    def eval_breguet(self,range,tow,altp,mach,disa):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        g = earth.gravity()
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
        fuel_mission,time_mission = self.breguet_range(range,tow,altp,mach,disa)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)     # mass is not landing weight

        # Arrival ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_landing = 1e-4*(0.5+2.3/engine_bpr)*mass
        time_landing = 180.

        fuel_taxi_in = (26. + 1.8e-4*reference_thrust)*n_engine
        time_taxi_in = 420.

        # Block fuel and time
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_block = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        #-----------------------------------------------------------------------------------------------------------
        fuel_diversion,t = self.breguet_range(self.diversion_range,tow,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.65 * mach
        pamb,tamb,tstd,dtodz = earth.atmosphere(altp_holding, disa)
        dict = self.level_flight(pamb,tamb,mach_holding,mass)
        fuel_holding = dict["sfc"]*(mass*g/dict["lod"])*self.holding_time

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission*self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.fuel_block + self.fuel_reserve

        #-----------------------------------------------------------------------------------------------------------
        return


class Mission_range_from_payload_and_tow(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from payload and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and total_fuel from payload and tow
    """
    def __init__(self, aircraft):
        super(Mission_range_from_payload_and_tow, self).__init__(aircraft)

    def eval(self,payload,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.fuel_total)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_tow(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from total fuel and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and payload from total_fuel and tow
    """
    def __init__(self, aircraft):
        super(Mission_range_from_fuel_and_tow, self).__init__(aircraft)

    def eval(self,fuel_total,tow,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = tow      # Take Off Weight
        self.payload = tow-owe-fuel_total  # Mission payload

        def fct(range):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.fuel_total)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,tow,altp,mach,disa)


class Mission_range_from_fuel_and_payload(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from payload and total fuel
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and tow from total_fuel and payload
    """
    def __init__(self, aircraft):
        super(Mission_range_from_fuel_and_payload, self).__init__(aircraft)

    def eval(self,fuel_total,payload,owe,altp,mach,disa):
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.tow = owe+payload+fuel_total     # Take Off Weight
        self.payload = payload  # Mission payload

        def fct(range):
            self.eval_breguet(range,self.tow,altp,mach,disa)
            return  self.tow - (owe+self.payload+self.fuel_total)

        range_ini = [self.aircraft.requirement.design_range]
        output_dict = fsolve(fct, x0=range_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.range = output_dict[0][0]                           # Coupling variable
        self.eval_breguet(self.range,self.tow,altp,mach,disa)


class Mission_fuel_from_range_and_payload(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from range and payload
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes total_fuel and tow from range and payload
    """
    def __init__(self, aircraft):
        super(Mission_fuel_from_range_and_payload, self).__init__(aircraft)

    def eval(self,range,payload,owe,altp,mach,disa):
        self.range = range     # Take Off Weight
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number
        self.payload = payload  # Mission payload

        def fct(tow):
            self.eval_breguet(range,tow,altp,mach,disa)
            return  tow - (owe+self.payload+self.fuel_total)

        tow_ini = [self.aircraft.weight_cg.mtow]
        output_dict = fsolve(fct, x0=tow_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        self.tow = output_dict[0][0]
        self.eval_breguet(self.range,self.tow,altp,mach,disa)








# TODO

class Mission_generic(Mission_fuel_from_range_and_tow):
    """Specific mission evaluation from payload and take off weight
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This version computes range and total_fuel from payload and tow
    """
    def __init__(self, aircraft):
        super(Mission_generic, self).__init__(aircraft)

    def eval(self,owe,altp,mach,disa,**kwargs):
        """Generic mission solver
        kwargs must contain affectations to the parameters that are fixed
        among the following list : range, tow, payload, fuel_total
        """
        range = 0.
        tow = 0.
        payload = 0.
        fuel_total = 0.

        vars = list(set(["range","tow","payload","fuel_total"])-set(kwargs.keys())) # extract variable names
        for key,val in kwargs.items():      # load parameter values, this quantities will not be modified
            exec(key+" = val")
            print(key,eval(key))
        print(payload,tow)
        raise Exception()

        def fct(x_in):
            for k,key in enumerate(vars):      # load variable values
                exec(key+" = x_in[k]")
            self.eval_breguet(range,tow,altp,mach,disa)         # eval Breguet equation, fuel_total is updated in the object
            return  [self.fuel_total - fuel_total,
                     self.tow - (owe+payload+self.fuel_total)]  # constraints residuals are sent back

        x_ini = np.zeros(2)
        for k,key in enumerate(vars):              # load init values from object
            if (key=="fuel_total"): x_ini[k] = 0.25*owe
            elif (key=="payload"): x_ini[k] = 0.25*owe
            elif (key=="range"): x_ini[k] = self.aircraft.requirement.design_range
            elif (key=="tow"): x_ini[k] = self.aircraft.weight_cg.mtow
        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        for k,key in enumerate(vars):              # get solution
            exec(key+" = output_dict[0][k]")
        self.eval_breguet(range,tow,altp,mach,disa)


class Mission_def(object):
    """Defines a mission evaluation for a fuel based propulsion system (kerozen, H2 ...etc)"""
    def __init__(self,aircraft):
        # Inputs
        self.aircraft = aircraft
        self.disa = None  # Mean cruise temperature shift
        self.altp = None  # Mean cruise altitude
        self.mach = None  # Cruise mach number
        self.range = None  # Mission distance
        self.owe = None # default Operating Weight Empty
        # Outputs
        self.tow = None  # Take Off Weight
        self.payload = None  # Mission payload
        self.time_block = None  # Mission block duration
        self.fuel_block = None  # Mission block fuel consumption
        self.fuel_reserve = None  # Mission reserve fuel
        self.fuel_total = None  # Mission total fuel

        self.holding_time = unit.s_min(30)  # Holding duration
        self.reserve_fuel_ratio = self.__reserve_fuel_ratio__()  # Ratio of mission fuel to account into reserve
        self.diversion_range = self.__diversion_range__()  # Diversion leg

    def set_mission_parameters(self,mach=None, altp=None, disa=None, owe=None):
        """Set the flight condition of the mission:
            1) reset to default aircraft requirements value if no value is specified
            2) Change only one attribut if a value is specified

        :param mach: cruise Mach number
        :param altp: cruise altitude
        :param disa: mean temperature shift
        :param owe: Operating Weight empty
        """
        if mach==None and altp==None and disa==None and owe==None: # 1: reset to default
            self.mach = self.aircraft.requirement.cruise_mach
            self.altp = self.aircraft.requirement.cruise_altp
            self.disa = self.aircraft.requirement.cruise_disa
            self.owe = self.aircraft.weight_cg.owe

        else:
            if mach != None:
                self.mach = mach
            if disa != None:
                self.disa = disa
            if owe != None:
                self.owe = owe
            if altp != None:
                self.altp = altp

    def eval(self, inputs={'range':None,'tow':None}, **kwargs):
        """Solve mission equations for given inputs.
        During a mission at given cruise mach, altitude, temperature shift (disa) and Operating Weight Empty (owe)
        the four following parameters are linked
            * tow : Take-Off Weight
            * payload : weight of Payload
            * range : mission range
            * fuel_total : weight of fuel taking into account safety margins
        by two equations :
            1) fSolve mission constraint for given inputs.
        During a mission at given cruise mach, altitude, temperature shift (disa) and Operating Weight Empty (owe)
        the four following variables are linked
            * tow : Take-Off Weight
            * payload : weight of Payload
            * range : mission range
            * fuel_total : weight of fuel taking into account safety margins
        by two equations :
            1) fuel_total = eval_Breguet(range,tow, altp, mach, disa)
            2) tow - payload - fuel_total - owe = 0
        By fixing two of the previous variables, we deduce the two remaining unknowns.

        :param inputs: a dictionary of two fixed parameters. Default is {'range','tow'}
        :param kwargs: optional named parameters for set_mission_parameters(**kwargs)
        :return: a dictionary of the two remaining unknown parameter. By default {'range':value, 'fuel_total':value}
        """
        # range,tow,altp,mach,disa
        # payload,tow,owe,altp,mach,disa
        # fuel_total,tow,owe,altp,mach,disa
        # fuel_total,payload,owe,altp,mach,disa
        # range,payload,owe,altp,mach,disa
        # range, tow, altp, mach, disa, payload, owe, fuel_total

        if len(kwargs)>0:
            self.set_mission_parameters(**kwargs)

        # Read the 2 inputs and store values in attributs
        for key,val in inputs:
            self.__dict__[key] = val

        # Build the unknown dict
        all_variables = ['range','tow','payload','fuel_total']
        unknowns = []
        for name in all_variables:
            if name not in inputs.keys():
                unknowns.append(name)
        unknowns = dict.fromkeys(unknowns) # Build an empty dict

        # TODO: implement the solve function

        self.bob
        self.range = range  # Mission distance
        self.tow = tow  # Take Off Weight
        self.eval_breguet(range, tow, altp, mach, disa)
        self.eval_payload(owe)

    def __reserve_fuel_ratio__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range > unit.m_NM(6500.)):
            reserve_fuel_ratio = 0.03
        else:
            reserve_fuel_ratio = 0.05
        return reserve_fuel_ratio

    def __diversion_range__(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range > unit.m_NM(200.)):
            diversion_range = unit.m_NM(200.)
        else:
            diversion_range = design_range
        return diversion_range

    def eval_payload(self, owe):
        """
        Computing resulting payload
        """
        self.payload = self.tow - self.fuel_total - owe

    def __mass_equation_to_solve__(self,unknowns,**kargs): # TODO
        all_variables = ['range','tow','payload','fuel_total']

        return kwargs['tow'] - kwargs['fuel_total'] - kwargs['owe'] - kwargs['payload']

    def eval_breguet(self, range, tow, altp, mach, disa):
        """
        Mission computation using breguet equation, fixed L/D and fixed sfc
        """

        g = earth.gravity()
        fhv = self.aircraft.power_system.fuel_heat
        n_engine = self.aircraft.airframe.nacelle.n_engine
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        engine_bpr = self.aircraft.airframe.nacelle.engine_bpr

        # Departure ground phases
        # -----------------------------------------------------------------------------------------------------------
        fuel_taxi_out = (34. + 2.3e-4 * reference_thrust) * n_engine
        time_taxi_out = 540.

        fuel_take_off = 1e-4 * (2.8 + 2.3 / engine_bpr) * tow
        time_take_off = 220. * tow / (reference_thrust * n_engine)

        # Mission leg
        # -----------------------------------------------------------------------------------------------------------
        fuel_mission, time_mission = self.aircraft.power_system.breguet_range(range, tow, altp, mach, disa)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)

        # Arrival ground phases
        # -----------------------------------------------------------------------------------------------------------
        fuel_landing = 1e-4 * (0.5 + 2.3 / engine_bpr) * mass
        time_landing = 180.

        fuel_taxi_in = (26. + 1.8e-4 * reference_thrust) * n_engine
        time_taxi_in = 420.

        # Block fuel and time
        # -----------------------------------------------------------------------------------------------------------
        self.block_fuel = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        # -----------------------------------------------------------------------------------------------------------
        fuel_diversion, t = self.aircraft.power_system.breguet_range(self.diversion_range, tow, altp, mach, disa)

        # Holding fuel
        # -----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.50 * mach
        pamb, tamb, tstd, dtodz = earth.atmosphere(altp_holding, disa)
        dict = self.aircraft.performance.level_flight(pamb, tamb, mach_holding, mass)
        fuel_holding = dict["sfc"] * (mass * g / dict["lod"]) * self.holding_time

        # Total
        # -----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission * self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.block_fuel + self.fuel_reserve

        # -----------------------------------------------------------------------------------------------------------
        return

