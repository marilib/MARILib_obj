#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from marilib.utils import earth, unit

import numpy as np
from scipy.optimize import fsolve

from marilib.aircraft.performance import Flight

class MissionBasic(Flight):
    """Definition of all mission types for fuel powered airplanes
    """
    def __init__(self, aircraft):
        super(MissionBasic, self).__init__(aircraft)
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
        fuel_max = self.aircraft.weight_cg.mfw
        nominal_payload = self.aircraft.airframe.cabin.nominal_payload
        design_range = self.aircraft.requirement.design_range
        cost_range = self.aircraft.requirement.cost_range

        disa = self.aircraft.requirement.cruise_disa
        altp = self.aircraft.requirement.cruise_altp
        mach = self.aircraft.requirement.cruise_mach

        self.max_payload.eval(owe,altp,mach,disa, payload=payload_max, tow=mtow)

        self.nominal.eval(design_range,mtow,owe,altp,mach,disa)

        self.max_fuel.eval(owe,altp,mach,disa, fuel_total=fuel_max, tow=mtow)

        self.zero_payload.eval(owe,altp,mach,disa, fuel_total=fuel_max, payload=0.)

        self.cost.eval(owe,altp,mach,disa, range=cost_range, payload=nominal_payload)


class Mission(MissionBasic):
    """Definition of all mission types for fuel powered airplanes
    """
    def __init__(self, aircraft):
        super(Mission, self).__init__(aircraft)
        self.aircraft = aircraft

        self.max_payload = MissionGeneric(aircraft)
        self.nominal = MissionNominal(aircraft)
        self.max_fuel = MissionGeneric(aircraft)
        self.zero_payload = MissionGeneric(aircraft)
        self.cost = MissionGeneric(aircraft)

        self.ktow = 0.90    # TOW ratio at which cruise mean consumption is computed

        self.crz_sar = None
        self.crz_cz = None
        self.crz_lod = None
        self.crz_thrust = None
        self.crz_throttle = None
        if self.aircraft.airframe.nacelle.sfc_type=="thrust":
            self.crz_tsfc = None
        elif self.aircraft.airframe.nacelle.sfc_type=="power":
            self.crz_psfc = None

        self.max_sar_altp = None
        self.max_sar = None
        self.max_sar_cz = None
        self.max_sar_lod = None
        self.max_sar_thrust = None
        self.max_sar_throttle = None
        if self.aircraft.airframe.nacelle.sfc_type=="thrust":
            self.max_sar_tsfc = None
        elif self.aircraft.airframe.nacelle.sfc_type=="power":
            self.max_sar_psfc = None

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
        if self.aircraft.airframe.nacelle.sfc_type=="thrust":
            self.crz_tsfc = lf_dict["sfc"]
        elif self.aircraft.airframe.nacelle.sfc_type=="power":
            self.crz_psfc = lf_dict["sfc"]

        self.max_sar_altp = sm_dict["altp"]
        self.max_sar = sm_dict["sar"]
        self.max_sar_cz = sm_dict["cz"]
        self.max_sar_lod = sm_dict["lod"]
        self.max_sar_thrust = sm_dict["fn"]
        self.max_sar_throttle = sm_dict["thtl"]
        if self.aircraft.airframe.nacelle.sfc_type=="thrust":
            self.max_sar_tsfc = sm_dict["sfc"]
        elif self.aircraft.airframe.nacelle.sfc_type=="power":
            self.max_sar_psfc = sm_dict["sfc"]

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


class MissionIsoMass(MissionBasic):
    """Definition of all mission types for battery powered airplanes
    """
    def __init__(self, aircraft):
        super(MissionIsoMass, self).__init__(aircraft)
        self.aircraft = aircraft

        self.max_payload = MissionIsoMassGeneric(aircraft)
        self.nominal = MissionIsoMassNominal(aircraft)
        self.max_fuel = MissionIsoMassGeneric(aircraft)
        self.zero_payload = MissionIsoMassGeneric(aircraft)
        self.cost = MissionIsoMassGeneric(aircraft)

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
        self.crz_power = lf_dict["sec"]*lf_dict["fn"]
        self.crz_sec = lf_dict["sec"]

        self.max_esar_altp = sm_dict["altp"]
        self.max_esar = sm_dict["sar"]
        self.max_esar_cz = sm_dict["cz"]
        self.max_esar_lod = sm_dict["lod"]
        self.max_esar_thrust = sm_dict["fn"]
        self.max_esar_throttle = sm_dict["thtl"]
        self.max_esar_power = sm_dict["sec"]*sm_dict["fn"]
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


class MissionNominal(Flight):
    """Define common features for all mission types.
    """
    def __init__(self, aircraft):
        super(MissionNominal, self).__init__(aircraft)
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
        self.reserve_fuel_ratio = self.reserve_fuel_ratio() # Ratio of mission fuel to account into reserve
        self.diversion_range = self.diversion_range()       # Diversion leg

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

    def reserve_fuel_ratio(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(6500.)):
            reserve_fuel_ratio = 0.03
        else:
            reserve_fuel_ratio = 0.05
        return reserve_fuel_ratio

    def diversion_range(self):
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

    def eval_breguet(self,range,tow,altp,mach,disa):
        """
        Mission computation using bregueçt equation, fixed L/D and fixed sfc
        """
        g = earth.gravity()

        n_engine = self.aircraft.airframe.nacelle.n_engine
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        engine_bpr = self.aircraft.airframe.nacelle.engine_bpr
        ktow = self.aircraft.performance.mission.ktow

        # Departure ground phases
        #-----------------------------------------------------------------------------------------------------------
        fuel_taxi_out = (34. + 2.3e-4*reference_thrust)*n_engine
        time_taxi_out = 540.

        fuel_take_off = 1e-4*(2.8+2.3/engine_bpr)*tow
        time_take_off = 220.*tow/(reference_thrust*n_engine)

        # Mission leg
        #-----------------------------------------------------------------------------------------------------------
        fuel_mission,time_mission = self.breguet_range(range,tow,ktow,altp,mach,disa)

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
        fuel_diversion,t = self.breguet_range(self.diversion_range,mass,0.99,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.65 * mach
        fuel_holding = self.holding(self.holding_time,mass,altp_holding,mach_holding,disa)

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission*self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.fuel_block + self.fuel_reserve

        #-----------------------------------------------------------------------------------------------------------
        return


class MissionGeneric(MissionNominal):
    """Generic mission evaluation
    Four variables are driving mission computation : total_fuel, tow, payload & range
    Two of them are necessary to compute the two others
    This class computes a mission from 2 input among 4
    """
    def __init__(self, aircraft):
        super(MissionGeneric, self).__init__(aircraft)

    def eval(self,owe,altp,mach,disa,**kwargs):
        """Generic mission solver
        kwargs must contain affectations to the parameters that are fixed
        among the following list : range, tow, payload, fuel_total
        """
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number

        range = [0.]
        tow = [0.]
        payload = [0.]
        fuel_total = [0.]

        for key,val in kwargs.items():      # load parameter values, this quantities will not be modified
            exec(key+"[0] = val")

        vars = list(set(["range","tow","payload","fuel_total"])-set(kwargs.keys())) # extract variable names

        def fct(x_in):
            for k,key in enumerate(vars):      # load variable values
                exec(key+"[0] = x_in[k]")
            self.eval_breguet(range[0],tow[0],altp,mach,disa)         # eval Breguet equation, fuel_total is updated in the object
            return  [self.fuel_total - fuel_total[0],
                     tow[0] - (owe+payload[0]+self.fuel_total)]  # constraints residuals are sent back

        x_ini = np.zeros(2)
        for k,key in enumerate(vars):              # load init values from object
            if (key=="fuel_total"): x_ini[k] = 0.25*owe
            elif (key=="payload"): x_ini[k] = 0.25*owe
            elif (key=="range"): x_ini[k] = self.aircraft.requirement.design_range
            elif (key=="tow"): x_ini[k] = self.aircraft.weight_cg.mtow
        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        for k,key in enumerate(vars):              # get solution
            exec(key+"[0] = output_dict[0][k]")
        self.eval_breguet(range[0],tow[0],altp,mach,disa)
        self.range = range[0]
        self.tow = tow[0]
        self.payload = payload[0]


class MissionIsoMassNominal(Flight):
    """Define common features for all mission types.
    """
    def __init__(self, aircraft):
        super(MissionIsoMassNominal, self).__init__(aircraft)
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
        self.reserve_enrg_ratio = self.reserve_enrg_ratio() # Ratio of mission fuel to account into reserve
        self.diversion_range = self.diversion_range()       # Diversion leg

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

    def reserve_enrg_ratio(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range> unit.m_NM(6500.)):
            reserve_enrg_ratio = 0.03
        else:
            reserve_enrg_ratio = 0.05
        return reserve_enrg_ratio

    def diversion_range(self):
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
        enrg_mission,time_mission = self.breguet_range(range,tow,1.,altp,mach,disa)

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
        enrg_diversion,t = self.breguet_range(self.diversion_range,tow,1.,altp,mach,disa)

        # Holding fuel
        #-----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.65 * mach
        enrg_holding = self.holding(self.holding_time,tow,altp_holding,mach_holding,disa)

        # Total
        #-----------------------------------------------------------------------------------------------------------
        self.enrg_reserve = enrg_mission*self.reserve_enrg_ratio + enrg_diversion + enrg_holding
        self.enrg_total = self.enrg_block + self.enrg_reserve
        self.battery_mass = self.enrg_total / self.aircraft.airframe.system.battery_energy_density

        #-----------------------------------------------------------------------------------------------------------
        return


class MissionIsoMassGeneric(MissionIsoMassNominal):
    """Generic mission evaluation
    Four variables are driving mission computation : battery_mass, tow, payload & range
    Two of them are necessary to compute the two others
    This class computes a mission from 2 input among 4
    """
    def __init__(self, aircraft):
        super(MissionIsoMassGeneric, self).__init__(aircraft)

    def eval(self,owe,altp,mach,disa,**kwargs):
        """Generic mission solver
        kwargs must contain affectations to the parameters that are fixed
        among the following list : range, tow, payload, fuel_total
        """
        self.disa = disa    # Mean cruise temperature shift
        self.altp = altp    # Mean cruise altitude
        self.mach = mach    # Cruise mach number

        range = [0.]
        tow = [0.]
        payload = [0.]
        fuel_total = [0.]

        for key,val in kwargs.items():      # load parameter values, this quantities will not be modified
            exec(key+"[0] = val")

        vars = list(set(["range","tow","payload","fuel_total"])-set(kwargs.keys())) # extract variable names

        def fct(x_in):
            for k,key in enumerate(vars):      # load variable values
                exec(key+"[0] = x_in[k]")
            self.eval_breguet(range[0],tow[0],altp,mach,disa)         # eval Breguet equation, fuel_total is updated in the object
            return  [self.battery_mass - fuel_total[0],
                     tow[0] - (owe+payload[0]+self.battery_mass)]  # constraints residuals are sent back

        x_ini = np.zeros(2)
        for k,key in enumerate(vars):              # load init values from object
            if (key=="fuel_total"): x_ini[k] = 0.25*owe
            elif (key=="payload"): x_ini[k] = 0.25*owe
            elif (key=="range"): x_ini[k] = self.aircraft.requirement.design_range
            elif (key=="tow"): x_ini[k] = self.aircraft.weight_cg.mtow
        output_dict = fsolve(fct, x0=x_ini, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")

        for k,key in enumerate(vars):              # get solution
            exec(key+"[0] = output_dict[0][k]")
        self.eval_breguet(range[0],tow[0],altp,mach,disa)
        self.range = range[0]
        self.tow = tow[0]
        self.payload = payload[0]


class MissionDef(Flight):
    """Defines a mission evaluation for a fuel based propulsion system (kerozen, H2 ...etc)

    .. warning::
        This class is not used. By default MARILib uses :class:`MissionGeneric`."""
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
        self.reserve_fuel_ratio = self.__reserve_fuel_ratio()  # Ratio of mission fuel to account into reserve
        self.diversion_range = self.diversion_range()  # Diversion leg

    def set_parameters(self, mach=None, altp=None, disa=None, owe=None):
        """Set the flight condition of the mission:
            1) if one or more value is specified : set the corresponding mission attributes
            2) if no value is specified : reset mission parameters to default aircraft requirements

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
        the four following parameters are linked together:

        - tow : Take-Off Weight
        - payload : weight of Payload
        - range : mission range
        - fuel_total : weight of fuel taking into account safety margins

        by two equations :

        1) fuel_total = `eval_Breguet(range,tow, altp, mach, disa)`
        2) tow - payload - fuel_total - owe = 0

        By fixing two of the previous variables, we deduce the two remaining unknowns.

        :param inputs: a dictionary of two fixed parameters. Default is `{'range','tow'}`
        :param kwargs: optional named parameters for :py:meth:`set_parameters`
        :return: a dictionary of the two remaining unknown parameter. By default `{'range', 'fuel_total'}`.

        Throws error if `fsolve` does not converge.
        """

        if len(kwargs)>0:
            self.set_parameters(**kwargs)

        # Build the unknown dict
        all_variables = ['range','tow','payload','fuel_total']
        unknowns = []
        for name in sorted(all_variables):
            if name not in inputs.keys():
                unknowns.append(name) # add a new unknown
            else:
                self.__dict__[name] = inputs[name] # set the input value in the mission attribute

        x0 = self.__init_unknowns(unknowns)
        x = fsolve(self.__mission_equations_to_solve, x0, args=({'inputs': inputs.keys(), 'unknowns': unknowns}))

        output = {}
        for k,solution in enumerate(x): # store the solutions
            self.__dict__[unknowns[k]] = solution # set the value in the mission attribute
            output[unknowns[k]] = solution

        return output


    def __init_unknowns(self,unknowns):
        """Initialize the value of the unknowns before calling fsolve
        :param unknowns: a list of two variable names in ['range', 'tow', 'payload', 'fuel_total']
        :return: two init values
        """
        x0 = [None,None]
        for k,unknown in enumerate(sorted(unknowns)):
            if (unknown == "fuel_total"):
                x0[k] = 0.25 * self.owe
            elif (unknown == "payload"):
                x0[k] = 0.25 * self.owe
            elif (unknown == "range"):
                x0[k] = self.aircraft.requirement.design_range
            elif (unknown == "tow"):
                x0[k] = self.aircraft.weight_cg.mtow
        return x0

    def __mission_equations_to_solve(self, unknowns, *args):
        """The set of two equations to solve to determine the two unknowns:
            1) `fuel_total - eval_Breguet(range,tow, altp, mach, disa) = 0`
            2) `tow - payload - fuel_total - owe = 0`
        :param unknowns: a list of two guess values for the unknowns
        :param args: a tuple containing a dict of inputs and unknowns names in ['range', 'tow', 'payload', 'fuel_total']
        :return: the values of the two equations
        """
        inputs = args[0]['inputs']
        unknowns_name = args[0]['unknowns']

        xx = {'range':None, 'tow':None, 'payload':None, 'fuel_total':None}

        k = 0
        for name in sorted(xx.keys()): # iterate over the list
            if name in inputs:
                xx[name] = self.__dict__[name]  # read the value  of the attribute
            elif name in unknowns_name:
                xx[name] = unknowns[k]  # read the value in the unknown list : the order matters -> sorted()
                k += 1

        self.eval_breguet(xx['range'], xx['tow'], self.altp, self.mach, self.disa)
        eq1 = xx['fuel_total'] - self.fuel_total
        eq2 = xx['tow'] - xx['fuel_total'] - self.owe - xx['payload']

        return eq1,eq2

    def __reserve_fuel_ratio(self):
        design_range = self.aircraft.requirement.design_range
        if (design_range > unit.m_NM(6500.)):
            reserve_fuel_ratio = 0.03
        else:
            reserve_fuel_ratio = 0.05
        return reserve_fuel_ratio

    def diversion_range(self):
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

    def eval_breguet(self, range, tow, altp, mach, disa):
        """
        Mission computation using breguet equation, fixed L/D and fixed sfc
        """
        g = earth.gravity()

        n_engine = self.aircraft.airframe.nacelle.n_engine
        reference_thrust = self.aircraft.airframe.nacelle.reference_thrust
        engine_bpr = self.aircraft.airframe.nacelle.engine_bpr
        ktow = self.aircraft.performance.mission.ktow

        # Departure ground phases
        # -----------------------------------------------------------------------------------------------------------
        fuel_taxi_out = (34. + 2.3e-4 * reference_thrust) * n_engine
        time_taxi_out = 540.

        fuel_take_off = 1e-4 * (2.8 + 2.3 / engine_bpr) * tow
        time_take_off = 220. * tow / (reference_thrust * n_engine)

        # Mission leg
        # -----------------------------------------------------------------------------------------------------------
        fuel_mission,time_mission = self.breguet_range(range,tow,ktow,altp,mach,disa)

        mass = tow - (fuel_taxi_out + fuel_take_off + fuel_mission)  # mass is not landing weight

        # Arrival ground phases
        # -----------------------------------------------------------------------------------------------------------
        fuel_landing = 1e-4 * (0.5 + 2.3 / engine_bpr) * mass
        time_landing = 180.

        fuel_taxi_in = (26. + 1.8e-4 * reference_thrust) * n_engine
        time_taxi_in = 420.

        # Block fuel and time
        # -----------------------------------------------------------------------------------------------------------
        self.fuel_block = fuel_taxi_out + fuel_take_off + fuel_mission + fuel_landing + fuel_taxi_in
        self.time_block = time_taxi_out + time_take_off + time_mission + time_landing + time_taxi_in

        # Diversion fuel
        # -----------------------------------------------------------------------------------------------------------
        fuel_diversion,t = self.breguet_range(self.diversion_range,mass,0.99,altp,mach,disa)

        # Holding fuel
        # -----------------------------------------------------------------------------------------------------------
        altp_holding = unit.m_ft(1500.)
        mach_holding = 0.65 * mach
        fuel_holding = self.holding(self.holding_time,mass,altp_holding,mach_holding,disa)

        # Total
        # -----------------------------------------------------------------------------------------------------------
        self.fuel_reserve = fuel_mission * self.reserve_fuel_ratio + fuel_diversion + fuel_holding
        self.fuel_total = self.fuel_block + self.fuel_reserve

        # -----------------------------------------------------------------------------------------------------------
        return

