#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC

"""

from marilib.utils import unit

from marilib.aircraft.model_config import get_init


class Requirement(object):
    """Initialize top level aircraft requirements
    """
    def __init__(self, n_pax_ref = 150.,
                 design_range = unit.m_NM(3000.),
                 cruise_mach = 0.78,
                 cruise_altp = unit.m_ft(35000.),
                 arrangement = None):

        self.cruise_disa = 0.
        self.cruise_altp = cruise_altp
        self.cruise_mach = cruise_mach
        self.design_range = design_range
        self.cost_range = get_init(self,"cost_range", val=self.__cost_mission_range())

        self.n_pax_ref = n_pax_ref

        self.take_off = TakeOffReq(arrangement, self)
        self.approach = ApproachReq(arrangement, self)
        self.oei_ceiling = OeiCeilingReq(arrangement, self)
        self.mcl_ceiling = MclCeilingReq(arrangement, self)
        self.mcr_ceiling = McrCeilingReq(arrangement, self)
        self.time_to_climb = TtcReq(arrangement, self)

    def __cost_mission_range(self):
        if(self.design_range < unit.m_NM(400.)): cost_mission_range = unit.m_NM(100.)
        elif(self.design_range < unit.m_NM(1000.)): cost_mission_range = unit.m_NM(200.)
        elif(self.design_range < unit.m_NM(2500.)): cost_mission_range = unit.m_NM(400.)
        elif(self.design_range < unit.m_NM(4500.)): cost_mission_range = unit.m_NM(800.)
        elif(self.design_range < unit.m_NM(6500.)): cost_mission_range = unit.m_NM(2000.)
        else: cost_mission_range = unit.m_NM(4000.)
        return cost_mission_range


class TakeOffReq(object):
    """Initialize take off requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = get_init(self,"disa")
        self.altp = get_init(self,"altp")
        self.kmtow = get_init(self,"kmtow")
        self.kvs1g = get_init(self,"kvs1g")
        self.s2_min_path = get_init(self,"s2_min_path", val=self.__s2_min_path(arrangement))
        self.tofl_req = get_init(self,"tofl_req", val=self.__tofl_req(requirement))

    def __s2_min_path(self,arrangement):
        """Regulatory min climb path versus number of engine
        """
        if(arrangement.number_of_engine == "twin"): s2_min_path = 0.024
     #   elif(arrangement.number_of_engine == "tri"): s2_min_path = 0.027
        elif(arrangement.number_of_engine >= "quadri"): s2_min_path = 0.030
        else: raise Exception("number of engine is not permitted")
        return s2_min_path

    def __tofl_req(self, requirement):
        if(requirement.design_range <= unit.m_NM(1500.)): tofl_req = 1500.
        elif(requirement.design_range <= unit.m_NM(3500.)): tofl_req = 2000.
        elif(requirement.design_range <= unit.m_NM(5500.)): tofl_req = 2500.
        else: tofl_req = 3000.
        return tofl_req


class ApproachReq(object):
    """Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = get_init(self,"disa")
        self.altp = get_init(self,"altp")
        self.kmlw = get_init(self,"kmlw")
        self.kvs1g = get_init(self,"kvs1g")
        self.app_speed_req = get_init(self,"app_speed_req", val=self.__app_speed_req(requirement))

    def __app_speed_req(self, requirement):
        if (requirement.n_pax_ref<=100): app_speed_req = unit.mps_kt(120.)
        elif (requirement.n_pax_ref<=200): app_speed_req = unit.mps_kt(137.)
        else: app_speed_req = unit.mps_kt(140.)
        return app_speed_req


class OeiCeilingReq(object):
    """Initialize one engine inoperative ceiling requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = get_init(self,"disa")
        self.altp = get_init(self,"altp", val=0.50*requirement.cruise_altp)
        self.kmtow = get_init(self,"kmtow")
        self.rating = get_init(self,"rating")
        self.speed_mode = get_init(self,"speed_mode")
        self.path_req = get_init(self,"path_req", val=self.__oei_min_path(arrangement))

    def __oei_min_path(self, arrangement):
        """Regulatory min climb path depending on the number of engine
        """
        if(arrangement.number_of_engine == "twin"): oei_min_path = 0.011
     #   elif(arrangement.number_of_engine == "tri"):  oei_min_path = 0.013
        elif(arrangement.number_of_engine >= "quadri"): oei_min_path = 0.016
        else: raise Exception("number of engine is not permitted")
        return oei_min_path


class ClimbReq(object):
    """Initialize climb speed requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = get_init(self,"disa")
        self.altp = get_init(self,"altp", val=self.top_of_climb(arrangement,requirement))
        self.mach = get_init(self,"mach", val=requirement.cruise_mach)
        self.kmtow = get_init(self,"kmtow")

    def top_of_climb(self, arrangement, requirement):
        if (arrangement.power_architecture in ["tf","extf"]): altp = unit.m_ft(35000.)
        elif (arrangement.power_architecture=="tp"): altp = unit.m_ft(16000.)
        elif (arrangement.power_architecture in ["ef","exef"]): altp = unit.m_ft(31000.)
        elif (arrangement.power_architecture=="ep"): altp = unit.m_ft(16000.)
        elif (arrangement.power_architecture=="pte"): altp = unit.m_ft(31000.)
        else: raise Exception("propulsion.architecture index is out of range")
        top_of_climb = min(altp, requirement.cruise_altp - unit.m_ft(4000.))
        return top_of_climb


class MclCeilingReq(ClimbReq):
    """Initialize climb speed requirements in MCL rating
    """
    def __init__(self, arrangement, requirement):
        super(MclCeilingReq, self).__init__(arrangement, requirement)
        self.rating = get_init(self,"rating")
        self.speed_mode = get_init(self,"speed_mode")
        self.vz_req = get_init(self,"vz_req", val=unit.mps_ftpmin(300.))


class McrCeilingReq(ClimbReq):
    """Initialize climb speed requirements in MCR rating
    """
    def __init__(self, arrangement, requirement):
        super(McrCeilingReq, self).__init__(arrangement, requirement)
        self.rating = get_init(self,"rating")
        self.speed_mode = get_init(self,"speed_mode")
        self.vz_req = get_init(self,"vz_req", val=unit.mps_ftpmin(0.))


class TtcReq(ClimbReq):
    """Initialize time to climb requirements
    """
    def __init__(self, arrangement, requirement):
        super(TtcReq, self).__init__(arrangement, requirement)
        self.cas1 = get_init(self,"cas1", val=self.__ttc_cas1(requirement))
        self.altp1 = get_init(self,"altp1")
        self.cas2 = get_init(self,"cas2", val=self.__ttc_cas2(requirement))
        self.altp2 = get_init(self,"altp2")
        self.altp = get_init(self,"altp", val=self.top_of_climb(arrangement,requirement))
        self.ttc_req = get_init(self,"ttc_req")

    def __ttc_cas1(self, requirement):
        if (requirement.cruise_mach>=0.6): cas1 = unit.mps_kt(180.)
        elif (requirement.cruise_mach>=0.4): cas1 = unit.mps_kt(130.)
        else: cas1 = unit.mps_kt(70.)
        return cas1

    def __ttc_cas2(self, requirement):
        if (requirement.cruise_mach>=0.6): cas2 = unit.mps_kt(250.)
        elif (requirement.cruise_mach>=0.4): cas2 = unit.mps_kt(200.)
        else: cas2 = unit.mps_kt(70.)
        return cas2


