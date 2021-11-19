#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC

"""

from marilib.utils import unit

from marilib.aircraft.model_config import ModelConfiguration


class Requirement(object):

    def __init__(self, n_pax_ref = 150.,
                 design_range = unit.m_NM(3000.),
                 cruise_mach = 0.78,
                 cruise_altp = unit.m_ft(35000.),
                 model_config=None):
        """Initialize top level aircraft requirements. The default requirements are for a typical jet liner.

        :param n_pax_ref: number of passangers. Default is 150.
        :param design_range: design range of the aircraft in meters. Default is equivalent to 3000 NM.
        :param cruise_mach: cruise Mach Number. Default is 0.78.
        :param cruise_altp: cruise altitude in meters. Default is equivalent to 35 000 ft.
        """
        self.cruise_altp = cruise_altp
        self.cruise_mach = cruise_mach
        self.design_range = design_range
        self.n_pax_ref = n_pax_ref

        if model_config is None:
            self.model_config = ModelConfiguration()
        else:
            self.model_config = model_config()
        
    def init_all_requirements(self,arrangement):
        """Initialize the following categories of requirements:

         * Take-Off : :class:`TakeOffReq`
         * Approach : :class:`ApproachReq`
         * One Engine Inoperative ceiling : :class:`OeiCeilingReq`
         * Maximum Climb thrust : :class:`MclCeilingReq`
         * Maximum Cruise thrust : :class:`McrCeilingReq`
         * Time to climb requirements : :class:`TtcReq`

         """
        self.cruise_disa = 0.
        self.cost_range = self.model_config.get__init(self, "cost_range", val=self.__cost_mission_range())
        self.max_fuel_range_factor = self.model_config.get__init(self, "max_fuel_range_factor")
        self.max_body_aspect_ratio = self.model_config.get__init(self, "max_body_aspect_ratio")
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
        self.disa = requirement.model_config.get__init(self,"disa")
        self.altp = requirement.model_config.get__init(self,"altp")
        self.kmtow = requirement.model_config.get__init(self,"kmtow")
        self.kvs1g = requirement.model_config.get__init(self,"kvs1g")
        self.s2_min_path = requirement.model_config.get__init(self,"s2_min_path", val=self.__s2_min_path(arrangement))
        self.tofl_req = requirement.model_config.get__init(self,"tofl_req", val=self.__tofl_req(requirement))

    def __s2_min_path(self,arrangement):
        """Regulatory min climb path versus number of engine
        """
        if(arrangement.number_of_engine == "twin"): s2_min_path = 0.024
     #   elif(arrangement.number_of_engine == "tri"): s2_min_path = 0.027
        elif(arrangement.number_of_engine == "quadri"): s2_min_path = 0.030
        elif(arrangement.number_of_engine == "hexa"): s2_min_path = 0.033
        else: raise Exception("number of engine is not permitted")
        return s2_min_path

    def __tofl_req(self, requirement):
        if(requirement.design_range <= unit.m_NM(500.)): tofl_req = 1200.
        elif(requirement.design_range <= unit.m_NM(1500.)): tofl_req = 1500.
        elif(requirement.design_range <= unit.m_NM(3500.)): tofl_req = 2300.
        elif(requirement.design_range <= unit.m_NM(5500.)): tofl_req = 2800.
        else: tofl_req = 3000.
        return tofl_req


class ApproachReq(object):
    """Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = requirement.model_config.get__init(self,"disa")
        self.altp = requirement.model_config.get__init(self,"altp")
        self.kmlw = requirement.model_config.get__init(self,"kmlw")
        self.kvs1g = requirement.model_config.get__init(self,"kvs1g")
        self.app_speed_req = requirement.model_config.get__init(self,"app_speed_req", val=self.__app_speed_req(requirement))

    def __app_speed_req(self, requirement):
        if (requirement.n_pax_ref<=40): app_speed_req = unit.mps_kt(110.)
        elif (requirement.n_pax_ref<=100): app_speed_req = unit.mps_kt(120.)
        elif (requirement.n_pax_ref<=200): app_speed_req = unit.mps_kt(137.)
        else: app_speed_req = unit.mps_kt(140.)
        return app_speed_req


class OeiCeilingReq(object):
    """Initialize one engine inoperative ceiling requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = requirement.model_config.get__init(self,"disa")
        if (arrangement.power_architecture in ["tp","ep"]):
            self.altp = requirement.model_config.get__init(self,"altp", val=unit.m_ft(5000))
        else:
            self.altp = requirement.model_config.get__init(self,"altp", val=0.40*requirement.cruise_altp)
        self.kmtow = requirement.model_config.get__init(self,"kmtow")
        self.rating = requirement.model_config.get__init(self,"rating")
        self.speed_mode = requirement.model_config.get__init(self,"speed_mode")
        self.path_req = requirement.model_config.get__init(self,"path_req", val=self.__oei_min_path(arrangement))

    def __oei_min_path(self, arrangement):
        """Regulatory min climb path depending on the number of engine
        """
        if(arrangement.number_of_engine == "twin"): oei_min_path = 0.011
     #   elif(arrangement.number_of_engine == "tri"):  oei_min_path = 0.013
        elif(arrangement.number_of_engine >= "quadri"): oei_min_path = 0.016
        elif(arrangement.number_of_engine >= "hexa"): oei_min_path = 0.019
        else: raise Exception("number of engine is not permitted")
        return oei_min_path


class ClimbReq(object):
    """A generic Climb requirement definition
    """
    def __init__(self, arrangement, requirement):
        self.disa = requirement.model_config.get__init(self,"disa")
        self.altp = requirement.model_config.get__init(self,"altp", val=self.top_of_climb(arrangement,requirement))
        self.mach = requirement.model_config.get__init(self,"mach", val=self.trajectory_speed(arrangement,requirement))
        self.kmtow = requirement.model_config.get__init(self,"kmtow")

    def trajectory_speed(self, arrangement, requirement):
        if (arrangement.power_architecture in ["tp","ep"]):
            if 0.3<requirement.cruise_mach:
                mach = requirement.cruise_mach - 0.10
            else:
                mach = requirement.cruise_mach
        else:
            mach = requirement.cruise_mach
        return mach

    def top_of_climb(self, arrangement, requirement):
        if (arrangement.power_architecture in ["tp","ep"]):
            altp = unit.m_ft(16000.)
        else:
            altp = unit.m_ft(35000.)
        # top_of_climb = min(altp, requirement.cruise_altp - unit.m_ft(4000.))
        return altp


class MclCeilingReq(ClimbReq):
    """Initialize climb speed requirements in **Maximum CLimb** thrust rating
    """
    def __init__(self, arrangement, requirement):
        super(MclCeilingReq, self).__init__(arrangement, requirement)
        self.rating = requirement.model_config.get__init(self,"rating")
        self.speed_mode = requirement.model_config.get__init(self,"speed_mode")
        if (arrangement.power_architecture in ["tp","ep"]):
            self.vz_req = requirement.model_config.get__init(self,"vz_req", val=unit.mps_ftpmin(100.))
        else:
            self.vz_req = requirement.model_config.get__init(self,"vz_req", val=unit.mps_ftpmin(300.))


class McrCeilingReq(ClimbReq):
    """Initialize climb speed requirements in **Maximum CRuise** thrust rating
    """
    def __init__(self, arrangement, requirement):
        super(McrCeilingReq, self).__init__(arrangement, requirement)
        self.rating = requirement.model_config.get__init(self,"rating")
        self.speed_mode = requirement.model_config.get__init(self,"speed_mode")
        self.vz_req = requirement.model_config.get__init(self,"vz_req", val=unit.mps_ftpmin(0.))


class TtcReq(ClimbReq):
    """Initialize time to climb requirements
    """
    def __init__(self, arrangement, requirement):
        super(TtcReq, self).__init__(arrangement, requirement)
        self.cas1 = requirement.model_config.get__init(self,"cas1", val=self.__ttc_cas1(requirement))
        self.altp1 = requirement.model_config.get__init(self,"altp1")
        self.cas2 = requirement.model_config.get__init(self,"cas2", val=self.__ttc_cas2(requirement))
        self.altp2 = requirement.model_config.get__init(self,"altp2")
        self.altp = requirement.model_config.get__init(self,"altp", val=self.top_of_climb(arrangement,requirement))
        self.ttc_req = requirement.model_config.get__init(self,"ttc_req")

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


