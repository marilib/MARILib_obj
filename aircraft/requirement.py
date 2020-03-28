#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin

"""

from aircraft.tool import unit


class Take_off_req(object):
    """
    Initialize take off requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = 15.
        self.altp = unit.m_ft(0.)
        self.kmtow = 1.
        self.kvs1g = 1.13
        self.s2_min_path = self.__s2_min_path__(arrangement)
        self.tofl_req = self.__tofl_req__(requirement)

    def __s2_min_path__(self,arrangement):
        """
        Regulatory min climb path versus number of engine
        """
        if(arrangement.number_of_engine == "twin"): s2_min_path = 0.024
     #   elif(arrangement.number_of_engine == "tri"): s2_min_path = 0.027
        elif(arrangement.number_of_engine >= "quadri"): s2_min_path = 0.030
        else: raise Exception("number of engine is not permitted")
        return s2_min_path

    def __tofl_req__(self, requirement):
        if(requirement.design_range <= unit.m_NM(1500.)): tofl_req = 1500.
        elif(requirement.design_range <= unit.m_NM(3500.)): tofl_req = 2000.
        elif(requirement.design_range <= unit.m_NM(5500.)): tofl_req = 2500.
        else: tofl_req = 3000.
        return tofl_req


class Approach_req(object):
    """
    Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = 15.
        self.altp = unit.m_ft(0.)
        self.kmlw = 1.
        self.kvs1g = 1.23
        self.app_speed_req = self.__app_speed_req__(requirement)

    def __app_speed_req__(self, requirement):
        if (requirement.n_pax_ref<=100): app_speed_req = unit.mps_kt(120.)
        elif (requirement.n_pax_ref<=200): app_speed_req = unit.mps_kt(137.)
        else: app_speed_req = unit.mps_kt(140.)
        return app_speed_req


class OEI_req(object):
    """
    Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = 15.
        self.altp = unit.m_ft(11000.)
        self.kmtow = 0.95
        self.min_path = self.__oei_min_path__(arrangement)

    def __oei_min_path__(self, arrangement):
        """
        Regulatory min climb path depending on the number of engine
        """
        if(arrangement.number_of_engine == "twin"): oei_min_path = 0.011
     #   elif(arrangement.number_of_engine == "tri"):  oei_min_path = 0.013
        elif(arrangement.number_of_engine >= "quadri"): oei_min_path = 0.016
        else: raise Exception("number of engine is not permitted")
        return oei_min_path


class Climb_req(object):
    """
    Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        self.disa = 15.
        self.altp = self.__top_of_climb__(arrangement,requirement)
        self.kmtow = 0.97

    def __top_of_climb__(self, arrangement, requirement):
        if (arrangement.power_architecture=="tf"): altp = unit.m_ft(31000.)
        elif (arrangement.power_architecture=="tp"): altp = unit.m_ft(16000.)
        elif (arrangement.power_architecture=="pte1"): altp = unit.m_ft(31000.)
        elif (arrangement.power_architecture=="ef1"): altp = unit.m_ft(21000.)
        elif (arrangement.power_architecture=="ep1"): altp = unit.m_ft(16000.)
        else: raise Exception("propulsion.architecture index is out of range")
        top_of_climb = min(altp, requirement.cruise_altp - unit.m_ft(4000.))
        return top_of_climb


class Vz_mcl_req(Climb_req):
    """
    Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        super(Vz_mcl_req, self).__init__(arrangement, requirement)
        self.mach = requirement.cruise_mach
        self.vz = unit.mps_ftpmin(300.)


class Vz_mcr_req(Climb_req):
    """
    Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        super(Vz_mcr_req, self).__init__(arrangement, requirement)
        self.mach = requirement.cruise_mach
        self.vz = unit.mps_ftpmin(0.)


class TTC_req(Climb_req):
    """
    Initialize approach requirements
    """
    def __init__(self, arrangement, requirement):
        super(TTC_req, self).__init__(arrangement, requirement)
        self.cas1 = self.__ttc_cas1__(requirement),
        self.altp1 = unit.m_ft(1500.),
        self.cas2 = self.__ttc_cas2__(requirement),
        self.altp2 = unit.m_ft(10000.),
        self.ttc = unit.s_min(25.)

    def __ttc_cas1__(self, requirement):
        if (requirement.cruise_mach>=0.6): cas1 = unit.mps_kt(250.)
        elif (requirement.cruise_mach>=0.4): cas1 = unit.mps_kt(180.)
        else: cas1 = unit.mps_kt(70.)
        return cas1

    def __ttc_cas2__(self, requirement):
        if (requirement.cruise_mach>=0.6): cas2 = unit.mps_kt(300.)
        elif (requirement.cruise_mach>=0.4): cas2 = unit.mps_kt(200.)
        else: cas2 = unit.mps_kt(70.)
        return cas2


#===========================================================================================================
class Requirement(object):
    """
    Initialize top level aircraft requirements
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
        self.cost_range = self.__cost_mission_range__()

        self.n_pax_ref = n_pax_ref
        self.n_pax_front = self.__n_pax_front__()
        self.n_aisle = self.__n_aisle__()
        self.m_pax_nominal = self.__m_pax_nominal__()
        self.m_pax_max = self.__m_pax_max__()

        self.take_off = Take_off_req(arrangement, self)
        self.approach = Approach_req(arrangement, self)
        self.oei = OEI_req(arrangement, self)
        self.vz_mcl = Vz_mcl_req(arrangement, self)
        self.vz_mcr = Vz_mcr_req(arrangement, self)
        self.ttc = TTC_req(arrangement, self)

    def __n_pax_front__(self):
        if  (self.n_pax_ref<=8):   n_pax_front = 2
        elif(self.n_pax_ref<=16):  n_pax_front = 3
        elif(self.n_pax_ref<=70):  n_pax_front = 4
        elif(self.n_pax_ref<=120): n_pax_front = 5
        elif(self.n_pax_ref<=225): n_pax_front = 6
        elif(self.n_pax_ref<=300): n_pax_front = 8
        elif(self.n_pax_ref<=375): n_pax_front = 9
        else:                      n_pax_front = 10
        return n_pax_front

    def __n_aisle__(self):
        if(self.n_pax_front <= 6): n_aisle = 1
        else:                      n_aisle = 2
        return n_aisle

    def __m_pax_nominal__(self):
        if(self.design_range <= unit.m_NM(500.)): m_pax_nominal = 85.
        elif(self.design_range <= unit.m_NM(1500.)): m_pax_nominal = 95.
        elif(self.design_range <= unit.m_NM(3500.)): m_pax_nominal = 100.
        elif(self.design_range <= unit.m_NM(5500.)): m_pax_nominal = 105.
        else: m_pax_nominal = 110.
        return m_pax_nominal

    def __m_pax_max__(self):
        if(self.design_range <= unit.m_NM(500.)): m_pax_max = 95.
        elif(self.design_range <= unit.m_NM(1500.)): m_pax_max = 105.
        elif(self.design_range <= unit.m_NM(3500.)): m_pax_max = 120.
        elif(self.design_range <= unit.m_NM(5500.)): m_pax_max = 135.
        else: m_pax_max = 150.
        return m_pax_max

    def __cost_mission_range__(self):
        if(self.design_range < unit.m_NM(400.)): cost_mission_range = unit.m_NM(100.)
        elif(self.design_range < unit.m_NM(1000.)): cost_mission_range = unit.m_NM(200.)
        elif(self.design_range < unit.m_NM(2500.)): cost_mission_range = unit.m_NM(400.)
        elif(self.design_range < unit.m_NM(4500.)): cost_mission_range = unit.m_NM(800.)
        elif(self.design_range < unit.m_NM(6500.)): cost_mission_range = unit.m_NM(2000.)
        else: cost_mission_range = unit.m_NM(4000.)
        return cost_mission_range
