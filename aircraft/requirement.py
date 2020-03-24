#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin

"""

import unit
import earth

#===========================================================================================================
class Requirement(object):
    """
    Initialize top level aircraft requirements
    """
    def __init__(self,n_pax_ref = 150.,
                      design_range = unit.m_NM(3000.),
                      cruise_mach = 0.78,
                      cruise_altp = unit.m_ft(35000.),
                      arrangement = None):

        self.design_range = design_range
        self.cruise_mach = cruise_mach
        self.cruise_altp = cruise_altp

        self.n_pax_ref = n_pax_ref
        self.n_pax_front = self.__n_pax_front__()
        self.n_aisle = self.__n_aisle__()
        self.m_pax_nominal = self.__m_pax_nominal__()
        self.m_pax_max = self.__m_pax_max__()

        self.tofl = [
                     {"disa": 15.,
                      "altp": 0.,
                      "kvs1g": 1.13,
                      "seg2_min_path": self.__seg2_min_path__(arrangement),
                      "tofl": self.__tofl__()}
                     ]

        self.approach = [
                         {"disa": 15.,
                          "altp": 0.,
                          "kvs1g": 1.23,
                          "speed": self.__app_speed__()}
                         ]

        self.oei = [
                    {"disa": 15.,
                     "altp": unit.m_ft(11000.),
                     "min_path": self.__oei_min_path__(arrangement)}
                    ]

        self.vz_mcl = [
                       {"disa": 15.,
                        "altp": self.__top_of_climb__(arrangement),
                        "vz": unit.mps_ftpmin(300.)}
                       ]

        self.vz_mcr = [
                       {"disa": 15.,
                        "altp": self.__top_of_climb__(arrangement),
                        "vz": unit.mps_ftpmin(0.)}
                       ]

        self.ttc = [
                    {"disa": 15.,
                     "altp": self.__top_of_climb__(arrangement),
                     "cas1": self.__ttc_cas1__(),
                     "cas2": self.__ttc_cas2__()}
                    ]


    #-----------------------------------------------------------------------------------------------------------
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

    #-----------------------------------------------------------------------------------------------------------
    def __n_aisle__(self):
        if(self.n_pax_front <= 6): n_aisle = 1
        else:                      n_aisle = 2
        return n_aisle

    #-----------------------------------------------------------------------------------------------------------
    def __m_pax_nominal__(self):
        if(self.design_range <= unit.m_NM(500.)):
            m_pax_nominal = 85.
        elif(self.design_range <= unit.m_NM(1500.)):
            m_pax_nominal = 95.
        elif(self.design_range <= unit.m_NM(3500.)):
            m_pax_nominal = 100.
        elif(self.design_range <= unit.m_NM(5500.)):
            m_pax_nominal = 105.
        else:
            m_pax_nominal = 110.
        return m_pax_nominal

    #-----------------------------------------------------------------------------------------------------------
    def __m_pax_max__(self):
        if(self.design_range <= unit.m_NM(500.)):
            m_pax_max = 95.
        elif(self.design_range <= unit.m_NM(1500.)):
            m_pax_max = 105.
        elif(self.design_range <= unit.m_NM(3500.)):
            m_pax_max = 120.
        elif(self.design_range <= unit.m_NM(5500.)):
            m_pax_max = 135.
        else:
            m_pax_max = 150.
        return m_pax_max

    #-----------------------------------------------------------------------------------------------------------
    def __tofl__(self):
        if(self.design_range <= unit.m_NM(1500.)):
            req_tofl = 1500.
        elif(self.design_range <= unit.m_NM(3500.)):
            req_tofl = 2000.
        elif(self.design_range <= unit.m_NM(5500.)):
            req_tofl = 2500.
        else:
            req_tofl = 3000.
        return req_tofl


    #-----------------------------------------------------------------------------------------------------------
    def __seg2_min_path__(self,arrangement):
        """
        Regulatory min climb path versus number of engine
        """
        if(arrangement.number_of_engine == "twin"):
            seg2_min_path = 0.024
        elif(arrangement.number_of_engine == "tri"):
            seg2_min_path = 0.027
        elif(arrangement.number_of_engine >= "quadri"):
            seg2_min_path = 0.030
        return seg2_min_path

    #-----------------------------------------------------------------------------------------------------------
    def __app_speed__(self):
        if (self.n_pax_ref<=100):
            req_app_speed = unit.mps_kt(120.)
        elif (self.n_pax_ref<=200):
            req_app_speed = unit.mps_kt(137.)
        else:
            req_app_speed = unit.mps_kt(140.)
        return req_app_speed

    #-----------------------------------------------------------------------------------------------------------
    def __oei_min_path__(self,arrangement):
        """
        Regulatory min climb path depending on the number of engine
        """
        if(arrangement.number_of_engine == "twin"):
            oei_min_path = 0.011
        elif(arrangement.number_of_engine == "tri"):
            oei_min_path = 0.013
        elif(arrangement.number_of_engine >= "quadri"):
            oei_min_path = 0.016
        return oei_min_path

    #-----------------------------------------------------------------------------------------------------------
    def __top_of_climb__(self,arrangement):
        if (arrangement.power_architecture=="tf"):
            altp = unit.m_ft(31000.)
        elif (arrangement.power_architecture=="tp"):
            altp = unit.m_ft(16000.)
        elif (arrangement.power_architecture=="pte1"):
            altp = unit.m_ft(31000.)
        elif (arrangement.power_architecture=="ef1"):
            altp = unit.m_ft(21000.)
        elif (arrangement.power_architecture=="ep1"):
            altp = unit.m_ft(16000.)
        else:
            raise Exception("propulsion.architecture index is out of range")
        top_of_climb = min(altp, self.cruise_altp-unit.m_ft(4000.))
        return top_of_climb


    #-----------------------------------------------------------------------------------------------------------
    def __ttc_cas1__(self):
        if (self.cruise_mach>=0.6):
            cas1 = unit.mps_kt(250.)
        elif (self.cruise_mach>=0.4):
            cas1 = unit.mps_kt(180.)
        else:
            cas1 = unit.mps_kt(70.)
        return cas1

    #-----------------------------------------------------------------------------------------------------------
    def __ttc_cas2__(self):
        if (self.cruise_mach>=0.6):
            cas2 = unit.mps_kt(300.)
        elif (self.cruise_mach>=0.4):
            cas2 = unit.mps_kt(200.)
        else:
            cas2 = unit.mps_kt(70.)
        return cas2


