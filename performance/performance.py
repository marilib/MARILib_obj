#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry

"""




#--------------------------------------------------------------------------------------------------------------------------------
class Performance(object):
    """
    Logical aircraft description
    """
    def __init__(self):
        """
        Data structure, only one sub-level allowed
        """

        self.low_speed = None
        self.high_speed = None
        self.max_payload_mission = None
        self.nominal_mission = None
        self.max_fuel_mission = None
        self.zero_payload_mission = None
        self.cost_mission = None
        self.toy_mission = None


