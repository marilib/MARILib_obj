#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 31 10:36 2020
@author: Nicolas Peteilh
"""


# --------------------------------------------------------------------------------------------------------------------------------
class AircraftFactory(object):
    """
    Aircraft factory
    """

    def __init__(self, name=None, engine_type="TF", engine_number=2):
        self.name = name
        self.engine_type = engine_type
        self.engine_number = engine_number

    def create_aircraft(self):
        ac = Aircraft(engine_type=self.engine_type,
                      engine_number=self.engine_number)
        if self.engine_type=="TF":
            en = Turbofan(aircraft=ac)
        elif self.engine_type=="TP":
            en = Turboprop(aircraft=ac)
        ac.engine = en
        ac.total_thrust = ac.engine.compute_total_thrust()
        return ac


# --------------------------------------------------------------------------------------------------------------------------------
class Aircraft(object):
    """
    Assembling all aircraft data branches
    """

    def __init__(self, name=None, engine_type=None, engine_number=None):
        """
            Data structure branches, no ramification
        """
        self.name = name
        self.engine = None
        self.engine_type = engine_type
        self.engine_number = engine_number
        self.total_thrust = None

# --------------------------------------------------------------------------------------------------------------------------------
class Engine(object):
    """
    Assembling all aircraft data branches
    """

    def __init__(self, name=None, aircraft=None):
        """
            Data structure branches, no ramification
        """
        self.name = name
        self.aircraft = aircraft

    def compute_thrust(self):
        raise NotImplementedError

    def compute_total_thrust(self):
        return self.compute_thrust()*self.aircraft.engine_number


# --------------------------------------------------------------------------------------------------------------------------------
class Turbofan(Engine):
    """
    Assembling all aircraft data branches
    """

    def __init__(self, name=None, aircraft=None):
        """
             J'ai déjà le name dans la classe mère donc je ne le rajoute pas là
             Mais il faut peut-être ici rappeler __init__ de Engine pour ne pas l'écrase ? Si oui, on fait comment
        """
        super(Turbofan, self).__init__(name,aircraft)

    def compute_thrust(self):
        print(self.name)
        thrust = 9.0
        return thrust

# --------------------------------------------------------------------------------------------------------------------------------
class Turboprop(Engine):
    """
    Assembling all aircraft data branches
    """

    # def __init__(self):
    #     """
    #         J'ai déjà le name dans la classe mère donc je ne le rajoute pas là
    #         Mais il faut peut-être ici rappeler __init__ de Engine pour ne pas l'écrase ? Si oui, on fait comment
    #     """

    def compute_thrust(self):
        thrust = 5.0
        return thrust


if __name__ == "__main__":
    usine_d_avion = AircraftFactory(name="CD_factory",
                                     engine_number=3,
                                     engine_type="TF")
    avion = usine_d_avion.create_aircraft()
    print("avion.total_thrust = ", avion.total_thrust)

