#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

import unit

from physical_data import PhysicalData



class LH2Tank(object):
    """Tank for liquid hydrogen
    """
    def __init__(self, phd):
        self.phd = phd

        self.h2_tank_gravimetric_index = 0.2    # kgH2/(kgH2+kgTank), Tank gravimetric index
        self.h2_tank_volumetric_index = 50.     # kgH2/(m3H2+m3Tank), Tank volumetric index

        self.h2_max_mass = None                 # Mass of liquid hydrogen stored in the cryogenic tank

        self.tank_volume = None                 # Cryogenic tank volume
        self.tank_mass = None                   # Cryogenic tank mass

    def design(self, h2_mass):
        self.h2_max_mass = h2_mass
        self.tank_volume = self.h2_max_mass / self.h2_tank_volumetric_index
        self.tank_mass = self.h2_max_mass * (1./self.h2_tank_gravimetric_index - 1.)

    def print(self):
        print("")
        print("Tank system")
        print("-------------------------------------------------------------------")
        print("Maximum capacity of LH2 = ", "%.0f"%self.h2_max_mass, " kg")
        print("")
        print("Total tank mass = ", "%.0f"%self.tank_mass, " kg")
        print("Total tank volume = ", "%.1f"%self.tank_volume, " m3")



if __name__ == "__main__":

    # Design time
    #-----------------------------------------------------------------
    phd = PhysicalData()    # Create phd object

    h2t = LH2Tank(phd)      # Create tank object

    h2_mass = 500.          # Required capacity

    h2t.design(h2_mass)     # Design the tank

    h2t.print()             # Print tank data

