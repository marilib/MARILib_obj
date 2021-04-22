#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np
from scipy.optimize import fsolve

import unit

from physical_data import PhysicalData



class LH2Tank(object):
    """Tank for liquid hydrogen
    """
    def __init__(self, phd):
        self.phd = phd

        self.grav_index_ref = 0.15   # kgH2/(kgH2+kgTank), Tank gravimetric index
        self.vol_index_ref = 43.    # kgH2/(m3H2+m3Tank), Tank volumetric index
        self.h2_mass_ref = 5.       # LH2 mass for these indices
        self.gas_ratio = 0.05       # Fraction of internal volume filled with gas

        self.width = None               # Tank width
        self.length = None              # Tank length
        self.internal_volume = None     # Tank internal volume
        self.external_volume = None     # Tank external volume
        self.wall_thickness = None      # Tank wall thickness
        self.wall_density = None        # Tank wall thickness
        self.mass = None                # Tank mass

        self.liquid_h2_volume = None    # Liquid H2 volume
        self.h2_max_mass = None         # Liquid H2 mass stored in the tank

        self.gravimetric_index = None    # kgH2/(kgH2+kgTank), Tank gravimetric index
        self.volumetric_index = None     # kgH2/(m3H2+m3Tank), Tank volumetric index

    def design(self, width, length):
        self.width = width
        self.length = length

        wolr = self.width/self.length
        lh2_density = self.phd.fuel_density("liquid_h2")
        int_volume_ref = (self.h2_mass_ref/lh2_density)/(1.-self.gas_ratio)
        str_volume_ref = (self.h2_mass_ref/self.vol_index_ref) - int_volume_ref
        ext_volume_ref = int_volume_ref + str_volume_ref
        length_ref = (ext_volume_ref/((np.pi/4.)*(1.-wolr/3.)*wolr**2))**(1./3.)
        width_ref = length_ref * wolr
        tank_mass_ref = self.h2_mass_ref*((1./self.grav_index_ref)-1.)

        def tank_volume(length,width,thickness):
            return (np.pi/4.)*(length-width/3.-(4./3.)*thickness)*(width-2.*thickness)**2

        def fct(thickness):
            return int_volume_ref - tank_volume(length_ref,width_ref,thickness)

        output_dict = fsolve(fct, x0=0.01, args=(), full_output=True)
        if (output_dict[2]!=1): raise Exception("Convergence problem")
        self.wall_thickness = output_dict[0][0]

        # Density supposing a structure with skin frames and stiffeners and a constant thickness of the skin
        # self.wall_density = tank_mass_ref / str_volume_ref

        # Assumptions : reference shells have equal mass of skin versus frames and stiffeners
        #               inner shell and outer shell have the same surface mass
        #               when volume increases : outer shell is supposed to keep the same surface mass
        #                                       inner shell skin keep the same thickness but frames and stiffeners mass
        #                                       increases proportionnally to the diameter
        self.wall_density = (tank_mass_ref / str_volume_ref) * (1. + self.width/width_ref)/2.

        self.external_volume = tank_volume(self.length,self.width,0.)
        self.internal_volume = tank_volume(self.length,self.width,self.wall_thickness)

        self.h2_volume = (1.-self.gas_ratio)*self.internal_volume
        self.h2_max_mass = self.h2_volume*lh2_density

        self.mass = (self.external_volume - self.internal_volume)*self.wall_density

        self.gravimetric_index = self.h2_max_mass / (self.h2_max_mass + self.mass)
        self.volumetric_index = self.h2_max_mass / self.external_volume

    def print(self):
        print("")
        print("Tank system")
        print("-------------------------------------------------------------------")
        print("Maximum capacity of LH2 = ", "%.0f"%self.h2_max_mass, " kg")
        print("")
        print("Total tank mass = ", "%.0f"%self.mass, " kg")
        print("Total tank volume = ", "%.2f"%self.external_volume, " m3")
        print("Tank wall thickness = ", "%.3f"%self.wall_thickness, " m")
        print("")
        print("Gravimetric index = ", "%.3f"%self.gravimetric_index, " kg/kg")
        print("Volumetric index = ", "%.1f"%self.volumetric_index, " kg/m3")



if __name__ == "__main__":

    # Design time
    #-----------------------------------------------------------------
    phd = PhysicalData()    # Create phd object

    h2t = LH2Tank(phd)      # Create tank object

    width = 3.5             # Tank diameter
    length = 5.0            # Tank length

    h2t.design(width,length)    # Design the tank

    h2t.print()                 # Print tank data

