#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np


def Pam3pkg_barLpkg(barLpkg): return barLpkg*1.e2  # Translate bar.L/kg into Pa.m3/kg
def barLpkg_Pam3pkg(Pam3pkg): return Pam3pkg/1.e2  # Translate Pa.m3/kg into bar.L/kg

def Pa_bar(bar): return bar*1.e5   # Translate bar into Pascal
def bar_Pa(Pa): return Pa/1.e5   # Translate Pascal into bar

class generic_fuel_tank(object):

    def __init__(self):

        # function 1 : Structural packaging and shielding
        self.structure_shell_surface_mass = 15.  # kg/m2
        self.structure_shell_thickness = 0.05    # m

        # function 2 : Pressure containment
        self.min_pressure_shell_efficiency = Pam3pkg_barLpkg(250.)  # bar.L/kg,  minimum pressurized tank efficiency
        self.max_pressure_shell_efficiency = Pam3pkg_barLpkg(650.)  # bar.L/kg,  maximum pressurized tank efficiency
        self.pressure_shell_density = 2000.                         # kg/m3, pressure material density

        # function 3 : Thermal insulation
        self.insulation_steel_density = 8000.    # kg/m3, inox
        self.insulation_sheet_tickness = 0.0013  # m, thickness of thermal walls
        self.insulation_gap_thickness = 0.02     # m, distance in between the the thermal walls

        self.external_area = None
        self.external_volume = None

        self.structure_internal_volume = None
        self.structure_shell_volume = None
        self.structure_shell_mass = None

        self.pressure_shell_volume = None
        self.pressure_shell_thickness = None
        self.pressure_shell_mass = None

        self.insulated_shell_mass = None
        self.insulation_shell_thickness = None
        self.insulated_shell_volume = None

        self.tank_mass = None
        self.fuel_volume = None
        self.fuel_mass = None

    def size_fuel_tank(self,location,fuel_type,over_pressure,fuel_density,length,width):
        # The overall shape of the tank is supposed to be a cylinder with two emispherical ends
        self.external_area = np.pi*width**2 + (length-width) * (np.pi*width)
        self.external_volume = (1./6.)*np.pi*width**3 + (length-width) * (0.25*np.pi*width**2)

        if location=="external":
            self.structure_internal_volume = (1./6.)*np.pi*(width-2.*self.structure_shell_thickness)**3 + (length-width) * (0.25*np.pi*(width-2.*self.structure_shell_thickness)**2)
            self.structure_shell_volume = self.external_volume - self.structure_internal_volume
            self.structure_shell_mass = self.structure_shell_surface_mass * self.external_area
        elif location=="internal":
            self.structure_shell_surface_mass = 0.  # kg/m2
            self.structure_shell_thickness = 0.     # m
            self.structure_internal_volume = self.external_volume
            self.structure_shell_volume = 0.
            self.structure_shell_mass = 0.
        else:
            raise Exception("Tank location is unknown")

        self.pressure_shell_area = np.pi*(width-2.*self.structure_shell_thickness)**2 + (length-width) * (np.pi*(width-2.*self.structure_shell_thickness))
        if fuel_type=="liquid_h2":
            pressure_shell_efficiency = self.min_pressure_shell_efficiency
        elif fuel_type=="compressed_h2":
            pressure_shell_efficiency = self.max_pressure_shell_efficiency
        else:
            pressure_shell_efficiency = 0.
        if over_pressure>0.:
            self.pressure_shell_volume = self.external_volume / (1.+pressure_shell_efficiency*self.pressure_shell_density/over_pressure)
            self.pressure_shell_thickness = self.pressure_shell_volume / self.pressure_shell_area
            self.pressure_shell_mass = self.pressure_shell_volume * self.pressure_shell_density
        else:
            self.pressure_shell_volume = 0.
            self.pressure_shell_thickness = 0.
            self.pressure_shell_mass = 0.

        thickness = self.structure_shell_thickness + self.pressure_shell_thickness
        self.insulation_shell_area = np.pi*(width-2.*thickness)**2 + (length-width) * (np.pi*(width-2.*thickness))    # insulated area
        if fuel_type=="liquid_h2":
            self.insulated_shell_mass = self.insulation_steel_density*self.insulation_sheet_tickness*self.insulation_shell_area
            self.insulation_shell_thickness = self.insulation_gap_thickness + 2.*self.insulation_sheet_tickness
            self.insulated_shell_volume = self.insulation_shell_area * self.insulation_shell_thickness
        else:
            self.insulated_shell_mass = 0.
            self.insulation_shell_thickness = 0.
            self.insulated_shell_volume = 0.

        self.tank_mass = self.structure_shell_mass + self.pressure_shell_mass + self.insulated_shell_mass
        self.fuel_volume = self.external_volume - self.structure_shell_volume - self.pressure_shell_volume - self.insulated_shell_volume
        self.fuel_mass = fuel_density * self.fuel_volume

        return

# Data
#-----------------------------------------------------------------------------------------------
lh2_density = 71.               # kg/m3, liquid H2 density
over_pressure = Pa_bar(5.)      # bar, fuel delta pressure

# Tank geometry, cylinder with emispherical ends
length = 12.    # m, tank length
width = 2.5     # m, tank diameter


tk = generic_fuel_tank()

# tk.size_fuel_tank("internal","kerosene",0.,43.,length,width)
tk.size_fuel_tank("external","liquid_h2",over_pressure,lh2_density,length,width)


# Print
#-----------------------------------------------------------------------------------------------
print("--------------------------------------------------------")
print("structure shell thickness = ", "%0.2f"%(tk.structure_shell_thickness*100.), " cm")
print("structure shell mass = ", "%0.1f"%tk.structure_shell_mass, " kg")
print("pressure shell thickness = ", "%0.2f"%(tk.pressure_shell_thickness*100.), " cm")
print("pressure shell mass = ", "%0.1f"%tk.pressure_shell_mass, " kg")
print("insulation shell thickness = ", "%0.2f"%(tk.insulation_shell_thickness*100.), " cm")
print("insulation shell mass = ", "%0.1f"%tk.insulated_shell_mass, " kg")
print("")
print("fuel mass = ", "%0.1f"%tk.fuel_mass, " kg")
print("tank mass = ", "%0.1f"%tk.tank_mass, " kg")
print("tank mass over LH2 mass = ", "%0.3f"%(tk.tank_mass/tk.fuel_mass), " kg/kg")



