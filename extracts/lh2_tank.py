#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

from marilib.utils import earth, unit



class generic_fuel_tank(object):

    def __init__(self):

        # function 1 : Structural packaging and shielding
        self.structure_shell_surface_mass = 15.  # kg/m2
        self.structure_shell_thickness = 0.10    # m

        # function 2 : Pressure containment
        self.min_pressure_shell_efficiency = unit.Pam3pkg_barLpkg(250.)  # bar.L/kg,  minimum pressurized tank efficiency
        self.max_pressure_shell_efficiency = unit.Pam3pkg_barLpkg(650.)  # bar.L/kg,  maximum pressurized tank efficiency
        self.pressure_shell_density = 2000.                              # kg/m3, pressure material density

        # function 3 : Thermal insulation
        self.insulation_shell_density = 240.     # kg/m3, total insulation volumetric density
        self.insulation_shell_thickness = 0.09   # m, thickness of thermal walls

        # Function 4 : fuel management
        self.fuel_management_density = 5.       # kg/m3, required system mass per m3 of LH2

        # Dewar insulation parameters
        self.dewar_ext_shell_thickness = 0.005   # m, Mean thickness of the external shell
        self.dewar_int_shell_thickness = 0.003   # m, Mean thickness of the internal shell
        self.dewar_inter_shell_gap = 0.08             # m, Mean gap between the shell
        self.dewar_material_density = 2700.     # kg/m3, Shell material density (2700. : aluminium)

        self.external_area = None
        self.external_volume = None

        self.structure_internal_volume = None
        self.structure_shell_volume = None
        self.structure_shell_mass = None

        self.pressure_shell_volume = None
        self.pressure_shell_thickness = None
        self.pressure_shell_mass = None

        self.insulated_shell_mass = None
        self.insulated_shell_volume = None

        self.tank_mass = None
        self.fuel_volume = None
        self.fuel_mass = None

        self.gravimetric_enrg_density = None
        self.volumetric_enrg_density = None
        self.volumetric_storage_density = None

    def dewar_insulation(self):
        # Compute thickness and overall density of a Dewar's insulation
        self.insulation_shell_thickness = self.dewar_ext_shell_thickness + self.dewar_int_shell_thickness + self.dewar_inter_shell_gap
        self.insulation_shell_density = self.dewar_material_density * (self.dewar_ext_shell_thickness + self.dewar_int_shell_thickness) / self.insulation_shell_thickness

    def size_fuel_tank(self,location,fuel_type,over_pressure,fuel_density,fuel_heat,length,width):
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
            self.insulated_shell_volume = self.insulation_shell_area * self.insulation_shell_thickness
            self.insulated_shell_mass = self.insulated_shell_volume * self.insulation_shell_density
        else:
            self.insulated_shell_volume = 0.
            self.insulated_shell_mass = 0.
            self.insulation_shell_thickness = 0.

        self.fuel_volume = self.external_volume - self.structure_shell_volume - self.pressure_shell_volume - self.insulated_shell_volume
        self.fuel_mass = fuel_density * self.fuel_volume

        # Basic fuel management system is already included within total aircraft system mass
        if fuel_type in ["liquid_h2","compressed_h2"]:
            self.specific_system_mass = self.fuel_management_density * self.fuel_volume
        else:
            self.specific_system_mass = 0.

        self.tank_mass = self.structure_shell_mass + self.pressure_shell_mass + self.insulated_shell_mass + self.specific_system_mass

        self.gravimetric_enrg_density = self.fuel_mass*fuel_heat / (self.fuel_mass+self.pressure_shell_mass+self.insulated_shell_mass+self.specific_system_mass)
        self.volumetric_enrg_density = self.fuel_mass*fuel_heat / (self.fuel_volume+self.pressure_shell_volume+self.insulated_shell_volume)
        self.volumetric_storage_density = self.fuel_mass / (self.fuel_volume+self.pressure_shell_volume+self.insulated_shell_volume)

        return

# Data
#-----------------------------------------------------------------------------------------------
lh2_density = 71.               # kg/m3, liquid H2 density
lh2_heat = 171.e6               # MJ/kg
over_pressure = unit.Pa_bar(5.) # bar, fuel delta pressure

# Tank geometry, cylinder with emispherical ends
length = 12.    # m, tank length
width = 2.5     # m, tank diameter


tk = generic_fuel_tank()

# tk.dewar_insulation()

# tk.size_fuel_tank("internal","kerosene",0.,803.,43000000.,length,width)
tk.size_fuel_tank("external","liquid_h2",over_pressure,lh2_density,lh2_heat,length,width)


# Print
#-----------------------------------------------------------------------------------------------
print("--------------------------------------------------------")
print("specific system mass = ", "%0.1f"%(tk.specific_system_mass), " kg")
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
print("")
print("LH2 energy density = ","%0.1f"%unit.kWh_J(lh2_heat), " kWh/kg")
print("Gravimetric energy density = ","%0.1f"%unit.kWh_J(tk.gravimetric_enrg_density), " kWh/kg")
print("Volumetric energy density = ","%0.2f"%(unit.kWh_J(tk.volumetric_enrg_density)*1.e-3), " kWh/L")
print("Volumetric storage density = ","%0.2f"%(tk.volumetric_storage_density), " g/L")



