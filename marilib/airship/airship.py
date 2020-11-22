#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 20 20:20:20 2020
@author: Thierry DRUOT
"""

import numpy as np
from scipy.special import ellipkinc, ellipeinc

import matplotlib.pyplot as plt

from marilib.utils import unit
from marilib.utils.math import lin_interp_1d



# ======================================================================================================
# Atmosphere
# ------------------------------------------------------------------------------------------------------
class Atmosphere(object):
    """Standard atmosphere
    """
    def __init__(self):
        pass

    def atmosphere(self, altp, disa=0.):
        """Ambiant data from pressure altitude from ground to 50 km according to Standard Atmosphere
        """
        g = 9.80665
        r = 287.053
        gam = 1.4

        Z = np.array([0., 11000., 20000., 32000., 47000., 50000.])
        dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])
        P = np.array([101325., 0., 0., 0., 0., 0.])
        T = np.array([288.15, 0., 0., 0., 0., 0.])

        if (Z[-1] < altp):
            raise Exception("atmosphere, altitude cannot exceed 50km")

        j = 0
        while (Z[1+j] <= altp):
            T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
            if (0. < np.abs(dtodz[j])):
                P[j+1] = P[j]*(1. + (dtodz[j]/T[j]) * (Z[j+1]-Z[j]))**(-g/(r*dtodz[j]))
            else:
                P[j+1] = P[j]*np.exp(-(g/r)*((Z[j+1]-Z[j])/T[j]))
            j = j + 1

        if (0. < np.abs(dtodz[j])):
            pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(r*dtodz[j]))
        else:
            pamb = P[j]*np.exp(-(g/r)*((altp-Z[j])/T[j]))
        tamb = T[j] + dtodz[j]*(altp-Z[j]) + disa
        return pamb, tamb, g

    def gas_data(self, gas="air"):
        """Gas data for a single gas
        """
        r = {"air" : 287.053 ,
             "helium" : 2077. ,
             "hydrogen" : 4124.
             }.get(gas, "Erreur: type of gas is unknown")

        gam = {"air" : 1.40 ,
               "helium" : 1.66 ,
               "hydrogen" : 1.41
               }.get(gas, "Erreur: type of gas is unknown")

        cv = r/(gam-1.)
        cp = gam*cv
        return r,gam,cp,cv

    def gas_density(self, pamb,tamb, gas="air"):
        """Ideal gas density
        """
        r,gam,Cp,Cv = self.gas_data(gas)
        rho = pamb / ( r * tamb )
        return rho

    def sound_speed(self, tamb):
        """Sound speed for ideal gas
        """
        r,gam,Cp,Cv = self.gas_data()
        vsnd = np.sqrt( gam * r * tamb )
        return vsnd

    def air_viscosity(self, tamb):
        """Mixed gas dynamic viscosity, Sutherland's formula
        WARNING : result will not be accurate if gas is mixing components of too different molecular weights
        """
        mu0,T0,S = 1.715e-5, 273.15, 110.4
        mu = (mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
        return mu

    def reynolds_number(self, pamb,tamb,tas):
        """Reynolds number based on Sutherland viscosity model
        """
        rho = self.gas_density(pamb,tamb)
        mu = self.air_viscosity(tamb)
        re = rho*tas/mu
        return re

    def fuel_density(self, fuel_type, press=101325.):
        """Reference fuel density
        """
        if (fuel_type=="kerosene"): return 803. # Kerosene : between 775-840 kg/m3
        elif (fuel_type=="liquid_h2"): return 70.8 # Liquid hydrogene
        elif (fuel_type=="Compressed_h2"):
            p = press*1.e-5
            return (-3.11480362e-05*p + 7.82320891e-02)*p + 1.03207822e-01 # Compressed hydrogen at 293.15 K
        elif (fuel_type=="methane"): return 422.6 # Liquid methane
        elif (fuel_type=="battery"): return 2800. # Lithium-ion
        else: raise Exception("fuel_type key is unknown")

    def fuel_heat(self, fuel_type):
        """Reference fuel lower heating value or battery energy density
        """
        if (fuel_type=="kerosene"): return 43.1e6 # J/kg, kerosene
        elif (fuel_type=="liquid_h2"): return 121.0e6 # J/kg, liquid hydrogene
        elif (fuel_type=="Compressed_h2"): return 140.0e6 # J/kg, compressed hydrogene
        elif (fuel_type=="methane"): return 50.3e6 # J/kg, Liquid methane
        elif (fuel_type=="battery"): return unit.J_Wh(200.) # J/kg, State of the art for lithium-ion
        else: raise Exception("fuel_type index is out of range")

    def stoichiometry(self, oxydizer,fuel):
        if oxydizer=="air":
            if fuel=="hydrogen": return 34.5
            else: raise Exception("Fuel type is unknown")
        else: raise Exception("Oxydizer type is unknown")


# ======================================================================================================
# Airship
# ------------------------------------------------------------------------------------------------------
class Airship(object):
    """Airship object
    """
    def __init__(self, atm, payload=10000., range=unit.m_NM(1000.), altp=unit.m_ft(10000.), disa=0., speed=unit.mps_kmph(100.)):
        self.atm = atm

        self.payload = payload      # Design mission payload
        self.range = range          # Design mission range
        self.cruise_altp = altp     # Reference cruise altitude
        self.cruise_disa = disa     # Reference standard temperature shift in cruise
        self.cruise_speed = speed   # Cruise speed

        self.length_o_width_ratio = 3.  # Length over width ratio
        self.length_o_height_ratio = 4. # Length over height ratio, WARNING l/h MUST BE HIGHER THAN l/w

        self.length = None          # Length of the ellipsoide
        self.width = None           # Width of the ellipsoide
        self.height = None          # Height of the ellipsoide
        self.gross_volume = None    # Total internal gross volume
        self.gross_area = None      # External area

        self.gondola_gravimetric_index = 0.200  # kg/kg, Mass of structure over max payload mass
        self.envelop_gravimetric_index = 0.200  # kg/m3, Mass of structure over total gross volume
        self.envelop_volumetric_index = 0.020   # m3/m3, Volume of structure over total gross volume
        self.buoyancy_reserve = 0.05            # m3/m3, Remaining air ballast volume over He volume at cruise altitude

        self.he_max_volume = None   # Max He volume
        self.he_max_mass = None     # Max He mass
        self.air_max_volume = None  # Max air volume in the ballasts

        self.n_engine = 4.                  # Number of engines
        self.engine_power = 1.e6            # Engine power
        self.h2_mass = 500.                 # Mass of liquid hydrogen stored in the cryogenic tank
        self.required_power = None          # Total required power
        self.fuel_cell_ref_power = None     # Fuel cell design power
        self.compressor_ref_power = None    # Compressor design power
        self.cooling_ref_power = None       # Cooling system design power
        self.heat_ref_power = None          # Dissipated heat power at design point

        self.nacelle_propulsive_efficiency = 0.82   # Thrust*TAS / shaft_power, propeller efficiency
        self.nacelle_gravimetric_index = 10.e3      # W/kg, Power density of electric motors
        self.h2_tank_gravimetric_index = 0.3        # kgH2/(kgH2+Tank), Tank gravimetric index

        self.motor_gravimetric_index = 15.e3        # W/kg, Power density of electric motors
        self.motor_efficiency = 0.95                # Electric motors efficiency

        self.inverter_gravimetric_index = 25.e3     # W/kg, Power density of inverters
        self.inverter_efficiency = 0.995            # Inverter efficiency

        self.wiring_gravimetric_index = 20.e3       # W/kg, Power density of wiring
        self.wiring_efficiency = 0.995              # Wiring efficiency

        self.fuel_cell_gravimetric_index = 2.e3     # W/kg, Power density of fuel cell stacks
        self.fuel_cell_efficiency = 0.50            # Fuel cell efficiency

        self.compressor_gravimetric_index = 1.e3    # W/kg, Power density of the air compressor
        self.compressor_over_pressure = 1.e5        # Pa, Compressor over pressure
        self.compressor_efficiency = 0.80           # Air compressor efficiency

        self.cooling_gravimetric_index = 5.e3       # W/kg, Dissipated power over cooling system mass
        self.cooling_power_index = 0.005            # W/W, Required cooling system power over dissipated power

        self.envelop_mass = None        # Mass of the envelop
        self.gondola_mass = None        # Mass of the gondola
        self.tank_mass                  # Cryogenic tank mass
        self.engine_mass = None         # Mass of the engines
        self.nacelle_mass = None        # Nacelle and mountings mass
        self.power_system_mass = None   # Mass of power system without motors and fuel cell
        self.fuel_cell_mass = None      # Mass of the fuel cell stack
        self.owe = None                 # Design mission Operating Empty Weight
        self.mtow = None                # Design mission Maximum Take Off Weight

        self.reference_area = None  # Aerodynamic reference area
        self.form_factor = 1.05     # Form factor for drag estimation

        self.fuel_mission = None    # Design mission fuel
        self.fuel_reserve = None    # Design mission reserve fuel
        self.kr = 0.05              # fraction of mission fuel for reserve

    def eval_design(self, length):
        """Compute geometrical datasc
        """
        if self.length_o_width_ratio > self.length_o_height_ratio:
            raise Exception("length_o_width must be lower than length_o_height")

        self.length = length
        self.width = self.length / self.length_o_width_ratio
        self.height = self.length / self.length_o_height_ratio

        self.gross_volume = (4./3.)*np.pi*self.length*self.width*self.height

        self.max_he_volume = (1.-self.envelop_volumetric_index) * self.gross_volume

        # Max He volume is computed at cruise altitude and high temperature
        pamb,tamb,g = atm.atmosphere(self.cruise_altp, 25.)
        self.he_max_mass = self.max_he_volume * atm.gas_density(pamb,tamb, gas="helium")

        # Max air volume is computed at sea level and low temperature
        pamb0,tamb0,g = atm.atmosphere(0., -35.)
        he_min_volume = self.he_max_mass / atm.gas_density(pamb0,tamb0, gas="helium")
        self.air_max_volume = self.max_he_volume - he_min_volume

        self.envelop_mass = self.gross_volume * self.envelop_gravimetric_index
        self.gondola_mass = self.payload * self.gondola_gravimetric_index

        self.required_power =    self.n_engine * self.engine_power \
                              / (self.motor_efficiency * self.inverter_efficiency * self.wiring_efficiency)

        data_dict = self.eval_fuel_cell_power(self.required_power,pamb,tamb)

        self.fuel_cell_ref_power = data_dict["fuel_cell_power"]
        self.compressor_ref_power = data_dict["compressor_power"]
        self.cooling_ref_power = data_dict["cooling_power"]
        self.heat_ref_power = data_dict["heat_power"]

        self.fuel_cell_mass = self.fuel_cell_ref_power / self.fuel_cell_gravimetric_index

        compressor_mass = self.compressor_ref_power / self.compressor_gravimetric_index
        cooling_mass = self.heat_ref_power / self.cooling_gravimetric_index
        wiring_mass = self.fuel_cell_ref_power / self.wiring_gravimetric_index
        self.power_system_mass = compressor_mass + cooling_mass + wiring_mass

        self.motor_mass = self.n_engine * self.engine_power / self.motor_gravimetric_index
        self.nacelle_mass = self.n_engine * self.engine_power / self.nacelle_gravimetric_index

        self.tank_mass = self.h2_mass * (1.-self.h2_tank_gravimetric_index)/self.h2_tank_gravimetric_index

        self.owe =   self.envelop_mass \
                   + self.gondola_mass \
                   + self.tank_mass \
                   + self.fuel_cell_mass \
                   + self.power_system_mass \
                   + self.nacelle_mass \
                   + self.motor_mass

        a, b, c = self.length, self.width, self.height
        cos_phi = c/a
        phi = np.arccos(cos_phi)
        sin_phi = np.sin(phi)
        k2 = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
        F = ellipkinc(phi, k2)
        E = ellipeinc(phi, k2)

        self.gross_area = 2.*np.pi*c**2 + ((2.*np.pi*a*b)/sin_phi) * (E*sin_phi**2 + F*cos_phi**2)
        self.reference_area = np.pi * a * b
        self.form_factor = 1.05

    def eval_design_constraints(self):
        """Evaluate the 3 design constraints that applies on the airship
        """








    def fuel_flow(self, pamb, tamb, tas, thrust):
        shaft_power = thrust*tas / self.nacelle_propulsive_efficiency
        req_power = shaft_power / self.motor_efficiency / self.inverter_efficiency / self.wiring_efficiency
        data_dict = self.eval_fuel_cell_power(req_power,pamb,tamb)
        return data_dict["fuel_flow"]

    def drag(self, pamb, tamb, tas):
        re = self.atm.reynolds_number(pamb, tamb, tas)
        vsnd = self.atm.sound_speed(tamb)
        mach = tas/vsnd
        fac = ( 1. + 0.126*mach**2 )
        nwa, ael, frm = self.gross_area, self.length, self.form_factor
        scxf = frm*((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) * nwa
        rho = self.atm.gas_density(pamb,tamb)
        drag = 0.5 * rho * tas**2 * scxf
        return drag

    def buoyancy(self, he_mass,pamb,tamb):
        """Compute the buoyancy force in given conditions
        """
        g = 9.80665
        rho_he = self.atm.gas_density(pamb,tamb,gas="helium")
        rho_air = self.atm.gas_density(pamb,tamb,gas="air")
        he_volume = rho_he / he_mass
        air_mass = rho_air * he_volume
        force = (air_mass - he_mass)*g
        return force

    def eval_fuel_cell_power(self, required_power,pamb,tamb):
        """Compute the power delivered by fuel cell stack according to required power and ambiant conditions
        """
        r,gam,Cp,Cv = self.atm.gas_data()

        fuel_heat = self.atm.fuel_heat("liquid_h2")

        # air_mass_flow = fuel_cell_power * relative_air_mass_flow
        st_mass_ratio = self.atm.stoichiometry("air","hydrogen")
        relative_fuel_flow = (1./self.fuel_cell_efficiency) / fuel_heat
        relative_air_mass_flow = relative_fuel_flow * st_mass_ratio
        relative_compressor_power = (1./self.compressor_efficiency)*(relative_air_mass_flow*Cv)*tamb*(((pamb+self.compressor_over_pressure)/pamb)**((gam-1.)/gam)-1.)

        # heat_power = fuel_cell_power * relative_heat_power
        relative_heat_power = (1.-self.fuel_cell_efficiency)/self.fuel_cell_efficiency
        relative_cooling_power = relative_heat_power*self.cooling_power_index

        fuel_cell_power = required_power / (1. - relative_compressor_power - relative_cooling_power)
        fuel_flow = fuel_cell_power * relative_fuel_flow

        compressor_power = fuel_cell_power * relative_compressor_power
        heat_power = fuel_cell_power * relative_heat_power
        cooling_power = heat_power * self.cooling_power_index

        return {"fuel_cell_power":fuel_cell_power,
                "compressor_power":compressor_power,
                "cooling_power":cooling_power,
                "heat_power":heat_power,
                "fuel_flow":fuel_flow}






atm = Atmosphere()

asp = Airship(atm)

asp.eval_design(50.)

