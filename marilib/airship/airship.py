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
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

from marilib.utils import unit



# ======================================================================================================
# Atmosphere
# ------------------------------------------------------------------------------------------------------
class PhysicalData(object):
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
# Propulsion system
# ------------------------------------------------------------------------------------------------------
class Propulsion(object):
    """Propulsion object
    """
    def __init__(self, phd, ne):
        self.phd = phd

        self.n_engine = ne                  # Number of engines
        self.engine_power = None            # Engine shaft power
        self.total_ref_max_power = None     # Rrequired total ref power

        self.nacelle_propulsive_efficiency = 0.80   # Thrust*TAS / shaft_power, propeller efficiency
        self.nacelle_gravimetric_index = 3.e3       # W/kg, Power density of electric motors

        self.motor_gravimetric_index = 5.e3         # W/kg, Power density of electric motors
        self.motor_efficiency = 0.95                # Electric motors efficiency

        self.inverter_gravimetric_index = 25.e3     # W/kg, Power density of inverters
        self.inverter_efficiency = 0.995            # Inverter efficiency

        self.wiring_gravimetric_index = 20.e3       # W/kg, Power density of wiring
        self.wiring_efficiency = 0.995              # Wiring efficiency

        self.motor_mass = None          # Mass of the engines
        self.nacelle_mass = None        # Mass of the nacelles and mountings

    def get_max_power(self):
        return self.total_ref_max_power

    def req_power(self, tas, thrust):
        shaft_power = thrust*tas / self.nacelle_propulsive_efficiency
        req_power = shaft_power / self.motor_efficiency / self.inverter_efficiency / self.wiring_efficiency
        return req_power

    def design(self, power):
        self.engine_power = power
        self.total_ref_max_power = 1.10 * self.n_engine * self.engine_power \
                                                        / self.motor_efficiency \
                                                        / self.inverter_efficiency \
                                                        / self.wiring_efficiency
        self.motor_mass = self.n_engine * self.engine_power / self.motor_gravimetric_index
        self.nacelle_mass = self.n_engine * self.engine_power / self.nacelle_gravimetric_index

    def get_volume(self):
        return 0.

    def get_mass(self):
        return self.motor_mass + self.nacelle_mass

    def print(self):
        print("")
        print("Propulsion system")
        print("-------------------------------------------------------------------")
        print("Number of engine = ", self.n_engine)
        print("One engine power = ", "%.0f"%unit.kW_W(self.engine_power), " kW")
        print("All engine max power = ", "%.0f"%unit.kW_W(self.total_ref_max_power), " kW")
        print("")
        print("Total engine mass = ", "%.0f"%self.motor_mass, " kg")
        print("Total nacelle mass = ", "%.0f"%self.nacelle_mass, " kg")


# ======================================================================================================
# Power system
# ------------------------------------------------------------------------------------------------------
class Power(object):
    """Power object
    """
    def __init__(self, phd):
        self.phd = phd

        self.output_power = None          # Total required power
        self.fuel_cell_ref_power = None     # Fuel cell design power
        self.compressor_ref_power = None    # Compressor design power
        self.cooling_ref_power = None       # Cooling system design power
        self.heat_ref_power = None          # Dissipated heat power at design point

        self.fuel_cell_gravimetric_index = 2.e3     # W/kg, Power density of fuel cell stacks
        self.fuel_cell_efficiency = 0.50            # Fuel cell efficiency

        self.compressor_gravimetric_index = 1.e3    # W/kg, Power density of the air compressor
        self.compressor_over_pressure = 1.e5        # Pa, Compressor over pressure
        self.compressor_efficiency = 0.80           # Air compressor efficiency

        self.cooling_gravimetric_index = 5.e3       # W/kg, Dissipated power over cooling system mass
        self.cooling_power_index = 0.005            # W/W, Required cooling system power over dissipated power

        self.total_volumetric_index = 500.e3        # W/m3, Total power density of the power system

        self.fuel_cell_mass = None          # Mass of the fuel cell stack
        self.compressor_mass = None         # Mass of the air compressor
        self.cooling_mass = None            # Mass of the cooling system

        self.power_system_volume = None     # Volume of the power system
        self.power_system_mass = None       # Mass of the power system including fuel cell stacks

    def fuel_flow(self, pamb, tamb, tas, req_power):
        data_dict = self.fuel_cell_power(req_power,pamb,tamb)
        return data_dict["fuel_flow"]

    def fuel_cell_power(self, output_power,pamb,tamb):
        """Compute the power delivered by fuel cell stack according to required power and ambiant conditions
        """
        r,gam,Cp,Cv = self.phd.gas_data()

        fuel_heat = self.phd.fuel_heat("liquid_h2")

        # air_mass_flow = fuel_cell_power * relative_air_mass_flow
        st_mass_ratio = self.phd.stoichiometry("air","hydrogen")
        relative_fuel_flow = (1./self.fuel_cell_efficiency) / fuel_heat
        relative_air_mass_flow = relative_fuel_flow * st_mass_ratio
        relative_compressor_power = (1./self.compressor_efficiency)*(relative_air_mass_flow*Cv)*tamb*(((pamb+self.compressor_over_pressure)/pamb)**((gam-1.)/gam)-1.)

        # heat_power = fuel_cell_power * relative_heat_power
        relative_heat_power = (1.-self.fuel_cell_efficiency)/self.fuel_cell_efficiency
        relative_cooling_power = relative_heat_power*self.cooling_power_index

        fuel_cell_power = output_power / (1. - relative_compressor_power - relative_cooling_power)
        fuel_flow = fuel_cell_power * relative_fuel_flow

        compressor_power = fuel_cell_power * relative_compressor_power
        heat_power = fuel_cell_power * relative_heat_power
        cooling_power = heat_power * self.cooling_power_index

        return {"fuel_cell_power":fuel_cell_power,
                "compressor_power":compressor_power,
                "cooling_power":cooling_power,
                "heat_power":heat_power,
                "fuel_flow":fuel_flow}

    def design(self, pamb, tamb, power):
        self.output_power = power

        data_dict = self.fuel_cell_power(self.output_power,pamb,tamb)

        self.fuel_cell_ref_power = data_dict["fuel_cell_power"]
        self.compressor_ref_power = data_dict["compressor_power"]
        self.cooling_ref_power = data_dict["cooling_power"]
        self.heat_ref_power = data_dict["heat_power"]

        self.fuel_cell_mass = self.fuel_cell_ref_power / self.fuel_cell_gravimetric_index
        self.compressor_mass = self.compressor_ref_power / self.compressor_gravimetric_index
        self.cooling_mass = self.heat_ref_power / self.cooling_gravimetric_index
        self.power_system_mass = self.fuel_cell_mass + self.compressor_mass + self.cooling_mass
        self.power_system_volume = self.fuel_cell_ref_power / self.total_volumetric_index

    def get_volume(self):
        return self.power_system_volume

    def get_mass(self):
        return self.power_system_mass

    def print(self):
        print("")
        print("Power system")
        print("-------------------------------------------------------------------")
        print("Total output power = ", "%.0f"%unit.kW_W(self.output_power), " kW")
        print("Fuel cell reference power = ", "%.0f"%unit.kW_W(self.fuel_cell_ref_power), " kW")
        print("Compressor reference power = ", "%.0f"%unit.kW_W(self.compressor_ref_power), " kW")
        print("Cooler reference power = ", "%.0f"%unit.kW_W(self.cooling_ref_power), " kW")
        print("Dissipation reference power = ", "%.0f"%unit.kW_W(self.heat_ref_power), " kW")
        print("")
        print("Fuel cell mass = ", "%.0f"%self.fuel_cell_mass, " kg")
        print("Compressor mass = ", "%.0f"%self.compressor_mass, " kg")
        print("Cooler mass = ", "%.0f"%self.cooling_mass, " kg")
        print("")
        print("Total power system mass = ", "%.0f"%self.power_system_mass, " kg")
        print("Total power system volume = ", "%.0f"%self.power_system_volume, " m2")

# ======================================================================================================
# Tank system
# ------------------------------------------------------------------------------------------------------
class Tank(object):
    """Tank object
    """
    def __init__(self, phd):
        self.phd = phd

        self.h2_max_mass = None                     # Mass of liquid hydrogen stored in the cryogenic tank

        self.h2_tank_gravimetric_index = 0.2    # kgH2/(kgH2+kgTank), Tank gravimetric index
        self.h2_tank_volumetric_index = 50.     # kgH2/(m3H2+m3Tank), Tank volumetric index

        self.tank_volume = None                 # Cryogenic tank volume
        self.tank_mass = None                   # Cryogenic tank mass

    def design(self, h2_mass):
        self.h2_max_mass = h2_mass
        self.tank_volume = self.h2_max_mass / self.h2_tank_volumetric_index
        self.tank_mass = self.h2_max_mass * (1./self.h2_tank_gravimetric_index - 1.)

    def get_volume(self):
        return self.tank_volume

    def get_mass(self):
        return self.tank_mass

    def print(self):
        print("")
        print("Tank system")
        print("-------------------------------------------------------------------")
        print("Maximum capacity of LH2 = ", "%.0f"%self.h2_max_mass, " kg")
        print("")
        print("Total tank mass = ", "%.0f"%self.tank_mass, " kg")
        print("Total tank volume = ", "%.0f"%self.tank_volume, " m3")

# ======================================================================================================
# Airship
# ------------------------------------------------------------------------------------------------------
class Airship(object):
    """Airship object
    """
    def __init__(self, phd, ne=6, payload=10000., range=unit.m_NM(1000.), altp=unit.m_ft(10000.), disa=0., speed=unit.mps_kmph(100.)):
        self.phd = phd

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

        self.n_fin = 4              # Number of fins
        self.fin_area = None        # Area of one fin

        self.gondola_gravimetric_index = 0.200  # kg/kg, Mass of structure over max payload mass
        self.gondola_volumetric_index = 0.020   # m3/kg, Volume of structure over max payload mass
        self.envelop_gravimetric_index = 0.500  # kg/m2, Mass of structure over total gross area
        self.envelop_volumetric_index = 0.020   # m3/m3, Volume of structure over total gross volume
        self.buoyancy_reserve = 0.05            # m3/m3, Remaining air ballast volume over He volume at cruise altitude

        self.he_max_volume = None       # Max He volume
        self.he_max_mass = None         # Max He mass
        self.air_max_volume = None      # Max air volume in the ballasts

        self.envelop_mass = None        # Mass of the envelop
        self.gondola_mass = None        # Mass of the gondola
        self.owe = None                 # Design mission Operating Empty Weight
        self.mtow = None                # Design mission Maximum Take Off Weight

        self.reference_area = None      # Aerodynamic reference area
        self.envelop_form_factor = 1.05 # Envelop form factor for drag estimation
        self.fin_form_factor = 1.15     # Fin form factor for drag estimation

        self.fuel_factor = 0.15         # fraction of mission fuel for reserve

        # Propulsion system
        #---------------------------------------------------------------------------------------------------------------
        self.propulsion = Propulsion(phd, ne)

        # Power system
        #---------------------------------------------------------------------------------------------------------------
        self.power = Power(phd)

        # Tank system
        #---------------------------------------------------------------------------------------------------------------
        self.tank = Tank(phd)

        # Design the airship
        #---------------------------------------------------------------------------------------------------------------
        self.design()

    def drag_force(self, pamb, tamb, tas):
        re = self.phd.reynolds_number(pamb, tamb, tas)
        vsnd = self.phd.sound_speed(tamb)
        mach = tas/vsnd
        fac = ( 1. + 0.126*mach**2 )

        nwa, ael, frm = self.gross_area, self.length, self.envelop_form_factor
        scxf_env = frm*((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) * nwa

        nwa, ael, frm = self.fin_area*self.n_fin*2., np.sqrt(self.fin_area), self.fin_form_factor
        scxf_fin = frm*((0.455/fac)*(np.log(10)/np.log(re*ael))**2.58 ) * nwa

        rho = self.phd.gas_density(pamb,tamb)
        drag_force = 0.5 * rho * tas**2 * (scxf_env + scxf_fin)
        return drag_force

    def buoyancy_force(self, he_mass,pamb,tamb):
        """Compute the buoyancy force in given conditions
        """
        g = 9.80665
        rho_he = self.phd.gas_density(pamb,tamb,gas="helium")
        rho_air = self.phd.gas_density(pamb,tamb,gas="air")
        he_volume = he_mass / rho_he
        air_mass = rho_air * he_volume
        force = (air_mass - he_mass)*g
        return force

    def eval_design(self, length, power, h2_mass):
        """Compute geometrical datasc
        """
        if self.length_o_width_ratio > self.length_o_height_ratio:
            raise Exception("length_o_width must be lower than length_o_height")

        self.length = length                # Length of the ellipsoide
        self.width = self.length / self.length_o_width_ratio
        self.height = self.length / self.length_o_height_ratio

        self.fin_area = (0.80 * self.length) / self.n_fin

        self.gross_volume = (4./3.)*np.pi*self.length*self.width*self.height

        a, b, c = self.length, self.width, self.height
        cos_phi = c/a
        phi = np.arccos(cos_phi)
        sin_phi = np.sin(phi)
        k2 = (a**2 * (b**2 - c**2)) / (b**2 * (a**2 - c**2))
        F = ellipkinc(phi, k2)
        E = ellipeinc(phi, k2)

        self.gross_area = 2.*np.pi*c**2 + ((2.*np.pi*a*b)/sin_phi) * (E*sin_phi**2 + F*cos_phi**2)
        self.reference_area = np.pi * a * b

        # Propulsion system design
        self.propulsion.design(power)

        # Power system design
        output_power = self.propulsion.get_max_power()
        pamb,tamb,g = phd.atmosphere(0., 25.)
        self.power.design(pamb,tamb,output_power)

        # Tank system design
        self.tank.design(h2_mass)

        # max He volume corresponds to max internal volume minus structure volume, buoyancy reserve, payload volume, fuel tank volume
        self.he_max_volume =   (1. - self.envelop_volumetric_index - self.buoyancy_reserve) * self.gross_volume \
                             - self.gondola_volumetric_index * self.payload \
                             - self.power.get_volume() \
                             - self.tank.get_volume()

        # Max He volume is computed at cruise altitude and high temperature
        pamb1,tamb1,g = phd.atmosphere(self.cruise_altp, 25.)
        self.he_max_mass = self.he_max_volume * phd.gas_density(pamb1,tamb1, gas="helium")

        # Max air volume is computed at sea level and low temperature
        pamb0,tamb0,g = phd.atmosphere(0., -35.)
        he_min_volume = self.he_max_mass / phd.gas_density(pamb0,tamb0, gas="helium")
        self.air_max_volume = self.he_max_volume - he_min_volume

        self.envelop_mass = self.gross_area * self.envelop_gravimetric_index
        self.gondola_mass = self.payload * self.gondola_gravimetric_index

        self.owe =   self.envelop_mass \
                   + self.gondola_mass \
                   + self.propulsion.get_mass() \
                   + self.power.get_mass() \
                   + self.tank.get_mass()

        self.mtow = self.owe + self.payload + h2_mass

    def eval_design_constraints(self):
        """Evaluate the 3 design constraints that applies on the airship
        """
        # Power constraint
        pamb,tamb,g = phd.atmosphere(self.cruise_altp, self.cruise_disa)
        thrust = self.drag_force(pamb,tamb,self.cruise_speed) / self.propulsion.n_engine
        shaft_power = self.propulsion.req_power(self.cruise_speed, thrust)

        # Energy constraint
        req_power = self.propulsion.req_power(self.cruise_speed, thrust)
        total_req_power = req_power * self.propulsion.n_engine
        fuel_flow = self.power.fuel_flow(pamb, tamb, self.cruise_speed, total_req_power)
        time = self.range / self.cruise_speed
        fuel_mass = fuel_flow * time * (1.+self.fuel_factor)

        # Buoyancy constraint
        buoyancy = self.buoyancy_force(self.he_max_mass,pamb,tamb)
        mass = self.mtow

        return {"power": self.propulsion.engine_power - shaft_power,
                "energy": self.tank.h2_max_mass - fuel_mass,
                "buoyancy": buoyancy - mass*g}

    def design(self):
        """This method designs the airship according to TLARs
        """
        def fct(X):
            length, power, h2_mass = X
            self.eval_design(length, power, h2_mass)
            cst = self.eval_design_constraints()
            return [cst["power"], cst["energy"], cst["buoyancy"]]

        Xini = [50., 5.e4, 500.]

        out_dict = fsolve(fct, x0=Xini, args=(), full_output=True)
        if (out_dict[2]!=1): raise Exception("Convergence problem")
        length, power, h2_mass = altg = out_dict[0]

        self.eval_design(length, power, h2_mass)



    def print(self):
        """Print all airship data
        """
        print("")
        print("Airship")
        print("-------------------------------------------------------------------")
        print("Payload = ", "%.0f"%self.payload, " kg")
        print("Range = ", "%.0f"%unit.NM_m(self.range), " NM")
        print("Cruise altitude = ", "%.0f"%unit.ft_m(self.cruise_altp), " ft")
        print("Cruise disa = ", "%.0f"%self.cruise_disa, " K")
        print("Cruise speed = ", "%.0f"%unit.kmph_mps(self.cruise_speed), " km/h")
        print("Reserve fuel factor = ", "%.0f"%self.fuel_factor)
        print("")
        print("Envelop length = ", "%.0f"%self.length, " m")
        print("Envelop widdh = ", "%.0f"%self.width, " m")
        print("Envelop height = ", "%.0f"%self.height, " m")
        print("Envelop gross area = ", "%.0f"%self.gross_area, " m2")
        print("Envelop gross volume = ", "%.0f"%self.gross_volume, " m3")
        print("Reference area = ", "%.0f"%self.reference_area, " m2")
        print("")
        print("Number of fins = ", self.n_fin)
        print("One fin area = ", "%.0f"%self.fin_area, " m2")
        print("")
        print("Maximum volume of ballonets = ", "%.0f"%self.air_max_volume, " m3")
        print("Maximum volume of Helium = ", "%.0f"%self.he_max_volume, " m3")
        print("Maximum mass of Helium = ", "%.0f"%self.he_max_mass, " kg")
        print("")
        print("Total envelop mass = ", "%.0f"%self.envelop_mass, " kg")
        print("Total gondola mass = ", "%.0f"%self.gondola_mass, " kg")
        print("Operating weight empty (without He) = ", "%.0f"%self.owe, " kg")
        print("Maximum take off weight (without He) = ", "%.0f"%self.mtow, " kg")

        self.propulsion.print()
        self.power.print()
        self.tank.print()





phd = PhysicalData()

ne = 6                      # Number of engines
payload = 10000.            # Design payload
range = unit.m_NM(1000.)    # Design range

speed = unit.mps_kmph(100.) # Nominal cruise speed
altp = unit.m_ft(10000.)    # Nominal cruise altitude
disa = 0.                   # Nominal temperature shift

asp = Airship(phd, ne, payload, range, altp, disa, speed)

asp.print()

cst = asp.eval_design_constraints()

print("")
print("-------------------------------------------------------------------")
print("Power constraint = ", cst["power"], "Capability to sustain required cruise speed")
print("Energy constraint = ", cst["energy"], "Capability to fly the required range")
print("Buoyancy constraint = ", cst["buoyancy"], "Capability to reach required altitude")

