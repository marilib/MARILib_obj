#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 20 20:20:20 2020
@author: Thierry DRUOT
"""

import numpy as np
from scipy import interpolate

from marilib.utils import unit


# ======================================================================================================
# Physical data
# ------------------------------------------------------------------------------------------------------
class PhysicalData(object):
    """Standard atmosphere
    """
    def __init__(self):
        pass

    def sea_level_data(self):
        p0 = 101325.    # Pascals
        t0 = 288.15    # Kelvins
        return p0,t0

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

    def lh2_latent_heat(self):
        boil_temp = 20.4
        latent_heat = unit.J_kJ(454.3)  # J/kg
        return latent_heat, boil_temp

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

    def total_temperature(self, tamb,mach):
        """Stagnation temperature
        """
        r,gam,Cp,Cv = self.gas_data()
        ttot = tamb*(1.+((gam-1.)/2.)*mach**2)
        return ttot

    def total_pressure(self, pamb,mach):
        """Stagnation pressure
        """
        r,gam,Cp,Cv = self.gas_data()
        ptot = pamb*(1+((gam-1.)/2.)*mach**2)**(gam/(gam-1.))
        return ptot

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

    def air_thermal_transfer_data(self, pamb,tamb,air_speed, x):
        """Thermal transfert factor for turbulent air flow
        """
        r,gam,cp,cv = self.gas_data()
        rho = self.gas_density(pamb,tamb)
        mu = self.air_viscosity(tamb)
        alpha = self.air_thermal_diffusivity()
        pr = mu / (alpha * rho)                         # Prandtl number
        re = rho * air_speed / mu                       # Reynolds number
        if (re*x)<1.e8 and 0.6<pr and pr<60.:
            nu = 0.0296 * (re*x)**(4/5) * pr**(1/3)     # Nusselt number
        else:
            print("Re = ", re*x, "  Pr = ", pr)
            raise Exception("Re or Pr are not in the valid domain")
        lbd = alpha * rho * cp      # Thermal conductivity
        h = lbd * nu / x
        return h, rho, cp, mu, pr, re*x, nu, lbd

    def air_thermal_diffusivity(self):
        """Thermal diffusivity of the air at 300 K
        """
        thermal_diffusivity = 20.e-6   # m2/s
        return thermal_diffusivity

    def fluid_thermal_transfer_data(self, temp,fluid_speed,tube_length, fluid="water"):
        """Thermal transfert factor for turbulent flow : W/m2/K
        """
        if fluid!="water":
            raise Exception("fluide type is not permitted")

        # Fluid thermal conductivity
        if temp<274. or 370.<temp:
            raise Exception("fluid temperature is out of range")
        t0 = 298.15
        c0 = 0.6065
        lbd = c0*(-1.48445 + 4.12292*(temp/t0) - 1.63866*(temp/t0)**2)      # Thermal conductivity

        # Nusselt number
        rho, cp, mu = self.fluid_data(temp, fluid=fluid)
        pr = mu * cp / lbd                                # Prandtl number
        re = (rho * fluid_speed / mu) * tube_length       # Reynolds number
        if 1.e5<re and re<1.e8 and 0.6<pr and pr<60.:
            nu = 0.0296 * re**(4/5) * pr**(1/3)   # Nusselt number
        else:
            print("fluid_speed = ", fluid_speed, "Re = ", re, "  Pr = ", pr)
            raise Exception("Re or Pr are not in the valid domain")

        # Thermal transfert factor
        h = lbd * nu / tube_length
        return h, rho, cp, mu, pr, re, nu, lbd

    def fluid_data(self, temp, fluid="water"):
        if fluid!="water":
            raise Exception("fluide type is not permitted")
        t0 = 273.15
        temp_list = [275.15, 276.15, 277.15, 278.15, 279.15, 280.15, 281.15, 282.15, 283.15, 284.15, 285.15, 286.15, 287.15,
                     288.15, 289.15, 290.15, 291.15, 292.15, 293.15, 294.15, 295.15, 296.15, 297.15, 298.15, 299.15, 300.15,
                     301.15, 302.15, 303.15, 304.15, 305.15, 306.15, 307.15, 308.15, 309.15, 310.15, 311.15, 312.15, 313.15,
                     318.15, 323.15, 328.15, 333.15, 338.15, 343.15, 348.15, 353.15]
        mu_list =   [1.6735, 1.6190, 1.5673, 1.5182, 1.4715, 1.4271, 1.3847, 1.3444, 1.3059, 1.2692, 1.2340, 1.2005, 1.1683,
                     1.1375, 1.1081, 1.0798, 1.0526, 1.0266, 1.0016, 0.9775, 0.9544, 0.9321, 0.9107, 0.8900, 0.8701, 0.8509,
                     0.8324, 0.8145, 0.7972, 0.7805, 0.7644, 0.7488, 0.7337, 0.7191, 0.7050, 0.6913, 0.6780, 0.6652, 0.6527,
                     0.5958, 0.5465, 0.5036, 0.4660, 0.4329, 0.4035, 0.3774, 0.3540]
        mu_f = interpolate.interp1d(temp_list,mu_list,kind="cubic")
        rho_f = 1000.   # Water
        cp_f = 4200.    # Water
        return rho_f, cp_f, mu_f(temp)*1.e-3

    def fuel_density(self, fuel_type, press=101325.):
        """Reference fuel density
        """
        if (fuel_type=="kerosene"): return 803. # Kerosene : between 775-840 kg/m3
        elif (fuel_type=="gasoline"): return 800. # Gasoline : between 775-840 kg/m3
        elif (fuel_type=="liquid_h2"): return 70.8 # Liquid hydrogene
        elif (fuel_type=="compressed_h2"):
            p = press*1.e-5
            return (-3.11480362e-05*p + 7.82320891e-02)*p + 1.03207822e-01 # Compressed hydrogen at 293.15 K
        elif (fuel_type=="methane"): return 422.6 # Liquid methane
        elif (fuel_type=="battery"): return 2800. # Lithium-ion
        else: raise Exception("fuel_type key is unknown")

    def fuel_heat(self, fuel_type):
        """Reference fuel lower heating value or battery energy density
        """
        if (fuel_type=="kerosene"): return 43.1e6 # J/kg, kerosene
        elif (fuel_type=="gasoline"): return 46.4e6 # J/kg, gasoline
        elif (fuel_type=="liquid_h2"): return 121.0e6 # J/kg, liquid hydrogene
        elif (fuel_type=="compressed_h2"): return 140.0e6 # J/kg, compressed hydrogene
        elif (fuel_type=="methane"): return 50.3e6 # J/kg, Liquid methane
        elif (fuel_type=="battery"): return unit.J_Wh(200.) # J/kg, State of the art for lithium-ion
        else: raise Exception("fuel_type index is out of range")

    def stoichiometry(self, oxydizer,fuel):
        if oxydizer=="air":
            if fuel=="hydrogen": return 34.5
            else: raise Exception("Fuel type is unknown")
        else: raise Exception("Oxydizer type is unknown")


