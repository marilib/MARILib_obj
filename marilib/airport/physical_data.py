#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on November 20 20:20:20 2020
@author: Thierry DRUOT
"""

import numpy as np

from marilib.utils import unit


# ======================================================================================================
# Physical data
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
        elif (fuel_type=="gasoline"): return 800. # Gasoline : between 775-840 kg/m3
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
        elif (fuel_type=="gasoline"): return 46.4e6 # J/kg, gasoline
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


