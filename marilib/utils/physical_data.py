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

    def air_thermal_transfer_data(self, pamb,tamb,fluid_temp,air_speed, x):
        """Thermal transfert factor assuming turbulent air flow
        """
        temp = 0.5*(tamb+fluid_temp)
        r,gam,cp,cv = self.gas_data()
        rho = self.gas_density(pamb,temp)
        mu = self.air_viscosity(temp)
        alpha = self.air_thermal_diffusivity()
        pr = mu / (alpha * rho)                         # Prandtl number
        re = rho * air_speed / mu                       # 1/m, Reynolds number
        if (re*x)<1.e8 and 0.6<pr and pr<60.:
            nu = 0.0296 * (re*x)**(4/5) * pr**(1/3)     # Nusselt number, turbulent flow
        else:
            print("Re = ", re*x, "  Pr = ", pr)
            raise Exception("Re or Pr are not in the valid domain")
        lbd = alpha * rho * cp      # Thermal conductivity
        h = lbd * nu / x
        return h, rho, cp, mu, pr, re*x, nu, lbd

    def air_thermal_diffusivity(self):
        """Thermal diffusivity of the air at 300 K
        """
        # lambda/rho/cp
        thermal_diffusivity = 20.e-6   # m2/s
        return thermal_diffusivity

    def fluid_thermal_transfer_data(self, temp, wkg_temp, fluid_speed, tube_hydro_width):
        """Thermal transfert factor for turbulent flow in a tube : W/m2/K
        """
        rho, cp, mu, lbd, min_temp, max_temp = self.fluid_data(temp, wkg_temp)

        # Nusselt number
        pr = mu * cp / lbd                                  # Prandtl number
        red = (rho * fluid_speed / mu) * tube_hydro_width   # Reynolds number
        nu = 3.66   # Nusselt number in fully established stream with constant wall thermal flow

        # Thermal transfert factor
        h = lbd * nu / tube_hydro_width
        return h, rho, cp, mu, pr, red, nu, lbd

    def fluid_data(self, temp, working_temp):
        # Data from water + monopropylen 30%,  https://www.celsius-process.com/wp-content/uploads/2020/01/mpg-30.pdf
        # Heat Transfer Fluid, http://www.dow.com/heattrans
        if 274.15 < working_temp and working_temp < 373.15:
            # Water MP30
            temp_list = [263.15, 268.15, 273.15, 278.15, 283.15, 288.15, 293.15, 298.15, 303.15, 308.15, 313.15, 318.15,
                     323.15, 328.15, 333.15, 338.15, 343.15, 348.15, 353.15, 358.15, 363.15, 368.15, 373.15]
            rho_list = [1035, 1033, 1031, 1029, 1027, 1025, 1022, 1019, 1016, 1013, 1010, 1007,
                    1004, 1000, 997, 994, 990, 987, 983, 980, 976, 973, 969]
            cp_list = [3829, 3835, 3841, 3847, 3854, 3860, 3867, 3873, 3879, 3886, 3892, 3899,
                   3905, 3911, 3918, 3924, 3930, 3936, 3943, 3949, 3955, 3961, 3966]
            mu_list = [0.003849, 0.003210, 0.002703, 0.002297, 0.001968, 0.001701, 0.001480, 0.001298, 0.001146, 0.001018, 0.000910, 0.000818,
                   0.000740, 0.000672, 0.000614, 0.000563, 0.000519, 0.000480, 0.000446, 0.000416, 0.000389, 0.000365, 0.000344]
            lbd_list = [0.445, 0.449, 0.453, 0.457, 0.461, 0.465, 0.468, 0.472, 0.476, 0.479, 0.482, 0.486,
                    0.489, 0.492, 0.495, 0.498, 0.501, 0.504, 0.507, 0.509, 0.512, 0.515, 0.517]
            rho_f = interpolate.interp1d(temp_list,rho_list,kind="cubic")
            cp_f = interpolate.interp1d(temp_list,cp_list,kind="cubic")
            mu_f = interpolate.interp1d(temp_list,mu_list,kind="cubic")
            lbd_f = interpolate.interp1d(temp_list,lbd_list,kind="cubic")
            min_temp, max_temp = 274.15, 373.15

        elif 238.15 < working_temp and working_temp < 473.15:
            # DOWTHERM Q Heat Transfer Fluid
            temp_list = [238.15, 243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15, 313.15, 323.15, 333.15, 343.15,
                     353.15, 363.15, 373.15, 383.15, 393.15, 403.15, 413.15, 423.15, 433.15, 443.15, 453.15, 463.15, 473.15]
            rho_list = [1011.4, 1003.2, 995.6, 988, 980.5, 972.9, 965.4, 957.8, 950.2, 942.7, 935.1, 927.6,
                    920, 912.4, 904.9, 897.3, 889.8, 882.2, 874.6, 867.1, 859.5, 852, 844.4, 836.8, 829.3]
            cp_list = [1478, 1492, 1525, 1557, 1589, 1621, 1653, 1685, 1716, 1748, 1779, 1811,
                   1842, 1873, 1904, 1935, 1966, 1997, 2027, 2058, 2088, 2118, 2148, 2178, 2208]
            mu_list = [0.0466, 0.0242, 0.0161, 0.0109, 0.00756, 0.00542, 0.004, 0.00304, 0.00237, 0.00189, 0.00154,
                   0.00128, 0.00107, 0.00092, 0.0008, 0.0007, 0.00062, 0.00055, 0.0005, 0.00045, 0.00041, 0.00038,
                   0.00035, 0.00033, 0.00031]
            lbd_list = [0.128, 0.1277, 0.1266, 0.1255, 0.1244, 0.1232, 0.122, 0.1208, 0.1195, 0.1183, 0.117, 0.1156,
                    0.1143, 0.1129, 0.1115, 0.1101, 0.1087, 0.1072, 0.1058, 0.1043, 0.1028, 0.1013, 0.0998, 0.0982, 0.0967]
            rho_f = interpolate.interp1d(temp_list,rho_list,kind="cubic")
            cp_f = interpolate.interp1d(temp_list,cp_list,kind="cubic")
            mu_f = interpolate.interp1d(temp_list,mu_list,kind="cubic")
            lbd_f = interpolate.interp1d(temp_list,lbd_list,kind="cubic")
            min_temp, max_temp = 238.15, 473.15

        else :
            raise Exception("Working temperature is not compatible with any fluid type")

        return rho_f(temp), cp_f(temp), mu_f(temp)*1.e-3, lbd_f(temp), min_temp, max_temp

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


