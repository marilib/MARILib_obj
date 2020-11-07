#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 23:22:21 2019

@author: DRUOT Thierry : original Scilab implementation
         PETEILH Nicolas : portage to Python
"""

import numpy

from scipy.optimize import fsolve


def gravity():
    """Reference gravity acceleration
    """
    g = 9.80665     # Gravity acceleration at sea level
    return g

def sea_level_density():
    """Reference air density at sea level
    """
    rho0 = 1.225    # (kg/m3) Air density at sea level
    return rho0

def sea_level_pressure():
    """Reference air pressure at sea level
    """
    P0 = 101325.    # Pascals
    return P0

def sea_level_temperature():
    """Reference air temperature at sea level
    """
    T0 = 288.15    # Kelvins
    return T0

def sea_level_sound_speed():
    """Reference sound speed at sea level
    """
    vc0 = 340.29    # m/s
    return vc0

def gas_data(gas="air"):
    """Gas data for a single gas
    """
    r = {"air" : 287.053 ,
         "argon" : 208. ,
         "carbon_dioxide" : 188.9 ,
         "carbon_monoxide" : 297. ,
         "helium" : 2077. ,
         "hydrogen" : 4124. ,
         "methane" : 518.3 ,
         "nitrogen" : 296.8 ,
         "oxygen" : 259.8 ,
         "propane" : 189. ,
         "sulphur_dioxide" : 130. ,
         "steam" : 462.
         }.get(gas, "Erreur: type of gas is unknown")

    gam = {"air" : 1.40 ,
           "argon" : 1.66 ,
           "carbon_dioxide" : 1.30 ,
           "carbon_monoxide" : 1.40 ,
           "helium" : 1.66 ,
           "hydrogen" : 1.41 ,
           "methane" : 1.32 ,
           "nitrogen" : 1.40 ,
           "oxygen" : 1.40 ,
           "propane" : 1.13 ,
           "sulphur_dioxide" : 1.29 ,
           "steam" : 1.33
           }.get(gas, "Erreur: type of gas is unknown")

    cv = r/(gam-1.)
    cp = gam*cv
    return r,gam,cp,cv

def gas_viscosity(tamb, gas="air"):
    """Mixed gas dynamic viscosity, Sutherland's formula
    WARNING : result will not be accurate if gas is mixing components of too different molecular weights
    """
    data = {"air"             : [1.715e-5, 273.15, 110.4] ,
            "ammonia"         : [0.92e-5, 273.15, 382.9] ,
            "argon"           : [2.10e-5, 273.15, 155.6] ,
            "benzene"         : [0.70e-5, 273.15, 173.1] ,
            "carbon_dioxide"  : [1.37e-5, 273.15, 253.4] ,
            "carbon_monoxide" : [1.66e-5, 273.15,  94.0] ,
            "chlorine"        : [1.23e-5, 273.15, 273.0] ,
            "chloroform"      : [0.94e-5, 273.15, 284.2] ,
            "ethylene"        : [0.97e-5, 273.15, 163.7] ,
            "helium"          : [1.87e-5, 273.15,  69.7] ,
            "hydrogen"        : [0.84e-5, 273.15,  60.4] ,
            "methane"         : [1.03e-5, 273.15, 166.3] ,
            "neon"            : [2.98e-5, 273.15,  80.8] ,
            "nitrogen"        : [1.66e-5, 273.15, 110.9] ,
            "nitrous oxide"   : [1.37e-5, 273.15, 253.4] ,
            "oxygen"          : [1.95e-5, 273.15,  57.9] ,
            "steam"           : [0.92e-5, 273.15, 154.8] ,
            "sulphur_dioxide" : [1.16e-5, 273.15, 482.3] ,
            "xenon"           : [2.12e-5, 273.15, 302.6]
            }                 #  mu0      T0      S
    # gas={"nitrogen":0.80, "oxygen":0.20}
    # mu = 0.
    # for g in list(gas.keys()):
    #     [mu0,T0,S] = data[g]
    #     mu = mu + gas[g]*(mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
    mu0,T0,S = data[gas]
    mu = (mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
    return mu

def reynolds_number_old(pamb,tamb,mach):
    """Reynolds number
    """
    fac = ( 1. + 0.126*mach**2 )
    re = 47899*pamb*mach*(fac*tamb + 110.4) / (tamb**2 * fac**2.5)
    return re

def reynolds_number(pamb,tamb,mach):
    """Reynolds number based on Sutherland viscosity model
    """
    vsnd = sound_speed(tamb)
    rho,sig = air_density(pamb,tamb)
    mu = gas_viscosity(tamb)
    re = rho*vsnd*mach/mu
    return re


def atmosphere(altp,disa):
    """Pressure from pressure altitude from ground to 50 km
    """
    g = gravity()
    R,gam,Cp,Cv = gas_data()

    Z = numpy.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = numpy.array([-0.0065, 0., 0.0010, 0.0028, 0.])

    P = numpy.array([sea_level_pressure(), 0., 0., 0., 0., 0.])
    T = numpy.array([sea_level_temperature(), 0., 0., 0., 0., 0.])

    if (Z[-1]<altp):
        raise Exception("atmosphere, altitude cannot exceed 50km")

    j = 0

    while (Z[1+j]<=altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<numpy.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(R*dtodz[j]))
        else:
            P[j+1] = P[j]*numpy.exp(-(g/R)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0.<numpy.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(R*dtodz[j]))
    else:
        pamb = P[j]*numpy.exp(-(g/R)*((altp-Z[j])/T[j]))
    tstd = T[j] + dtodz[j]*(altp-Z[j])
    tamb = tstd + disa

    return pamb,tamb,tstd,dtodz[j]


def altg_from_altp(altp,disa):
    """Geometrical altitude from pressure altitude
    """
    def fct(altg,altp,disa):
        pamb,tamb,dtodz = atmosphere_geo(altg,disa)
        zp = pressure_altitude(pamb)
        return altp-zp

    output_dict = fsolve(fct, x0=altp, args=(altp,disa), full_output=True)

    altg = output_dict[0][0]
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    return altg


def atmosphere_geo(altg,disa):
    """Pressure and temperature from geometrical altitude from ground to 50 km
    """
    g = gravity()
    R,gam,Cp,Cv = gas_data()

    Zi = numpy.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodzi = numpy.array([-0.0065, 0., 0.0010, 0.0028, 0.])

    Z = numpy.zeros_like(Zi)
    dtodz = numpy.zeros_like(dtodzi)

    P = numpy.array([sea_level_pressure(), 0., 0., 0., 0., 0.])
    T = numpy.array([sea_level_temperature(), 0., 0., 0., 0., 0.])

    K = 1 + disa/T[0]
    dtodz[0] = dtodzi[0]/K
    Z[1] = Z[0] + (Zi[1]-Zi[0])*K

    n = len(P)-1
    j = 0

    while (j<n and Z[1+j]<=altg):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<numpy.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/(T[j]+disa))*(Z[j+1]-Z[j]))**(-g/(R*dtodz[j]))
        else:
            P[j+1] = P[j]*numpy.exp(-(g/R)*((Z[j+1]-Z[j])/(T[j]+disa)))
        j = j + 1
        K = 1 + disa/T[j]
        dtodz[j] = dtodzi[j]/K
        Z[j+1] = Z[j] + (Zi[j+1]-Zi[j])*K

    if (Z[1+j]<altg):
        raise Exception("atmosphere_geo, altitude cannot exceed 50km")

    if (0.<numpy.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/(T[j]+disa))*(altg-Z[j]))**(-g/(R*dtodz[j]))
    else:
        pamb = P[j]*numpy.exp(-(g/R)*((altg-Z[j])/(T[j]+disa)))
    tamb = T[j] + dtodz[j]*(altg-Z[j]) + disa

    return pamb,tamb,dtodz[j]


def pressure_altitude(pamb):
    """Pressure altitude from ground to 50 km
    """
    g = gravity()
    R,gam,Cp,Cv = gas_data()

    Z = numpy.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = numpy.array([-0.0065, 0., 0.0010, 0.0028, 0.])

    P = numpy.array([sea_level_pressure(), 0., 0., 0., 0., 0.])
    T = numpy.array([sea_level_temperature(), 0., 0., 0., 0., 0.])

    j = 0
    n = len(P)-1
    P[1] = P[0]*(1. + (dtodz[0]/T[0])*(Z[1]-Z[0]))**(-g/(R*dtodz[0]))
    T[1] = T[0] + dtodz[0]*(Z[1]-Z[0])

    while (j<n and pamb<P[j+1]):
        j = j + 1
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<numpy.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(R*dtodz[j]))
        else:
            P[j+1] = P[j]*numpy.exp(-(g/R)*((Z[j+1]-Z[j])/T[j]))

    if (pamb<P[j+1]):
        raise Exception("pressure_altitude, altitude cannot exceed 50km")

    if (0.<numpy.abs(dtodz[j])):
        altp = Z[j] + ((pamb/P[j])**(-(R*dtodz[j])/g) - 1)*(T[j]/dtodz[j])
    else:
        altp = Z[j] - (T[j]/(g/R))*numpy.log(pamb/P[j])

    return altp


def pressure(altp):
    """Pressure from pressure altitude from ground to 50 km
    """
    g = gravity()
    R,gam,Cp,Cv = gas_data()

    Z = numpy.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = numpy.array([-0.0065, 0., 0.0010, 0.0028, 0.])

    P = numpy.array([sea_level_pressure(), 0., 0., 0., 0., 0.])
    T = numpy.array([sea_level_temperature(), 0., 0., 0., 0., 0.])

    if (Z[-1]<altp):
        raise Exception("pressure, altitude cannot exceed 50km")

    j = 0

    while (Z[1+j]<=altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<numpy.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(R*dtodz[j]))
        else:
            P[j+1] = P[j]*numpy.exp(-(g/R)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0.<numpy.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(R*dtodz[j]))
    else:
        pamb = P[j]*numpy.exp(-(g/R)*((altp-Z[j])/T[j]))

    return pamb


def air_density(pamb,tamb):
    """Ideal gas density
    """
    r,gam,Cp,Cv = gas_data()
    rho0 = sea_level_density()
    rho = pamb / ( r * tamb )
    sig = rho / rho0
    return rho, sig


def sound_speed(tamb):
    """Sound speed for ideal gas
    """
    r,gam,Cp,Cv = gas_data()
    vsnd = numpy.sqrt( gam * r * tamb )
    return vsnd

def total_temperature(tamb,mach):
    """Stagnation temperature
    """
    r,gam,Cp,Cv = gas_data()
    ttot = tamb*(1.+((gam-1.)/2.)*mach**2)
    return ttot

def total_pressure(pamb,mach):
    """Stagnation pressure
    """
    r,gam,Cp,Cv = gas_data()
    ptot = pamb*(1+((gam-1.)/2.)*mach**2)**(gam/(gam-1.))
    return ptot

def vtas_from_mach(altp,disa,mach):
    """True air speed from Mach number, subsonic only
    """
    pamb,tamb,tstd,dtodz = atmosphere(altp,disa)
    vsnd = sound_speed(tamb)
    vtas = vsnd*mach
    return vtas

def mach_from_vcas(pamb,Vcas):
    """Mach number from calibrated air speed, subsonic only
    """
    r,gam,Cp,Cv = gas_data()
    P0 = sea_level_pressure()
    vc0 = sea_level_sound_speed()
    fac = gam/(gam-1.)
    mach = numpy.sqrt(((((((gam-1.)/2.)*(Vcas/vc0)**2+1)**fac-1.)*P0/pamb+1.)**(1./fac)-1.)*(2./(gam-1.)))
    return mach

def vcas_from_mach(pamb,mach):
    """Calibrated air speed from Mach number, subsonic only
    """
    r,gam,Cp,Cv = gas_data()
    P0 = sea_level_pressure()
    vc0 = sea_level_sound_speed()
    fac = gam/(gam-1.)
    vcas = vc0*numpy.sqrt((2./(gam-1.))*((((pamb/P0)*((1.+((gam-1.)/2.)*mach**2)**fac-1.))+1.)**(1./fac)-1.))
    return vcas

def vtas_from_vcas(altp,disa,vcas):
    """True air speed from calibrated air speed, subsonic only
    """
    pamb,tamb,tstd,dtodz = atmosphere(altp,disa)
    mach = mach_from_vcas(pamb,vcas)
    vsnd = sound_speed(tamb)
    vtas = vsnd*mach
    return vtas

def cross_over_altp(Vcas,mach):
    """Altitude where constant calibrated air speed meets constant Mach number, subsonic only
    """
    r,gam,Cp,Cv = gas_data()
    P0 = sea_level_pressure()
    vc0 = sea_level_sound_speed()
    fac = gam/(gam-1)

    pamb = ((1.+((gam-1.)/2.)*(Vcas/vc0)**2)**fac-1.)*P0/((1.+((gam-1.)/2.)*mach**2)**fac-1.)

    altp = pressure_altitude(pamb)
    return altp

def climb_mode(speed_mode,mach,dtodz,tstd,disa):
    """Acceleration factor depending on speed driver ('cas': constant CAS, 'mach': constant Mach)
    WARNING : input is mach number whatever speed_mode
    """
    g = gravity()
    r,gam,Cp,Cv = gas_data()

    if (speed_mode=="cas"):
        fac = (gam-1.)/2.
        acc_factor = 1. + (((1.+fac*mach**2)**(gam/(gam-1.))-1.)/(1.+fac*mach**2)**(1./(gam-1.))) \
                        + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
    elif (speed_mode=="mach"):
        acc_factor = 1. + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
    else:
        raise Exception("climb_mode key is unknown")

    return acc_factor

def fuel_density(fuel_type, press=101325.):
    """Reference fuel density
    """
    if (fuel_type=="kerosene"):
        fuel_density = 803. # Kerosene : between 775-840 kg/m3
    elif (fuel_type=="liquid_h2"):
        fuel_density = 70.8 # Liquid hydrogene
    elif (fuel_type=="Compressed_h2"):
        p = press*1.e-5
        fuel_density = (-3.11480362e-05*p + 7.82320891e-02)*p + 1.03207822e-01 # Compressed hydrogen at 293.15 K
    elif (fuel_type=="methane"):
        fuel_density = 422.6 # Liquid methane
    elif (fuel_type=="battery"):
        fuel_density = 2800. # Lithium-ion
    else:
        raise Exception("fuel_type key is unknown")
    return fuel_density

def fuel_heat(fuel_type):
    """Reference fuel lower heating value or battery energy density
    """
    if (fuel_type=="kerosene"):
        fuel_heat = 43.1e6 # J/kg, kerosene
    elif (fuel_type in ["liquid_h2", "Compressed_h2"]):
        fuel_heat = 121.0e6 # J/kg, hydrogene
    elif (fuel_type=="methane"):
        fuel_heat = 50.3e6 # J/kg, Liquid methane
    else:
        raise Exception("fuel_type index is out of range")
    return fuel_heat

def stoichiometry(oxydizer,fuel):
    if oxydizer=="air":
        if fuel=="hydrogen":
            return 34.5
        else:
            raise Exception("Fuel type is unknown")
    else:
        raise Exception("Oxydizer type is unknown")

def emission_index(fuel_type,compound):
    """Various emitted compound depending on energy source
    """
    if (fuel_type in ["kerosene"]):
        index = {"CO2" : 3140./1000.,
                 "H2O" : 1290./1000.,
                 "SO2" : 0.8/1000.,
                 "NOx" : 14./1000.,
                 "CO" : 3./1000.,
                 "HC" : 0.4/1000.,
                 "sulfuric_acid" : 0.04/1000.,
                 "nitrous_acid" : 0.4/1000.,
                 "nitric_acid" : 0.2/1000.,
                 "soot" : 2.5e12}
        return index.get(compound)
    elif (fuel_type in ["liquid_h2", "Compressed_h2"]):
        index = {"CO2" : 0.,
                 "H2O" : 18000./1000.,
                 "SO2" : 0.,
                 "NOx" : 14./1000.,
                 "CO" : 0.,
                 "HC" : 0.,
                 "sulfuric_acid" : 0.,
                 "nitrous_acid" : 0.4/1000.,
                 "nitric_acid" : 0.2/1000.,
                 "soot" : 2.0e12}
        return index.get(compound)
    elif (fuel_type in ["battery"]):
        index = {"CO2" : 0.,
                 "H2O" : 0.,
                 "SO2" : 0.,
                 "NOx" : 0.,
                 "CO" : 0.,
                 "HC" : 0.,
                 "sulfuric_acid" : 0.,
                 "nitrous_acid" : 0.,
                 "nitric_acid" : 0.,
                 "soot" : 0.}
        return index.get(compound)


