#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

:author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

import numpy as np


def atmosphere(altp, disa=0.):
    """Ambient data from pressure altitude from ground to 50 km according to Standard Atmosphere
    """
    g, r, gam = 9.80665, 287.053, 1.4
    Z = [0., 11000., 20000., 32000., 47000., 50000.]
    dtodz = [-0.0065, 0., 0.0010, 0.0028, 0.]
    P = [101325., 0., 0., 0., 0., 0.]
    T = [288.15, 0., 0., 0., 0., 0.]
    if (Z[-1] < altp):
        raise Exception("atmosphere, altitude cannot exceed "+str(Z[-1]+" m"))
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


def gas_density(pamb,tamb):
    """Ideal gas density
    """
    r = 287.053
    rho = pamb / ( r * tamb )
    return rho


def sound_speed(tamb):
    """Sound speed for ideal gas
    """
    r, gam = 287.053, 1.4
    vsnd = np.sqrt( gam * r * tamb )
    return vsnd


def air_viscosity(tamb):
    """Mixed gas dynamic viscosity, Sutherland's formula
    WARNING : result will not be accurate if gas is mixing components of too different molecular weights
    """
    mu0,T0,S = 1.715e-5, 273.15, 110.4
    mu = (mu0*((T0+S)/(tamb+S))*(tamb/T0)**1.5)
    return mu


def get_pod_climb_thrust(mass, altp, disa):
    """Compute required speed, total thrust and stall margin in initial climb conditions
    stall margin must be higher or equal to 1.13 (check for CS23)
    REMARK : Aero model includes Reynolds effect
    """
    area,czmax,kvs1g,kv,a,b,tofl = 11.75,1.99,1.13,0.7,12,300,460
    pamb,tamb,g = atmosphere(altp, disa)
    rho = gas_density(pamb,tamb)
    mu = air_viscosity(tamb)
    vs1g = np.sqrt((2*mass*g) / (rho*area*czmax))
    sigma = rho/1.225
    cz = czmax / kvs1g**2
    ml_factor = (tofl - b) / a
    fn_clb = kv * mass**2 / (cz*ml_factor*area*sigma**0.8)  # Magic Line factor
    vtas = kvs1g * vs1g
    return fn_clb, vtas, kvs1g


def get_pod_cruise_thrust(mass, altp, disa, vtas):
    """Compute required total thrust and stall margin in cruise conditions
    stall margin must be higher or equal to 1.13 (check for CS23)
    REMARK : Aero model includes Reynolds effect
    """
    area,czmax,cx0,ki,kre,lref = 11.75,1.45,0.0292,0.048,0.796138,3.3
    pamb,tamb,g = atmosphere(altp, disa)
    rho = gas_density(pamb,tamb)
    mu = air_viscosity(tamb)
    re = rho*vtas/mu
    kr = 1e3*(1/np.log(re*lref))**2.58
    q = 0.5*rho*vtas**2
    vs1g = np.sqrt((2*mass*g) / (rho*area*czmax))
    kvs1g = vtas / vs1g
    fn = q*area*cx0*(kr/kre) + ki*(mass*g)**2/(q*area)
    return fn, kvs1g


def get_pod_criteria(pod_mass_init, pod_mass, vtas, pw):
    """ Compute
    ft : total flight time
    fd : total flight distance
    """
    battery_mass, battery_enrg_density = 230, 200*3600
    ft =  (battery_mass + pod_mass_init - pod_mass) \
        * battery_enrg_density \
        / pw
    fd = ft * vtas
    return ft, fd



if __name__ == '__main__':

    mass = 850  # kg, airplane mass
    altp = 900  # m, flight altitude
    disa = 0    # deg K

    vtas = 130 * 0.2777 # m/s

    print("")
    print("Mass = ", mass, " kg")
    print("")
    print("Pressure altitude = ", altp, " m")
    print("Temperature shift = ", disa, " °C")
    print("Air speed = ", "%.2f"%(vtas/0.2777), " km/h")

    fn, kvs1g = get_pod_cruise_thrust(mass, altp, disa, vtas)

    print("")
    print("===> Total cruise thrust = ", "%.1f"%fn, " N")
    print("===> Kvs1g = ", "%.3f"%kvs1g)

    altp = 0    # m, flight altitude
    disa = 15   # deg K

    print("")
    print("Pressure altitude = ", altp, " m")
    print("Temperature shift = ", disa, " °C")

    fn_clb, vtas, kvs1g = get_pod_climb_thrust(mass, altp, disa)

    print("")
    print("===> Total climb thrust = ", "%.1f"%fn_clb, " N")
    print("===> Climb speed = ", "%.2f"%(vtas/0.2777), " km/h")
    print("===> Kvs1g = ", "%.3f"%kvs1g)


    pod_mass_init = 30
    pod_mass = 30
    pw = 30000  # W

    print("")
    print("Initial pod mass = ", "%.2f"%pod_mass_init, " kg")
    print("Current pod mass = ", "%.2f"%pod_mass, " kg")

    ft,fd = get_pod_criteria(pod_mass_init, pod_mass, vtas, pw)

    print("")
    print("===> Total flight time = ", "%.2f"%(ft/60), " min")
    print("===> Total flight distance", "%.2f"%(fd/1000), " km")

