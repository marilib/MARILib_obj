#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np



def m_ft(ft): return ft*0.3048   # Translate feet into metres
def ft_m(m): return m/0.3048   # Translate metres into feet

def mps_kt(kt): return kt*1852/3600   # Translate knots into meters per second
def kt_mps(mps): return mps*3600./1852.   # Translate meters per second into knots


def atmosphere(altp, disa=0.):
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
    tstd = T[j] + dtodz[j]*(altp-Z[j])
    tamb = tstd + disa
    vsnd = np.sqrt(gam * r * tamb)
    rho = pamb / (r * tamb)
    sig = rho / 1.225
    return pamb,tamb,tstd,dtodz[j],vsnd,rho,sig,gam,g


def vcas_from_mach(pamb,mach):
    """Calibrated air speed from Mach number, subsonic only
    """
    gam = 1.4
    P0 = 101325.
    vc0 = 340.29
    fac = gam/(gam-1.)
    vcas = vc0*np.sqrt(5.*((((pamb/P0)*((1.+((gam-1.)/2.)*mach**2)**fac-1.))+1.)**(1./fac)-1.))
    return vcas


def climb_mode(speed_mode,mach,dtodz,tstd,disa):
    """Acceleration factor depending on speed driver ('cas': constant CAS, 'mach': constant Mach)
    WARNING : input is mach number whatever speed_mode
    """
    g = 9.80665
    r = 287.053
    gam = 1.4

    if (speed_mode=="cas"):
        fac = (gam-1.)/2.
        acc_factor = 1. + (((1.+fac*mach**2)**(gam/(gam-1.))-1.)/(1.+fac*mach**2)**(1./(gam-1.))) \
                        + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
    elif (speed_mode=="mach"):
        acc_factor = 1. + ((gam*r)/(2.*g))*(mach**2)*(tstd/(tstd+disa))*dtodz
    else:
        raise Exception("climb_mode key is unknown")

    return acc_factor


def high_lift(hld_type,hld_conf):
    """Retrieves max lift and zero aoa lift of a given deflection (from 0 to 1)
    0 =< hld_type =< 10 : type of high lift device
    0 =< hld_conf =< 1  : (slat) flap deflection
    Typically : hld_conf = 1 ==> cz_max_ld
              : hld_conf = 0.1 to 0.5 ==> cz_max_to
    """

    # Maximum lift coefficients of different airfoils, DUBS 1987
    czmax_ld = {0 : 1.45 ,  # Clean
                1 : 2.25 ,  # Flap only, Rotation without slot
                2 : 2.60 ,  # Flap only, Rotation single slot      (ATR)
                3 : 2.80 ,  # Flap only, Rotation double slot
                4 : 2.80 ,  # Fowler Flap
                5 : 2.00 ,  # Slat only
                6 : 2.45 ,  # Slat + Flap rotation without slot
                7 : 2.70 ,  # Slat + Flap rotation single slot
                8 : 2.90 ,  # Slat + Flap rotation double slot
                9 : 3.00 ,  # Slat + Fowler                      (A320)
                10 : 3.20,  # Slat + Fowler + Fowler double slot (A321)
                }.get(hld_type, "Erreur - high_lift_, HLDtype out of range")    # 9 is default if x not found

    if (hld_type<5):
        czmax_base = 1.45      # Flap only
    else:
        if (hld_conf==0):
            czmax_base = 1.45  # Clean
        else:
            czmax_base = 2.00  # Slat + Flap

    czmax = (1-hld_conf)*czmax_base + hld_conf*czmax_ld
    cz0 = czmax - czmax_base  # Assumed the Lift vs AoA is just translated upward and Cz0 clean equal to zero
    return czmax, cz0


def l_o_d(cz):
    """Lift to drag ratio
    """
    cx = 0.05 + 0.037*cz**2
    return cz/cx


def get_s2_min_path(ne):
    """Regulatory min climb path versus number of engine
    """
    if ne==2 : s2_min_path = 0.024
    elif ne==3 : s2_min_path = 0.027
    elif ne==4 : s2_min_path = 0.030
    else: raise Exception("number of engine is not permitted")
    return s2_min_path


def get_data():
    return {"n_engine":2,
            "engine_bpr":9.,
            "reference_thrust":90000.,
            "wing_area":140.,
            "hld_type":9}


def to_thrust(pamb,tamb,mach,throttle=1., nei=0):
    """Take off thrust
    """
    ne,bpr,fn_ref,wa,hld = get_data().values()

    kth =  0.475*mach**2 + 0.091*(bpr/10.)**2 \
         - 0.283*mach*bpr/10. \
         - 0.633*mach - 0.081*bpr/10. + 1.192

    r = 287.053
    rho = pamb / ( r * tamb )
    sig = rho / 1.225

    total_thrust = fn_ref * throttle * sig**0.75 * (ne - nei)

    return total_thrust


def take_off_field_length(disa,altp,mass,hld_conf):
    """Take off field length and climb path with eventual kVs1g increase to recover min regulatory slope
    """
    ne = get_data()["n_engine"]
    s2_min_path = get_s2_min_path(ne)
    kvs1g = 1.13

    tofl,s2_path,cas,mach = take_off(kvs1g,altp,disa,mass,hld_conf)

    if(s2_min_path<s2_path):
        limitation = "fl"   # field length
    else:
        dkvs1g = 0.005
        kvs1g_ = np.array([0.,0.])
        kvs1g_[0] = kvs1g
        kvs1g_[1] = kvs1g_[0] + dkvs1g

        s2_path_ = np.array([0.,0.])
        s2_path_[0] = s2_path
        tofl,s2_path_[1],cas,mach = take_off(kvs1g_[1],altp,disa,mass,hld_conf)

        while(s2_path_[0]<s2_path_[1] and s2_path_[1]<s2_min_path):
            kvs1g_[0] = kvs1g_[1]
            kvs1g_[1] = kvs1g_[1] + dkvs1g
            tofl,s2_path_[1],cas,mach = take_off(kvs1g_[1],altp,disa,mass,hld_conf)

        if(s2_min_path<s2_path_[1]):
            kvs1g = kvs1g_[0] + ((kvs1g_[1]-kvs1g_[0])/(s2_path_[1]-s2_path_[0]))*(s2_min_path-s2_path_[0])
            tofl,s2_path,cas,mach = take_off(kvs1g,altp,disa,mass,hld_conf)
            s2_path = s2_min_path
            limitation = "s2"   # second segment
        else:
            tofl = np.nan
            kvs1g = np.nan
            s2_path = 0.
            limitation = None

    return {"tofl":tofl, "kvs1g":kvs1g, "path":s2_path, "v2":cas, "mach2":mach, "limit":limitation}


def take_off(kvs1g,altp,disa,mass,hld_conf):
    """Take off field length and climb path at 35 ft depending on stall margin (kVs1g)
    """
    ne = get_data()["n_engine"]
    wing_area = get_data()["wing_area"]
    hld_type = get_data()["hld_type"]

    czmax,cz0 = high_lift(hld_type,hld_conf)

    pamb,tamb,tstd,dtodz,vsnd,rho,sig,gam,g = atmosphere(altp,disa)

    cz_to = czmax / kvs1g**2
    mach = np.sqrt((mass*g)/(0.5*gam*pamb*wing_area*cz_to))

    fn = to_thrust(pamb,tamb,mach)

    ml_factor = mass**2 / (cz_to*fn*wing_area*sig**0.8 )  # Magic Line factor

    tofl = 15.5*ml_factor + 100.    # Magic line

    cas = vcas_from_mach(pamb,mach)

    lod = l_o_d(cz_to)
    acc_factor = climb_mode('cas', mach, dtodz, tstd, disa)
    s2_path = ( (fn*(ne-1)/ne)/(mass*g) - 1./lod ) / acc_factor

    return tofl,s2_path,cas,mach



disa = 15.
altp = m_ft(2000.)
mass = 75000.
hld_conf = 0.3

dict = take_off_field_length(disa,altp,mass,hld_conf)

print("")
print(" Take off field length = ","%0.1f"%dict["tofl"], " m")
print(" kVs1g at 35 ft = ","%0.3f"%dict["kvs1g"])
print(" Air path at 35 ft = ","%0.2f"%(dict["path"]*100.), " %")
print(" Speed at 35 ft V2 = ","%0.1f"%kt_mps(dict["v2"]), " kt")
print(" Machh at 35 ft = ","%0.3f"%dict["mach2"])
print(" Active limit = ",dict["limit"])

