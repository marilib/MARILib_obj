#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""

import numpy as np

from scipy import interpolate
from scipy.optimize import fsolve

from marilib.utils import unit



def pressure_drop(tamb,rho,vfluid,width,length):
    """Pressure drop along a cylindrical tube
    """
    hd = 0.5*width  # Hydraulic diameter 4xSection_area/wetted_perimeter
    re = fluid_reynolds_number_v(tamb,rho,vfluid)
    rex = re * length
    cf = 0.5
    for j in range(6):
        cf = (1./(2.*np.log(rex*np.sqrt(cf))-0.8))**2
    dp = 0.5 * cf * (length/hd) * rho * vfluid**2
    return dp


def pressure_drop_flat_tube(tamb,rho,vfluid,width,height,length):
    """Pressure drop along a flat tube
    """
    sec = 0.25*np.pi*height**2 + height*(width-height)
    per = np.pi*height + 2.*(width-height)
    hd = 4.*sec / per  # Hydraulic diameter 4xSection_area/wetted_perimeter
    re = fluid_reynolds_number_v(tamb,rho,vfluid)
    rex = re * length
    cf = 0.5
    for j in range(6):
        cf = (1./(2.*np.log(rex*np.sqrt(cf))-0.8))**2
    dp = 0.5 * cf * (length/hd) * rho * vfluid**2
    return dp


def fluid_thermal_transfert_factor(tamb,vfluid,length, fluid="water"):
    """Thermal transfert factor for turbulent flow : W/m2/K
    """
    lbd = fluid_thermal_conductivity(tamb, fluid=fluid)
    nux = fluid_nusselt_number(tamb,vfluid,length, fluid=fluid)
    h = lbd * nux / length
    return h


def fluid_nusselt_number(tamb,vfluid,x, fluid="water"):
    """Nusselt number for turbulent flow
    """
    if fluid!="water":
        raise Exception("fluide type is not permitted")
    rho_water = 1000.
    pr = fluid_prandtl_number(tamb, fluid="water")
    re = fluid_reynolds_number_v(tamb,rho_water,vfluid, fluid="water")
    if (re*x)>5.e5 and 0.6<pr and pr<60.:
        nu = 0.0296 * (re*x)**(4/5) * pr**(1/3)
    else:
        print("Re = ", re*x, "  Pr = ", pr)
        raise Exception("Re or Pr are not in the valid domain")
    return nu


def fluid_prandtl_number(tamb, fluid="water"):
    """Prandtl number
    """
    if fluid!="water":
        raise Exception("fluide type is not permitted")
    Cp_water = 4200.
    mu = fluid_viscosity(tamb, fluid="water")
    lbd = fluid_thermal_conductivity(tamb, fluid="water")
    pr = mu * Cp_water / lbd
    return pr


def fluid_thermal_conductivity(tamb, fluid="water"):
    """Thermal conductivity of water
    """
    if fluid!="water":
        raise Exception("fluide type is not permitted")
    if tamb<274. or 370.<tamb:
        raise Exception("fluide temperature is out of range")
    t0 = 298.15
    c0 = 0.6065
    return c0*(-1.48445 + 4.12292*(tamb/t0) - 1.63866*(tamb/t0)**2)


def fluid_reynolds_number_v(tamb,rho,vfluid, fluid="water"):
    """Reynolds number
    """
    mu = fluid_viscosity(tamb, fluid=fluid)
    re = rho*vfluid/mu
    return re


def fluid_viscosity(tamb, fluid="water"):
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
    return mu_f(tamb)*1.e-3


def air_thermal_transfert_factor(pamb,tamb,vair,length):
    """Thermal transfert factor for turbulent air flow
    """
    lbd = air_thermal_conductivity(pamb,tamb)
    nux = air_nusselt_number(pamb,tamb,vair,length)
    h = lbd * nux / length
    return h

def air_nusselt_number(pamb,tamb,vair,x):
    """Nusselt number for turbulent air flow
    """
    rho,sig = air_density(pamb,tamb)
    pr = air_prandtl_number(pamb,tamb)
    re = air_reynolds_number_v(tamb,rho,vair)
    if (re*x)>5.e5 and 0.6<pr and pr<60.:
        nu = 0.0296 * (re*x)**(4/5) * pr**(1/3)
    else:
        print("Re = ", re*x, "  Pr = ", pr)
        raise Exception("Re or Pr are not in the valid domain")
    return nu


def air_prandtl_number(pamb,tamb):
    """Prandtl number
    """
    r,gam,Cp,Cv = gas_data()
    mu = gas_viscosity(tamb, gas="air")
    lbd = air_thermal_conductivity(pamb,tamb)
    pr = mu * Cp / lbd
    return pr


def air_thermal_diffusivity():
    """Thermal diffusivity of the air
    """
    thermal_diffusivity = 20.e-6   # m2/s
    return thermal_diffusivity


def air_thermal_conductivity(pamb,tamb):
    """Thermal conductivity of the air
    """
    r,gam,Cp,Cv = gas_data()
    th_diff = air_thermal_diffusivity()
    rho,sig = air_density(pamb,tamb)
    thermal_condictivity = th_diff * rho * Cp
    return thermal_condictivity


def air_reynolds_number_v(tamb,rho,vair):
    """Reynolds number based on Sutherland viscosity model
    """
    mu = gas_viscosity(tamb, gas="air")
    re = rho*vair/mu
    return re


def air_density(pamb,tamb):
    """Ideal gas density
    """
    r,gam,Cp,Cv = gas_data()
    rho0 = sea_level_density()
    rho = pamb / ( r * tamb )
    sig = rho / rho0
    return rho, sig


def sea_level_density():
    """Reference air density at sea level
    """
    rho0 = 1.225    # (kg/m3) Air density at sea level
    return rho0


def sound_speed(tamb):
    """Sound speed for ideal gas
    """
    r,gam,Cp,Cv = gas_data()
    vsnd = np.sqrt( gam * r * tamb )
    return vsnd


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


def atmosphere(altp,disa=0.):
    """Pressure from pressure altitude from ground to 50 km
    """
    g = 9.80665
    R,gam,Cp,Cv = gas_data()

    Z = np.array([0., 11000., 20000.,32000., 47000., 50000.])
    dtodz = np.array([-0.0065, 0., 0.0010, 0.0028, 0.])

    P = np.array([101325., 0., 0., 0., 0., 0.])
    T = np.array([288.15, 0., 0., 0., 0., 0.])

    if (Z[-1]<altp):
        raise Exception("atmosphere, altitude cannot exceed 50km")

    j = 0

    while (Z[1+j]<=altp):
        T[j+1] = T[j] + dtodz[j]*(Z[j+1]-Z[j])
        if (0.<np.abs(dtodz[j])):
            P[j+1] = P[j]*(1. + (dtodz[j]/T[j])*(Z[j+1]-Z[j]))**(-g/(R*dtodz[j]))
        else:
            P[j+1] = P[j]*np.exp(-(g/R)*((Z[j+1]-Z[j])/T[j]))
        j = j + 1

    if (0.<np.abs(dtodz[j])):
        pamb = P[j]*(1 + (dtodz[j]/T[j])*(altp-Z[j]))**(-g/(R*dtodz[j]))
    else:
        pamb = P[j]*np.exp(-(g/R)*((altp-Z[j])/T[j]))
    tstd = T[j] + dtodz[j]*(altp-Z[j])
    tamb = tstd + disa

    return pamb,tamb



t0 = 273.15

# Data
#-----------------------------------------------------------------------------------------------
pw_fc = unit.W_kW(50.)          # Total stack power
area_stack = 0.4/unit.W_kW(1.)  # m2/kW
se_fc = area_stack * pw_fc      # m2
length_fc = 0.4
temp_fc = t0 + 75.      # °K
h_fc = 350.             # W/m2/K

ct_alu = 237.           # W/m/K, Conductivité thermique de l'aluminium
e_alu = 0.002           # m, épaisseur alu
rho_fluid = 1000.       # density of water
cp_fluid = 4200.        # J/kg/K

n_tube = 50             # Number of tubes
d_tube = 0.004          # tube diameter
sp_fluid = 0.25*np.pi*d_tube**2 * n_tube    # m2, Total tube section


vair = 150.             # m/s
temp_air = t0 + 15.     # K

se_length = 10.         # m
se_width = 0.25         # m
se_air = se_length * se_width * 2   # m2, upper surface + lower surface


dp_fc = unit.Pa_bar(0.4)
dp_pipe = unit.Pa_bar(0.2)




altp = 0.
disa = 0.

pamb,tamb = atmosphere(altp, disa)

w_pump = unit.W_kW(0.1)




# Computation
#-----------------------------------------------------------------------------------------------
def fct(x):

    vfluid1, h_temp, l_temp = x

    temp_fluid = 0.5*(h_temp+l_temp)

    h_air = air_thermal_transfert_factor(pamb,tamb,vair,se_width)
    h_fluid = fluid_thermal_transfert_factor(temp_fluid,vfluid1,se_length)
    # h_fc = fluid_thermal_transfert_factor(temp_fluid,vfluid1,length_fc)
    h_rad = 1. / (1./h_air + e_alu/ct_alu + 1./h_fluid)

    dp_rad = pressure_drop(tamb,rho_fluid,vfluid1,d_tube,se_length)

    fluid_flow = w_pump / (dp_fc + dp_pipe + dp_rad)

    vfluid2 = fluid_flow / sp_fluid

    m_dot = fluid_flow * rho_fluid

    q1_fc = - h_fc * se_fc * (h_temp - l_temp) / np.log((temp_fc-h_temp)/(temp_fc-l_temp))

    q1_fluid = m_dot * cp_fluid * (h_temp - l_temp)

    q1_rad = - h_rad * se_air * (h_temp - l_temp) / np.log((l_temp-temp_air)/(h_temp-temp_air))

    return [vfluid1 - vfluid2, q1_fc - q1_fluid, q1_rad - q1_fluid]

r = 0.33
xini = [5.,
        (1.-r)*temp_fc + r*temp_air,
        r*temp_fc + (1.-r)*temp_air]

output_dict = fsolve(fct, x0=xini, args=(), full_output=True)

if (output_dict[2]!=1): raise Exception("Convergence problem")

vfluid, h_temp, l_temp = output_dict[0]




temp_fluid = 0.5*(h_temp+l_temp)

h_air = air_thermal_transfert_factor(pamb,tamb,vair,se_width)
h_fluid = fluid_thermal_transfert_factor(temp_fluid,vfluid,se_width)
# h_fc = fluid_thermal_transfert_factor(temp_fluid,vfluid,length_fc)
h_rad = 1. / (1./h_air + e_alu/ct_alu + 1./h_fluid)

dp_rad = pressure_drop(tamb,rho_fluid,vfluid,d_tube,se_length)
m_dot = (w_pump * rho_fluid) / (dp_fc + dp_pipe + dp_rad)
q1_fluid = m_dot * cp_fluid * (h_temp - l_temp)

print("Exchange area in the stack = ""%0.1f"%(se_fc), " m2")
print("Exchange area with air = ""%0.1f"%(se_air), " m2")
print("Fluid mass flow = ""%0.1f"%(m_dot), " kg/s")
print("Fluid speed in the dissipator = ""%0.1f"%(vfluid), " m/s")
print("Pressure drop in the dissipator = ""%0.2f"%unit.bar_Pa(dp_rad), " bar")
print("Total pressure drop = ""%0.2f"%unit.bar_Pa(dp_fc + dp_pipe + dp_rad), " bar")
print("Coefficient h air = ""%0.1f"%(h_air), " W/m2/K")
print("Coefficient h fluid = ""%0.1f"%(h_fluid), " W/m2/K")
print("Coefficient h radiateur = ""%0.1f"%(h_rad), " W/m2/K")
print("Coefficient h fuel cell = ""%0.1f"%(h_fc), " W/m2/K")
print("Thermal flow = ""%0.1f"%unit.kW_W(q1_fluid), " kW")
print("High temperature = ", "%0.1f"%(h_temp-t0), " °C")
print("Low temperature = ", "%0.1f"%(l_temp-t0), " °C")
