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
    if (re*x)<1.e8 and 0.6<pr and pr<60.:
        nu = 0.0296 * (re*x)**(4/5) * pr**(1/3)
    else:
        print("vfluid = ", vfluid, "Re = ", re*x, "  Pr = ", pr)
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
    if (re*x)<1.e8 and 0.6<pr and pr<60.:
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


def drag_force(pamb,tamb,vair,length,wet_area):
    """Compute drag coefficient
    """
    form_factor = 1.5
    rho,sig = air_density(pamb,tamb)
    re = air_reynolds_number_v(tamb,rho,vair)
    mach = vair / sound_speed(tamb)
    fac = ( 1. + 0.126*mach**2 )
    cxf = form_factor * ((0.455/fac)*(np.log(10)/np.log(re*length))**2.58)
    return (0.5*rho*vair**2)*cxf*wet_area


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

# Leading edge heat exchanger
#-----------------------------------------------------------------------------------------------
hex_length = 10.        # m, Exchanger span wise length
hex_chord = 0.25         # m, Exchanger chord
hex_air_area = hex_length * hex_chord * 2.  # lower + upper surface

hex_tube_period = 0.01  # m, Distance between tubes
hex_tube_width = 0.005  # m, Internal tube diameter

hex_tube_thick = 0.003  # m, Mean wall thickness
hex_conduct = 237.      # W/m/K, Conductivité thermique de l'aluminium

hex_tube_count = int(2.*hex_chord / hex_tube_period)

hex_tube_section = 0.25*np.pi*hex_tube_width**2
hex_tube_perimeter = np.pi*hex_tube_width
hex_hydro_width = 4.* hex_tube_section / hex_tube_perimeter
hex_tube_area = hex_tube_perimeter * hex_length * hex_tube_count

# Fuel cell stack data
#-----------------------------------------------------------------------------------------------
fcs_power = unit.W_kW(150.)                  # Total stack power
fcs_specific_area = 0.4/unit.W_kW(1.)       # m2/kW
fcs_area = fcs_specific_area * fcs_power    # m2
fcs_length = 0.4            # m
fcs_temp = t0 + 75.         # °K
fcs_h = 350.                # W/m2/K
fcs_pd = unit.Pa_bar(0.001)   # Fluid pressure drop through the fuel cell stack

# Fluid data
#-----------------------------------------------------------------------------------------------
fluid_rho = 1000.           # density of water
fluid_cp = 4200.            # J/kg/K
fluid_pd = unit.Pa_bar(0.2) # Pressure drop in the pipes



# Experiment
#-----------------------------------------------------------------------------------------------
air_speed = 50.        # m/s
altp = 0.
disa = 0.

pamb,tamb = atmosphere(altp, disa)

air_temp = tamb     # K

pump_power = unit.W_kW(0.50)


def fct_hex(x):

    vfluid1, h_temp, l_temp = x

    temp_fluid = 0.5*(h_temp + l_temp)

    h_air = air_thermal_transfert_factor(pamb,tamb,air_speed,hex_chord)
    h_fluid = fluid_thermal_transfert_factor(temp_fluid,vfluid1,hex_hydro_width)
    h_hex = 1. / (1./h_air + hex_air_area*hex_tube_thick/hex_conduct + hex_air_area/(hex_tube_area*h_fluid))

    hex_fluid_pd = pressure_drop(tamb,fluid_rho,vfluid1,hex_tube_width,hex_length)

    fluid_flow = pump_power / (fcs_pd + fluid_pd + hex_fluid_pd)

    vfluid2 = fluid_flow / (hex_tube_section * hex_tube_count)

    m_dot = fluid_flow * fluid_rho

    q1_fcs = - fcs_h * fcs_area * (h_temp - l_temp) / np.log((fcs_temp-h_temp)/(fcs_temp-l_temp))
    q1_fluid = m_dot * fluid_cp * (h_temp - l_temp)
    q1_hex = - h_hex * hex_air_area * (h_temp - l_temp) / np.log((l_temp-air_temp)/(h_temp-air_temp))

    return [vfluid1 - vfluid2, q1_fcs - q1_fluid, q1_hex - q1_fluid]


# r = 0.15
# xini = [5.,
#         (1.-r)*fcs_temp + r*air_temp,
#         r*fcs_temp + (1.-r)*air_temp]

r = 0.15
xini = [5.,
        (1.-r)*fcs_temp,
        (1.-r)*fcs_temp - 4.]

output_dict = fsolve(fct_hex, x0=xini, args=(), full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")
vfluid, h_temp, l_temp = output_dict[0]



temp_fluid = 0.5*(h_temp + l_temp)

h_air = air_thermal_transfert_factor(pamb,tamb,air_speed,hex_chord)
h_fluid = fluid_thermal_transfert_factor(temp_fluid,vfluid,hex_hydro_width)
h_hex = 1. / (1./h_air + hex_air_area*hex_tube_thick/hex_conduct + hex_air_area/(hex_tube_area*h_fluid))

hex_fluid_pd = pressure_drop(tamb,fluid_rho,vfluid,hex_tube_width,hex_length)

fluid_flow = pump_power / (fcs_pd + fluid_pd + hex_fluid_pd)

vfluid2 = fluid_flow / (hex_tube_section * hex_tube_count)

m_dot = fluid_flow * fluid_rho

q1_fcs = - fcs_h * fcs_area * (h_temp - l_temp) / np.log((fcs_temp-h_temp)/(fcs_temp-l_temp))
q1_fluid = m_dot * fluid_cp * (h_temp - l_temp)
q1_hex = - h_hex * hex_air_area * (h_temp - l_temp) / np.log((l_temp-air_temp)/(h_temp-air_temp))

print("")
print("Exchange area with air = ""%0.1f"%(hex_air_area), " m2")
print("Air speed through the exchanger = ""%0.1f"%(air_speed), " m/s")
print("Fluid mass flow = ""%0.1f"%(m_dot), " kg/s")
print("Fluid speed in the exchangeur = ""%0.1f"%(vfluid), " m/s")
print("Fluid pressure drop in the exchangeur = ""%0.2f"%unit.bar_Pa(hex_fluid_pd), " bar")
print("Total fluid pressure drop = ""%0.2f"%unit.bar_Pa(fcs_pd + fluid_pd + hex_fluid_pd), " bar")
print("")
print("Coefficient h air = ""%0.1f"%(h_air), " W/m2/K")
print("Coefficient h fluid = ""%0.1f"%(h_fluid), " W/m2/K")
print("Coefficient h exchanger = ""%0.1f"%(h_hex), " W/m2/K")
print("Coefficient h fuel cell = ""%0.1f"%(fcs_h), " W/m2/K")
print("")
print("Thermal flow = ""%0.1f"%unit.kW_W(q1_fluid), " kW")
print("High temperature = ", "%0.1f"%(h_temp-t0), " °C")
print("Low temperature = ", "%0.1f"%(l_temp-t0), " °C")







