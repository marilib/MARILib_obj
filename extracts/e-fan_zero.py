#!/usr/bin/env python3
"""
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve
from marilib.utils import earth, unit


def corrected_air_flow(ptot,ttot,Mach):
    """Computes the corrected air flow per square meter
    """
    r,gam,Cp,Cv = earth.gas_data()
    f_m = Mach*(1. + 0.5*(gam-1)*Mach**2)**(-(gam+1.)/(2.*(gam-1.)))
    mdoa = (np.sqrt(gam/r)*ptot/np.sqrt(ttot))*f_m
    return mdoa

def inv_corrected_air_flow(ptot,ttot,mdoa):
    """Computes the mach number from corrected air flow per square meter
    """
    r,gam,Cp,Cv = earth.gas_data()
    f_m = mdoa / (np.sqrt(gam/r)*ptot/np.sqrt(ttot))
    mach = 0.5
    for j in range(6):
        mach = f_m / (1. + 0.5*(gam-1)*mach**2)**(-(gam+1.)/(2.*(gam-1.)))
    return mach

def get_section_data(ptot,ttot,area,m_dot):
    """Computes the mach number from corrected air flow per square meter
    """
    r,gam,Cp,Cv = earth.gas_data()
    mdoa = m_dot / area
    mach = inv_corrected_air_flow(ptot,ttot,mdoa)
    fac = (1.+0.5*(gam-1.)*mach**2)
    tsta = ttot / fac
    psta = ptot / fac**(gam/(gam-1.))
    vair = mach * np.sqrt(gam*r*tsta)
    return psta,tsta,vair,mach

def fct0(x, pamb,ptot,ttot,shaft_power):
    """With fan isentropic efficiency and adapted nozzle
    """
    m_dot = x[0]
    nozzle_area = x[1]
    ttot_jet = ttot + shaft_power/(m_dot*cp)
    ptot_jet = ptot * (1. + fan_isent_eff*(ttot_jet/ttot-1.))**(gam/(gam-1.))
    mach_jet = np.sqrt(((ptot_jet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
    m_dot_ = nozzle_area * corrected_air_flow(ptot_jet,ttot_jet,mach_jet)

    tsta_jet = ttot_jet / (1.+0.5*(gam-1.)*mach**2)
    vjet = mach_jet * np.sqrt(gam*r*tsta_jet)
    fn = m_dot*(vjet - vair)
    eta_prop_ = fn*vair / shaft_power

    return [m_dot_ - m_dot,
            eta_prop_ - eta_prop]

def fct1(x, pamb,ptot,ttot,shaft_power):
    """With fan isentropic efficiency and adapted nozzle
    """
    m_dot = x[0]
    ttot_jet = ttot + shaft_power/(m_dot*cp)
    ptot_jet = ptot * (1. + fan_isent_eff*(ttot_jet/ttot-1.))**(gam/(gam-1.))
    mach_jet = np.sqrt(((ptot_jet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
    m_dot_ = nozzle_area * corrected_air_flow(ptot_jet,ttot_jet,mach_jet)
    return [m_dot_ - m_dot]

def fct2(x, pamb,ttot,vair,shaft_power):
    """With fan kinetic efficiency and adapted nozzle
    """
    m_dot = x[0]
    ttot_jet = ttot + shaft_power/(m_dot*cp)            # Stagnation temperature increases due to introduced work
    vjet = np.sqrt(2.*fan_kinet_eff*shaft_power/m_dot + vair**2)   # Supposing adapted nozzle
    tsta_jet = ttot_jet - 0.5*vjet**2/cp         # Static temperature
    mach_jet = vjet/np.sqrt(gam*r*tsta_jet)                      # Mach number at nozzle output, ignoring when Mach > 1
    ptot_jet = pamb * (1.+0.5*(gam-1.)*mach_jet**2)
    m_dot_ = nozzle_area * corrected_air_flow(ptot_jet,ttot_jet,mach_jet)    # Corrected air flow per area at fan position
    return [m_dot_ - m_dot]

def fct3(x, ptot,ttot,vair,nozzle_area,shaft_power):
    """With fan isentropic and kinetic efficiencies, thus, no need of adapted nozzle
    """
    m_dot = x[0]
    ttot_jet = ttot + shaft_power/(m_dot*cp)
    ptot_jet = ptot * (1. + fan_isent_eff*(ttot_jet/ttot-1.))**(gam/(gam-1.))
    vjet = np.sqrt(2.*fan_kinet_eff*shaft_power/m_dot + vair**2)   # Supposing adapted nozzle
    tsta_jet = ttot_jet - 0.5*vjet**2/cp         # Static temperature
    mach_jet = vjet/np.sqrt(gam*r*tsta_jet)                      # Mach number at nozzle output, ignoring when Mach > 1
    m_dot_ = nozzle_area * corrected_air_flow(ptot_jet,ttot_jet,mach_jet)    # Corrected air flow per area at fan position
    return [m_dot_ - m_dot]



altp = unit.m_ft(35000.)
disa = 0.
mach = 0.78
shaft_power = unit.W_kW(5000.)


r,gam,cp,cv = earth.gas_data()
fan_kinet_eff = 0.95
fan_isent_eff = 0.80
nozzle_area = 2.4

pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)

ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position
vair = mach * earth.sound_speed(tamb)


print("")
print("With isentropic efficiency and adapted nozzle")
print("------------------------------")

fct_arg = (pamb,ptot,ttot,shaft_power)

m_dot_init = nozzle_area * corrected_air_flow(ptot,ttot,mach)       # Corrected air flow per area at fan position
x_init = [m_dot_init]

output_dict = fsolve(fct1, x0=x_init, args=fct_arg, full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

m_dot = output_dict[0][0]

ttot_jet = ttot + shaft_power/(m_dot*cp)
ptot_jet = ptot * (1. + fan_isent_eff*(ttot_jet/ttot-1.))**(gam/(gam-1.))
mach_jet = np.sqrt(((ptot_jet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
tsta_jet = ttot_jet / (1.+0.5*(gam-1.)*mach_jet**2)
vjet = mach_jet * np.sqrt(gam*r*tsta_jet)
fn = m_dot*(vjet - vair)

eta_prop = fn*vair / (0.5*m_dot*(vjet**2 - vair**2))
eta_propeller = fn*vair / shaft_power

print("Eta prop = ", "%.3f"%eta_prop)
print("Eta propeller = ", "%.3f"%eta_propeller)
print("Air flow = ", "%.2f"%m_dot)
print("Nozzle width= ", "%.2f"%np.sqrt(4.*nozzle_area/np.pi))
print("Mach jet = ", "%.2f"%mach_jet)
print("Ttot = ", "%.2f"%ttot)
print("Ttot jet = ", "%.2f"%ttot_jet)
print("Speed jet = ", "%.2f"%vjet)
print("Delta speed = ", "%.2f"%(vjet-vair))
print("Fn = ", "%.2f"%fn)


print("")
print("With kinetic efficiency and adapted nozzle")
print("------------------------------")

fct_arg = (pamb,ttot,vair,shaft_power)

m_dot_init = nozzle_area * corrected_air_flow(ptot,ttot,mach)
x_init = [m_dot_init]

output_dict = fsolve(fct2, x0=x_init, args=fct_arg, full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

m_dot = output_dict[0][0]

ttot_jet = ttot + shaft_power/(m_dot*cp)
vjet = np.sqrt(2.*fan_kinet_eff*shaft_power/m_dot + vair**2)
tsta_jet = ttot_jet - 0.5*vjet**2/cp
mach_jet = vjet/np.sqrt(gam*r*tsta_jet)
ptot_jet = pamb * (1.+0.5*(gam-1.)*mach_jet**2)
fn = m_dot*(vjet - vair)

eta_prop = fn*vair / (0.5*m_dot*(vjet**2 - vair**2))
eta_propeller = fn*vair / shaft_power

print("Eta prop = ", "%.3f"%eta_prop)
print("Eta propeller = ", "%.3f"%eta_propeller)
print("Air flow = ", "%.2f"%m_dot)
print("Mach jet = ", "%.2f"%mach_jet)
print("Ttot jet = ", "%.2f"%ttot_jet)
print("Speed jet = ", "%.2f"%vjet)
print("Fn = ", "%.2f"%fn)


print("")
print("With isentropic and kinetic efficiencies")
print("------------------------------")

fct_arg = (ptot,ttot,vair,nozzle_area,shaft_power)

m_dot_init = nozzle_area * corrected_air_flow(ptot,ttot,mach)
x_init = [m_dot_init]

output_dict = fsolve(fct3, x0=x_init, args=fct_arg, full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

m_dot = output_dict[0][0]

ttot_jet = ttot + shaft_power/(m_dot*cp)
ptot_jet = ptot * (1. + fan_isent_eff*(ttot_jet/ttot-1.))**(gam/(gam-1.))
vjet = np.sqrt(2.*fan_kinet_eff*shaft_power/m_dot + vair**2)
tsta_jet = ttot_jet - 0.5*vjet**2/cp
mach_jet = vjet/np.sqrt(gam*r*tsta_jet)

pout = ptot_jet / (1.+0.5*(gam-1.)*mach_jet**2)
fn = m_dot*(vjet - vair) + nozzle_area*(pout - pamb)

eta_prop = fn*vair / (0.5*m_dot*(vjet**2 - vair**2))
eta_propeller = fn*vair / shaft_power

print("Eta prop = ", "%.3f"%eta_prop)
print("Eta propeller = ", "%.3f"%eta_propeller)
print("Air flow = ", "%.2f"%m_dot)
print("Mach jet = ", "%.2f"%mach_jet)
print("Ttot jet = ", "%.2f"%ttot_jet)
print("Speed jet = ", "%.2f"%vjet)
print("Fn = ", "%.2f"%fn)
print("Pamb = ", "%.2f"%pamb)
print("Pout = ", "%.2f"%pout)

