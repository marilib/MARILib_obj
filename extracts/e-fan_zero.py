#!/usr/bin/env python3
"""
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC

.. note:: All physical parameters are given in SI units.
"""

import numpy as np
from scipy.optimize import fsolve
from marilib.utils import earth, unit, math


def corrected_air_flow(Ptot,Ttot,Mach):
    """Computes the corrected air flow per square meter
    """
    r,gam,Cp,Cv = earth.gas_data()
    f_m = Mach*(1. + 0.5*(gam-1)*Mach**2)**(-(gam+1.)/(2.*(gam-1.)))
    cqoa = (np.sqrt(gam/r)*Ptot/np.sqrt(Ttot))*f_m
    return cqoa

def fct(x, pamb,ptot,ttot,shaft_power):
    m_dot = x[0]
    ttot_jet = ttot + shaft_power/(m_dot*cp)
    ptot_jet = ptot * (1. + fan_eff*(ttot_jet/ttot-1.))**(gam/(gam-1.))
    mach_jet = np.sqrt(((ptot_jet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))
    m_dot_ = nozzle_area * corrected_air_flow(ptot_jet,ttot_jet,mach_jet)

    return [m_dot_ - m_dot]


altp = unit.m_ft(1500.)
disa = 0.
mach = 0.35
shaft_power = unit.W_kW(500.)


r,gam,cp,cv = earth.gas_data()
fan_eff = 0.94
nozzle_area = 2.

pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)

ptot = earth.total_pressure(pamb, mach)        # Total pressure at inlet position
ttot = earth.total_temperature(tamb, mach)     # Total temperature at inlet position
vair = mach * earth.sound_speed(tamb)

fct_arg = (pamb,ptot,ttot,shaft_power)

m_dot_init = nozzle_area * corrected_air_flow(ptot,ttot,mach)       # Corrected air flow per area at fan position
x_init = [m_dot_init]

# Computation of air flow
output_dict = fsolve(fct, x0=x_init, args=fct_arg, full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

m_dot = output_dict[0][0]

ttot_jet = ttot + shaft_power/(m_dot*cp)
ptot_jet = ptot * (1. + fan_eff*(ttot_jet/ttot-1.))**(gam/(gam-1.))
mach_jet = np.sqrt(((ptot_jet/pamb)**((gam-1.)/gam) - 1.) * (2./(gam-1.)))  # adapted nozzle
tamb_jet = ttot_jet / (1.+(0.5*(gam-1.))*mach_jet**2)
vjet = mach_jet * earth.sound_speed(tamb_jet)
fn = m_dot *(vjet - vair)

print("Air flow = ", "%.2f"%m_dot)
print("Mach jet = ", "%.2f"%mach_jet)



