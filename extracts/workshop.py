#!/usr/bin/env python3
"""
@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC

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


def fct(q,pw_shaft,pamb,Ttot,Vair):
    Vinlet = Vair
    pw_input = eff_fan*pw_shaft
    Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)   # Supposing adapted nozzle
    TtotJet = Ttot + pw_shaft/(q*Cp)            # Stagnation temperature increases due to introduced work
    TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
    VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
    MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
    PtotJet = earth.total_pressure(pamb, MachJet)               # total pressure at nozzle exhaust (P = pamb)
    CQoA1 = corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
    q0 = CQoA1*nozzle_area
    qf = q * bpr/(1.+bpr)               # Here, it is fan air flow only
    y = q0 - qf
    return y


# Data
#--------------------------------------------------------------------------------
t4 = 1700.
opr = 50.
bpr = 12.
pw_split = 0.90
mach_fan = 0.55
hub_width = 0.2

eff_fan = 0.95
eff_compressor = 0.95
eff_thermal = 0.46
eff_mechanical = 0.99

g = earth.gravity()
r,gam,Cp,Cv = earth.gas_data()
fhv = earth.fuel_heat("kerosene")

# Design point
#--------------------------------------------------------------------------------
disa = 0.
altp = unit.m_ft(35000.)
mach = 0.78

mass = 77000.*0.97
lod = 17.
fn_req = 0.5*mass*g/lod

pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
ptot = earth.total_pressure(pamb,mach)        # Total pressure at inlet position
ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position
vair = mach * earth.sound_speed(tamb)


qf = 0.329

pw_fuel = qf*fhv
pwu_core = pw_fuel * eff_thermal * eff_mechanical

t3 = ttot*(1.+(opr**((gam-1.)/gam)-1.)/eff_compressor)
q_core = pw_fuel/((t4-t3)*Cp)
vj_core = np.sqrt(vair**2 + 2.*pwu_core*(1.-pw_split)/q_core)
fn_core = q_core*(vj_core - vair)

q_fan = q_core * bpr
pws_fan = pwu_core * pw_split
pwu_fan = pws_fan * eff_fan
vj_fan = np.sqrt(vair**2 + 2.*pwu_fan/q_fan)
fn_fan = q_fan*(vj_fan - vair)

fn_eff = fn_core + fn_fan
sfc = qf/fn_eff


ttot_fan_jet = ttot + pws_fan/(q_fan*Cp)   # Stagnation pressure increases due to introduced work
tstat_fan_jet = ttot_fan_jet - 0.5*vj_fan**2/Cp     # static temperature
mach_fan_jet = vj_fan/earth.sound_speed(tstat_fan_jet)                  # Mach number at nozzle output

pstat_fan_jet = pamb                                   # Assuming nozzle adapted
ptot_fan_jet = pstat_fan_jet*(ttot_fan_jet/tstat_fan_jet)**(gam/(gam-1))
fpr = ptot_fan_jet/ptot

ttot_core_jet = ttot_fan_jet + pw_fuel/(q_core*Cp)   # Core is working inside the fan output flow
tstat_core_jet = ttot_core_jet - vj_core**2/(2*Cp)

pstat_core_jet = pamb                                   # Assuming nozzle adapted
ptot_core_jet = pstat_core_jet*(ttot_core_jet/tstat_core_jet)**(gam/(gam-1))
mach_core_jet = vj_core/earth.sound_speed(tstat_core_jet)

cq_o_a2 = corrected_air_flow(ptot_core_jet,ttot_core_jet,mach_core_jet)       # Corrected air flow per area at fan position
core_nozzle_area = q_core/cq_o_a2
core_nozzle_width = np.sqrt(4.*core_nozzle_area/np.pi)        # Fan diameter

cq_o_a1 = corrected_air_flow(ptot,ttot,mach_fan)        # Corrected air flow per area at fan position
fan_area = q_fan/cq_o_a1     # Fan area around the hub
fan_width = np.sqrt(hub_width**2 + 4.*fan_area/np.pi)        # Fan diameter

ttot_fan_jet = ttot + pws_fan/(q_fan*Cp)           # Stagnation pressure increases due to introduced work
tstat = ttot_fan_jet - 0.5*vj_fan**2/Cp            # static temperature
mach_fan_jet = vj_fan/np.sqrt(gam*r*tstat)              # Mach number at nozzle output
ptot_fan_jet = earth.total_pressure(pamb, mach_fan_jet)      # total pressure at nozzle exhaust (P = Pamb)

cq_o_a3 = corrected_air_flow(ptot_fan_jet,ttot_fan_jet,mach_fan_jet)    # Corrected air flow per area at nozzle output
fan_nozzle_area = q_fan/cq_o_a3                                      # Fan nozzle area around the core nozzle
fan_nozzle_width = np.sqrt(core_nozzle_width**2 + 4.*fan_area/np.pi)        # Fan diameter




print("fn_req = ",fn_req)
print("fn_eff = ",fn_eff)
print("sfc = ",unit.convert_to("kg/daN/h",sfc))

print("fpr = ",fpr)
print("fan_width = ",fan_width)
print("fan_nozzle_width = ",fan_nozzle_width)
print("core_nozzle_width = ",core_nozzle_width)

