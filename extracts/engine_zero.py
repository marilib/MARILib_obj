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


def  design_tf(altp,disa,mach,qf):
    # Design data
    t4_max = 1700.
    t4_fac = {"MTO":1., "MCN":0.95, "MCL":0.93, "MCR":0.90, "FID":0.90}
    opr = 50.
    bpr = 12.
    pw_split = 0.80
    mach_fan = 0.55
    hub_width = 0.2

    eff_fan = 0.95
    eff_compressor = 0.95
    eff_thermal = 0.46
    eff_mechanical = 0.99

    r,gam,Cp,Cv = earth.gas_data()
    fhv = earth.fuel_heat("kerosene")

    # Design conditions
    pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
    ptot = earth.total_pressure(pamb,mach)        # Total pressure at inlet position
    ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position
    vair = mach * earth.sound_speed(tamb)

    # Engine cycle definition
    t4 = t4_max * t4_fac["MCR"]
    pw_fuel = qf*fhv                                                # Fuel power input
    pwu_core = pw_fuel * eff_thermal * eff_mechanical               # Effective usable power

    t3 = ttot*(1.+(opr**((gam-1.)/gam)-1.)/eff_compressor)          # Stagnation temperature after compressors
    q_core = pw_fuel/((t4-t3)*Cp)                                   # Core air flow that ensure targetted T4
    vj_core = np.sqrt(vair**2 + 2.*pwu_core*(1.-pw_split)/q_core)   # Core jet velocity according to power sharing
    fn_core = q_core*(vj_core - vair)                               # Resulting core thrust

    q_cold = q_core * bpr                               # Targetted fan air flow according to BPR
    q_total = q_core  + q_cold                            # Targetted fan air flow according to BPR
    pws_fan = pwu_core * pw_split                       # Fan shaft power according to power sharing
    pwu_fan = pws_fan * eff_fan                         # Util power delivered to the air flow
    vj_fan = np.sqrt(vair**2 + 2.*pwu_fan/q_total)        # Fan jet velocity
    fn_fan = q_cold*(vj_fan - vair)                     # Resulting fan thrust

    fn_ref = fn_core + fn_fan           # Total engine thrust
    sfc = qf/fn_ref                     # Resulting sfc

    # Air vein sizing
    ttot_fan_jet = ttot + pws_fan/(q_total*Cp)                # Stagnation temperature increases due to introduced work
    tstat_fan_jet = ttot_fan_jet - 0.5*vj_fan**2/Cp         # Static temperature
    mach_fan_jet = vj_fan/earth.sound_speed(tstat_fan_jet)  # Fan jet Mach number

    pstat_fan_jet = pamb                                                        # Assuming nozzle is adapted
    ptot_fan_jet = pstat_fan_jet*(ttot_fan_jet/tstat_fan_jet)**(gam/(gam-1))    # Fan jet stagnation pressure
    fpr = ptot_fan_jet/ptot                                                     # Resulting Fan Pressure Ratio

    ttot_core_jet = ttot_fan_jet + pw_fuel/(q_core*Cp)   # Core inlet is behind the fan
    tstat_core_jet = ttot_core_jet - vj_core**2/(2*Cp)   # Core jet stagnation temperature

    pstat_core_jet = pamb                                                           # Assuming nozzle adapted
    ptot_core_jet = pstat_core_jet*(ttot_core_jet/tstat_core_jet)**(gam/(gam-1))    # Core jet stagnation pressure
    mach_core_jet = vj_core/earth.sound_speed(tstat_core_jet)                       # Core jet Mach number

    cq_o_a2 = corrected_air_flow(ptot_core_jet,ttot_core_jet,mach_core_jet)     # Corrected air flow per area in core jet flow
    core_nozzle_area = q_core/cq_o_a2                                           # Required core nozzle area
    core_nozzle_width = np.sqrt(4.*core_nozzle_area/np.pi)                      # Resulting core nozzle diameter

    cq_o_a1 = corrected_air_flow(ptot,ttot,mach_fan)        # Corrected air flow per area in fan inlet flow
    fan_area = q_cold/cq_o_a1                               # Required fan nozzle area
    fan_width = np.sqrt(hub_width**2 + 4.*fan_area/np.pi)   # Resulting fan diameter

    cq_o_a3 = corrected_air_flow(ptot_fan_jet,ttot_fan_jet,mach_fan_jet)    # Corrected air flow per area in fan jet flow
    fan_nozzle_area = q_cold/cq_o_a3                                        # Required fan nozzle area
    fan_nozzle_width = np.sqrt(core_nozzle_width**2 + 4.*fan_area/np.pi)    # Fan nozzle diameter taking into account the core nozzle

    return {"fn":fn_ref, "sfc":sfc, "fpr":fpr,
            "fan_area":fan_area, "fan_width":fan_width,
            "fan_nozzle_area":fan_nozzle_area, "fan_nozzle_width":fan_nozzle_width,
            "mach_core_jet":mach_core_jet, "mach_fan_jet":mach_fan_jet}


# Design point
#--------------------------------------------------------------------------------
g = earth.gravity()
disa = 0.
altp = unit.m_ft(35000.)
mach = 0.78

mass = 77000.*0.97
lod = 17.
fn_req = 0.5*mass*g/lod

def fct(qf,altp,disa,mach):
    dict =  design_tf(altp,disa,mach,qf)
    return fn_req-dict["fn"]

qf_ini = 0.3

output_dict = fsolve(fct, x0=qf_ini, args=(altp,disa,mach), full_output=True)
if (output_dict[2]!=1): raise Exception("Convergence problem")

qf = output_dict[0][0]
dict =  design_tf(altp,disa,mach,qf)


print("qf = ",qf)
print("fn_req = ",fn_req)
print("fn_eff = ",dict["fn"])
print("sfc = ",unit.convert_to("kg/daN/h",dict["sfc"]))
print("")
print("fpr = ",dict["fpr"])
print("mach_core_jet = ",dict["mach_core_jet"])
print("mach_fan_jet = ",dict["mach_fan_jet"])
print("")
print("fan_width = ",dict["fan_width"])
print("fan_nozzle_area = ",dict["fan_nozzle_area"])
print("fan_nozzle_width = ",dict["fan_nozzle_width"])


def  off_design_tf(altp,disa,mach,rating,qfuel,fia,fna):
    # Design data
    t4_max = 1700.
    t4_fac = {"MTO":1., "MCN":0.95, "MCL":0.93, "MCR":0.90, "FID":0.90}
    opr = 50.
    pw_split = 0.80
    mach_fan = 0.55
    hub_width = 0.2

    eff_fan = 0.95
    eff_compressor = 0.95
    eff_thermal = 0.46
    eff_mechanical = 0.99

    def fct(q,q_core,pw_shaft,pamb,Ttot,Vair,fna):
        Vinlet = Vair
        pw_input = eff_fan*pw_shaft
        Vjet = np.sqrt(2.*pw_input/q + Vinlet**2)   # Supposing adapted nozzle
        TtotJet = Ttot + pw_shaft/(q*Cp)            # Stagnation temperature increases due to introduced work
        TstatJet = TtotJet - 0.5*Vjet**2/Cp         # Static temperature
        VsndJet = earth.sound_speed(TstatJet)       # Sound speed at nozzle exhaust
        MachJet = Vjet/VsndJet                      # Mach number at nozzle output, ignoring when Mach > 1
        PtotJet = earth.total_pressure(pamb, MachJet)               # total pressure at nozzle exhaust (P = pamb)
        CQoA1 = corrected_air_flow(PtotJet,TtotJet,MachJet)    # Corrected air flow per area at fan position
        q0 = CQoA1*fna
        q_cold = q - q_core               # Here, it is fan air flow only
        y = q0 - q_cold
        return y

    r,gam,Cp,Cv = earth.gas_data()
    fhv = earth.fuel_heat("kerosene")

    # Design conditions
    pamb,tamb,tstd,dtodz = earth.atmosphere(altp,disa)
    ptot = earth.total_pressure(pamb,mach)        # Total pressure at inlet position
    ttot = earth.total_temperature(tamb,mach)     # Total temperature at inlet position
    vair = mach * earth.sound_speed(tamb)

    # Engine cycle definition
    t4 = t4_max * t4_fac[rating]
    pw_fuel = qfuel*fhv                                                # Fuel power input
    pwu_core = pw_fuel * eff_thermal * eff_mechanical               # Effective usable power

    t3 = ttot*(1.+(opr**((gam-1.)/gam)-1.)/eff_compressor)          # Stagnation temperature after compressors
    q_core = pw_fuel/((t4-t3)*Cp)                                   # Core air flow that ensure targetted T4
    vj_core = np.sqrt(vair**2 + 2.*pwu_core*(1.-pw_split)/q_core)   # Core jet velocity according to power sharing
    fn_core = q_core*(vj_core - vair)                               # Resulting core thrust

    pws_fan = pwu_core * pw_split                       # Fan shaft power according to power sharing

    fct_arg = (q_core,pws_fan,pamb,ttot,vair,fna)

    cq_o_a0 = corrected_air_flow(ptot,ttot,mach)       # Corrected air flow per area at fan position
    q0init = cq_o_a0*(0.25*np.pi*fia**2)

    # Computation of the air flow swallowed by the inlet
    output_dict = fsolve(fct, x0=q0init, args=fct_arg, full_output=True)
    if (output_dict[2]!=1): raise Exception("Convergence problem")

    q_total = output_dict[0][0]
    q_cold = q_total - q_core

    bpr = q_cold / q_core

    vi_fan = vair
    pw_input = eff_fan*pws_fan
    vj_fan = np.sqrt(2.*pw_input/q_total + vi_fan**2)
    fn_fan = q_cold*(vj_fan - vi_fan)

    fn = fn_fan + fn_core

    sfc = qfuel/fn

    ttot_fan_jet = ttot + pws_fan/(q_total*Cp)                # Stagnation temperature increases due to introduced work
    tstat_fan_jet = ttot_fan_jet - 0.5*vj_fan**2/Cp         # Static temperature
    mach_fan_jet = vj_fan/earth.sound_speed(tstat_fan_jet)  # Fan jet Mach number

    pstat_fan_jet = pamb                                                        # Assuming nozzle is adapted
    ptot_fan_jet = pstat_fan_jet*(ttot_fan_jet/tstat_fan_jet)**(gam/(gam-1))    # Fan jet stagnation pressure
    fpr = ptot_fan_jet/ptot                                                     # Resulting Fan Pressure Ratio

    ttot_core_jet = ttot_fan_jet + pw_fuel/(q_core*Cp)   # Core inlet is behind the fan
    tstat_core_jet = ttot_core_jet - vj_core**2/(2*Cp)   # Core jet stagnation temperature
    mach_core_jet = vj_core/earth.sound_speed(tstat_core_jet)                       # Core jet Mach number

    return {"fn":fn, "sfc":sfc, "fpr":fpr, "bpr":bpr, "mach_core_jet":mach_core_jet, "mach_fan_jet":mach_fan_jet}


fia = dict["fan_area"]
fna = dict["fan_nozzle_area"]

disa = 0.
altp = unit.m_ft(35000.)
mach = 0.78
rating = "MCR"
qfuel = 0.9*qf
dict1 =  off_design_tf(altp,disa,mach,rating,qfuel,fia,fna)

print("")
print("")
print("qf = ",qfuel)
print("fn_eff = ",dict1["fn"])
print("sfc = ",unit.convert_to("kg/daN/h",dict1["sfc"]))
print("fpr = ",dict1["fpr"])
print("bpr = ",dict1["bpr"])
print("mach_core_jet = ",dict1["mach_core_jet"])
print("mach_fan_jet = ",dict1["mach_fan_jet"])

