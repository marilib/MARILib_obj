#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

import numpy as np
from tabulate import tabulate

from marilib.utils import unit
from marilib.aircraft.aircraft_root import Arrangement, Aircraft
from marilib.aircraft.requirement import Requirement
from marilib.utils.read_write import MarilibIO
from marilib.aircraft.design import process

from marilib.aircraft.step_mission import StepMission

# Configure airplane arrangement
# ---------------------------------------------------------------------------------------------------------------------
agmt = Arrangement(body_type = "fuselage",           # "fuselage" or "blended"
                   wing_type = "classic",            # "classic" or "blended"
                   wing_attachment = "low",          # "low" or "high"
                   stab_architecture = "classic",    # "classic", "t_tail" or "h_tail"
                   tank_architecture = "wing_box",   # "wing_box", "rear", "piggy_back" or "pods"
                   number_of_engine = "twin",        # "twin", "quadri" or "hexa"
                   nacelle_attachment = "wing",      # "wing", "rear" or "pods"
                   power_architecture = "tf",        # "tf", "tp", "ef", "ep", "pte", , "extf", "exef"
                   power_source = "fuel",            # "fuel", "battery", "fuel_cell"
                   fuel_type = "kerosene")           # "kerosene", "liquid_h2", "Compressed_h2", "battery"

reqs = Requirement(n_pax_ref = 150.,
                   design_range = unit.m_NM(3000.),
                   cruise_mach = 0.78,
                   cruise_altp = unit.m_ft(35000.))

ac = Aircraft("This_plane")     # Instantiate an Aircraft object

ac.factory(agmt, reqs)          # Configure the object according to Arrangement, WARNING : arrangement must not be changed after this line

# overwrite default values for design space graph centering (see below)
ac.power_system.reference_thrust = unit.N_kN(120.)
ac.airframe.wing.area = 122.5


process.mda(ac)                 # Run an MDA on the object (All internal constraints will be solved)



miss = StepMission(ac)

disa = 0.

range = unit.m_NM(3000.)

tow = 77866.
owe = 45000.

altp1 = unit.m_ft(1500.)
cas1 = unit.mps_kt(250.)

altp2 = unit.m_ft(10000.)
cas2 = unit.mps_kt(300.)

cruise_mach = 0.78

vz_mcr = unit.mps_ftpmin(0.)
vz_mcl = unit.mps_ftpmin(300.)

heading = "east"

miss.fly_mission(disa,range,tow,owe,altp1,cas1,altp2,cas2,cruise_mach,vz_mcr,vz_mcl,heading)

print("------------------------------")
table1 = np.array([[t1,unit.NM_m(x1),unit.ft_m(z1),m1] for t1,x1,z1,m1 in zip(miss.flight_profile["data"][:,0],
                                                                              miss.flight_profile["data"][:,1],
                                                                              miss.flight_profile["data"][:,2],
                                                                              miss.flight_profile["data"][:,3])])
k_list = [k for k,t in enumerate(table1[1:,0]) if t==table1[k,0]]
table2 = np.delete(table1,k_list,axis=0)
print(tabulate(table2))

# miss.draw_flight_profile()

profile1 = miss.flight_profile["data"][:,[0,1,2,4,5,7]]
k_list = [k for k,t in enumerate(profile1[1:,0]) if t==profile1[k,0]]
profile2 = np.delete(profile1,k_list,axis=0)

print("------------------------------")
print(tabulate(profile2))

flight_path = miss.fly_this_profile(disa,tow,profile2)

print("------------------------------")
print(miss.flight_profile["data"][0,3] - miss.flight_profile["data"][-1,3])
print(flight_path["data"][0,3]-flight_path["data"][-1,3])

table4 = [[t1,t2,z1,m1,m2,dm] for t1,t2,z1,m1,m2,dm in zip(table2[:,0],flight_path["data"][:,0],
                                                     table2[:,2],
                                                     table2[:,3],flight_path["data"][:,3],
                                                     table2[:,3]-flight_path["data"][:,3])]

# for z1,m1,m2,f1,f2 in zip(miss.flight_profile["data"][:,2],
#                        miss.flight_profile["data"][:,3],flight_path["data"][:,3],
#                        miss.flight_profile["data"][:,9],flight_path["data"][:,9]):
#     print(unit.ft_m(z1),m1,m2,f1,f2)

print(tabulate(table4))


miss.plot_data_dict()
# miss.plot_data_dict_fkey("vz_mcl")
