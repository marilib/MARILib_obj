#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""


from marilib.aircraft.aircraft_root import Arrangement, Aircraft



ac = Aircraft("my_plane")


data = [["Thrust", "string", "%20s", "ac.name"],
        ["Design range", "km", "%8.0f", "ac.requirement.design_range"],
        ["Cruise Mach", "mach", "%8.0f", "ac.requirement.cruise_mach"],
        ["Nominal seat count", "int", "%8.0f", "ac.airframe.cabin.n_pax_ref"],
        ["Number of front seats", "int", "%8.0f", "ac.airframe.cabin.n_pax_front"],
        ["Fuselage length", "m", "%8.2f", "ac.airframe.body.length"],
        ["Fuselage width", "m", "%8.2f", "ac.airframe.body.width"],
        ["Wing area", "m", "%8.2f", "ac.airframe.wing.area"],
        ["Wing aspect ratio", "m", "%8.1f", "ac.airframe.wing.aspect_ratio"],
        ["Tank length", "m", "%8.2f", "ac.airframe.tank.length"],
        ["Tank diameter", "m", "%8.2f", "ac.airframe.tank.width"],
        ["Tank volume", "m3", "%8.2f", "ac.airframe.tank.max_volume"],
        ["Maximum thrust", "kN", "%8.2f", "ac.power_system.reference_thrust"],
        ["MTOW", "kg", "%8.0f", "ac.weight_cg.mtow"],
        ["OEW", "kg", "%8.0f", "ac.weight_cg.owe"],
        ["MFW", "kg", "%8.0f", "ac.weight_cg.owe"],
        ["Maximum range", "km", "%8.0f", "ac.performance.mission.max_fuel.range"],
        ["L/D start of cruise", "no_dim", "%8.2f", "ac.performance.mission.crz_lod"],
        ["Payload at maximum range", "kg", "%8.0f", "ac.performance.mission.max_fuel.payload"],
        ["Take off field length required", "m", "%8.0f", "ac.performance.take_off.tofl_req"],
        ["Take off field length effective", "m", "%8.0f", "ac.performance.take_off.tofl_eff"],
        ["Approach speed required", "kt", "%8.1f", "ac.performance.approach.app_speed_req"],
        ["Approach speed effective", "kt", "%8.1f", "ac.performance.approach.app_speed_eff"],
        ["Vertical speed required MCR rating", "ft/min", "%8.0f", "ac.performance.mcr_ceiling.vz_req"],
        ["Vertical speed effective MCR rating", "ft/min", "%8.0f", "ac.performance.mcr_ceiling.vz_eff"],
        ["Vertical speed required MCL rating", "ft/min", "%8.0f", "ac.performance.mcl_ceiling.vz_req"],
        ["Vertical speed effective MCL rating", "ft/min", "%8.0f", "ac.performance.mcl_ceiling.vz_eff"],
        ["One engine ceiling required altitude", "ft", "%8.0f", "ac.performance.oei_ceiling.altp"],
        ["One engine ceiling required minimum path", "%", "%8.1f", "ac.performance.oei_ceiling.path_req"],
        ["One engine ceiling effective path", "%", "%8.1f", "ac.performance.oei_ceiling.path_eff"],
        ["Time to climb required", "min", "%8.2f", "ac.performance.oei_ceiling.ttc_req"],
        ["Time to climb effective", "min", "%8.2f", "ac.performance.oei_ceiling.ttc_eff"]]







