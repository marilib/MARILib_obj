#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

from marilib.utils import unit


def get_init(obj,key,val=None):
    cls = obj.__class__.__name__
    dat = mc[cls][key]
    if dat[0]=="function":
        return unit.convert_from(dat[1],val)
    else:
        return unit.convert_from(dat[1],dat[0])


mc = {
    "Wing":{
        "aspect_ratio": [9, "no_dim", "Wing aspect ratio"],
        "hld_type": [9, "int", "Type of high lift device, from 0 to 10"]
    },
    "VtpClassic":{
        "aspect_ratio": [1.7, "no_dim", "Vertical tail aspect ratio"],
        "taper_ratio": [0.4, "no_dim", "Vertical tail taper ratio"],
        "toc": [0.1, "no_dim", "Thickness to chord ratio of the vertical tail"],
        "thrust_volume_factor": [0.4, "no_dim", "Volume coefficient of the vertical stabilizer according to engine failure"],
        "wing_volume_factor": [0.07, "no_dim", "Volume coefficient of the vertical stabilizer according to wing"],
        "anchor_ratio": [0.85, "no_dim", "Relative position of the reference point of the stabilization surface"]
    },
    "VtpTtail":{
        "aspect_ratio": [1.2, "no_dim", "Vertical tail aspect ratio"],
        "taper_ratio": [0.8, "no_dim", "Vertical tail taper ratio"],
        "toc": [0.1, "no_dim", "Thickness to chord ratio of the vertical tail"],
        "thrust_volume_factor": [0.4, "no_dim", "Volume coefficient of the vertical stabilizer according to engine failure"],
        "wing_volume_factor": [0.07, "no_dim", "Volume coefficient of the vertical stabilizer according to wing"],
        "anchor_ratio": [0.85, "no_dim", "Relative position of the reference point of the stabilization surface"]
    },
    "VtpHtail":{
        "aspect_ratio": [1.5, "no_dim", "Vertical tail aspect ratio"],
        "taper_ratio": [0.4, "no_dim", "Vertical tail taper ratio"],
        "toc": [0.1, "no_dim", "Thickness to chord ratio of the vertical tail"],
        "thrust_volume_factor": [0.4, "no_dim", "Volume coefficient of the vertical stabilizer according to engine failure"],
        "wing_volume_factor": [0.07, "no_dim", "Volume coefficient of the vertical stabilizer according to wing"]
    },
    "HtpClassic":{
        "aspect_ratio": [5.0, "no_dim", "Horizontal tail aspect ratio"],
        "taper_ratio": [0.35, "no_dim", "Horizontal tail taper ratio"],
        "toc": [0.1, "no_dim", "Thickness to chord ratio of the Horizontal tail"],
        "dihedral": [5, "deg", "Dihedral angle of the horizontal tail"],
        "volume_factor": [0.94, "no_dim", "Volume coefficient of the vertical stabilizer according to engine failure"]
    },
    "HtpTtail":{
        "aspect_ratio": [5.0, "no_dim", "Horizontal tail aspect ratio"],
        "taper_ratio": [0.35, "no_dim", "Horizontal tail taper ratio"],
        "toc": [0.1, "no_dim", "Thickness to chord ratio of the Horizontal tail"],
        "dihedral": [5, "deg", "Dihedral angle of the horizontal tail"],
        "volume_factor": [0.94, "no_dim", "Volume coefficient of the vertical stabilizer according to engine failure"]
    },
    "HtpHtail":{
        "aspect_ratio": [5.0, "no_dim", "Horizontal tail aspect ratio"],
        "taper_ratio": [0.35, "no_dim", "Horizontal tail taper ratio"],
        "toc": [0.1, "no_dim", "Thickness to chord ratio of the Horizontal tail"],
        "dihedral": [5, "deg", "Dihedral angle of the horizontal tail"],
        "volume_factor": [0.94, "no_dim", "Volume coefficient of the vertical stabilizer according to engine failure"]
    },
    "TankWingBox":{
        "fuel_pressure": [0, "bar", "Maximum over pressure of the fuel in the tank"],
        "shell_parameter": [700, "bar.l/kg", "Tank structural efficiency"],
        "shell_density": [1750, "kg/m3", "Tank shell material density"]
    },
    "TankWingPod":{
        "fuel_pressure": [0, "bar", "Maximum over pressure of the fuel in the tank"],
        "shell_parameter": [700, "bar.l/kg", "Tank structural efficiency"],
        "shell_density": [1750, "kg/m3", "Tank shell material density"]
    },
    "TankPiggyBack":{
        "fuel_pressure": [0, "bar", "Maximum over pressure of the fuel in the tank"],
        "shell_parameter": [700, "bar.l/kg", "Tank structural efficiency"],
        "shell_density": [1750, "kg/m3", "Tank shell material density"]
    },
    "LandingGear":{
    },
    "InboradWingMountedNacelle":{
    },
    "OutboradWingMountedNacelle":{
    },
    "RearFuselageMountedNacelle":{
    },
    "System":{
    },
    "SystemWithBattery":{
        "wiring_efficiency": [0.995, "no_dim", "Electric wiring efficiency"],
        "wiring_pw_density": [20, "kW/kg", "Electric wiring power density"],
        "cooling_pw_density": [10, "kW/kg", "Cooling power density"],
        "Battery_density": [2800., "kg/m3", "Battery density"],
        "battery_energy_density": [0.4, "kW/kg", "Battery energy density"]
    },
    "SystemWithFuelCell":{
        "wiring_efficiency": [0.995, "no_dim", "Electric wiring efficiency"],
        "wiring_pw_density": [20, "kW/kg", "Electric wiring power density"],
        "cooling_pw_density": [10, "kW/kg", "Cooling power density"],
        "fuel_cell_pw_density": [1., "kW/kg", "Fuell cell power density"],
        "fuel_cell_efficiency": [0.5, "no_dim", "Fuell cell conversion efficiency"]
    },
    "SemiEmpiricTfNacelle":{
        "engine_bpr": ["function", "no_dim", "Reference By Pass Ratio of the engine"],
        "core_thrust_ratio": [0.13, "no_dim", "Reference ratio of the total thrust delivered by the core"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller like fan efficiency Thrust.Speed/shaft_power"]
    },
    "SemiEmpiricTpNacelle":{
        "propeller_efficiency": [0.82, "no_dim", "Propeller efficiency Thrust.Speed/shaft_power"],
        "propeller_disk_load": [3000., "N/m2", "Propeller disk load"]
    },
    "SemiEmpiricEpNacelle":{
        "propeller_efficiency": [0.82, "no_dim", "Propeller efficiency Thrust.Speed/shaft_power"],
        "propeller_disk_load": [3000., "N/m2", "Propeller disk load"],
        "hub_width": [0.2, "m", "Propeller hub diameter"],
        "motor_efficiency": [0.95, "no_dim", "Electric motor efficiency"],
        "controller_efficiency": [0.99, "no_dim", "Electric controller efficiency"],
        "controller_pw_density": [20.e3, "kW/kg", "Electric controller power density"],
        "nacelle_pw_density": [5.e3, "kW/kg", "Electric nacelle power density"],
        "motor_pw_density": [10.e3, "kW/kg", "Electric motor power density"]
    },
    "SemiEmpiricEfNacelle":{
        "propeller_efficiency": [0.82, "no_dim", "Propeller like fan efficiency Thrust.Speed/shaft_power"],
        "fan_efficiency": [0.82, "no_dim", "Classical fan efficiency"],
        "hub_width": [0.2, "m", "Propeller hub diameter"],
        "motor_efficiency": [0.95, "no_dim", "Electric motor efficiency"],
        "controller_efficiency": [0.99, "no_dim", "Electric controller efficiency"],
        "controller_pw_density": [20.e3, "kW/kg", "Electric controller power density"],
        "nacelle_pw_density": [5.e3, "kW/kg", "Electric nacelle power density"],
        "motor_pw_density": [10.e3, "kW/kg", "Electric motor power density"]
    },
    "AllMissionVarMass":{
        "ktow": [0.9, "no_dim", "Ratio of TOW defining the aircraft weight for mission breguet range"]
    },
    "MissionVarMassGeneric":{
        "holding_time": [30., "min", "Holding duration for reserve fuel evaluation"],
        "reserve_fuel_ratio": ["function", "no_dim", "Fraction of mission fuel for reserve fuel evaluation"],
        "diversion_range": ["function", "NM", "Range of diversion mission for reserve fuel evaluation"]
    },
    "AllMissionIsoMass":{
    },
    "MissionIsoMassGeneric":{
        "holding_time": [30., "min", "Holding duration for reserve fuel evaluation"],
        "reserve_fuel_ratio": ["function", "no_dim", "Fraction of mission fuel for reserve fuel evaluation"],
        "diversion_range": ["function", "NM", "Range of diversion mission for reserve fuel evaluation"]
    },
    "TakeOffReq":{
        "disa": [15., "degK", "Temperature shift for take off evaluation"],
        "altp": [0., "ft", "Altitude for take off evaluation"],
        "kmtow": [1., "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "kvs1g": [1.13, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "s2_min_path": ["function", "%", "Minimum trajectory slope at 35ft for take off"],
        "tofl_req": ["function", "m", "Maximum take off field length required in given conditions"]
    },
    "ApproachReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": [0., "ft", "Altitude for approach speed evaluation"],
        "kmlw": [1., "no_dim", "Ratio of MLW defining the aircraft weight"],
        "kvs1g": [1.23, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "app_speed_req": ["function", "m", "Maximum approach speed required in given conditions"]
    },
    "OeiCeilingReq":{
        "disa": [15., "degK", "Temperature shift for one engine performance evaluation"],
        "altp": ["function", "ft", "Altitude for one engine performance evaluation"],
        "kmtow": [1., "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "rating": ["MCN", "string", "Engine rating for one engine performance evaluation"],
        "speed_mode": ["cas", "string", "Speed mode for one engine performance evaluation, 'cas':constant CAS or 'mach':constant Mach"],
        "path_req": ["function", "%", "Minimum trajectory slope required in given conditions"]
    },
    "MclCeilingReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": ["function", "ft", "Altitude for approach speed evaluation"],
        "mach": ["function", "ft", "Speed defined by a Mach number"],
        "kmtow": [0.97, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "rating": ["MCL", "string", "Engine rating for one engine performance evaluation"],
        "speed_mode": ["cas", "string", "Speed mode for one engine performance evaluation, 'cas':constant CAS or 'mach':constant Mach"],
        "vz_req": ["function", "%", "Minimum vertical speed required in given conditions"]
    },
    "McrCeilingReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": ["function", "ft", "Altitude for approach speed evaluation"],
        "mach": ["function", "ft", "Speed defined by a Mach number"],
        "kmtow": [0.97, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "rating": ["MCR", "string", "Engine rating for one engine performance evaluation"],
        "speed_mode": ["mach", "string", "Speed mode for one engine performance evaluation, 'cas':constant CAS or 'mach':constant Mach"],
        "vz_req": ["function", "%", "Minimum vertical speed required in given conditions"]
    },
    "TtcReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": ["function", "ft", "Altitude for approach speed evaluation"],
        "mach": ["function", "ft", "Speed defined by a Mach number"],
        "kmtow": [0.97, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "altp1": [1500., "ft", "Starting pressure altitude for climb trajectory, typically 1500ft"],
        "cas1": ["function", "kt", "Calibrated Air Speed below altp1 during climb trajectory"],
        "altp2": [10000., "ft", "Transtion pressure altitude from cas1 to cas2, typically 10000ft"],
        "cas2": ["function", "kt", "Calibrated Air Speed above altp2"],
        "altp": ["function", "ft", "Targetted climb altitude"],
        "ttc_req": [25., "min", "Maximum time to climb required in given conditions"]
    }
}

