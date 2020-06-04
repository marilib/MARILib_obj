#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Avionic & Systems, Air Transport Departement, ENAC
"""

from marilib.utils import unit


model_config = {
    "Cabin":{
        "n_pax_front": ["function", "int", "Number of front seats in economic class (function)"],
        "n_aisle": ["function", "int", "Number of aisle in economic class (function)"],
        "m_pax_nominal": ["function", "kg", "Reference mass allowance per passenger for design (function)"],
        "m_pax_max": ["function", "kg", "Maximum mass allowance per passenger for design (function)"]
    },
    "Fuselage":{
        "forward_limit": [4., "m", "Distance between fuselage nose and forward cabin wall"],
        "wall_thickness": [0.2, "m", "Fuselage wall total tchickness"],
        "tail_cone_ratio": [3.45, "no_dim", "Fuselage tail cone length over fuselage diameter"]
    },
    "Wing":{
        "wing_morphing": ["aspect_ratio_driven", "no_dim", "Wing deformation mode, 'aspect_ratio_driven' or 'span_driven'"],
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
        "span_ratio": [0.60, "no_dim", "Relative span wise position of the tank"],
        "x_loc_ratio": [0.4, "no_dim", "Fraction of the tank length behind the wing"],
        "length": ["function", "m", "Length of the tank"],
        "width": ["function", "m", "Diameter of the tank"],
        "surface_mass": [10., "kg/m2", "Mass per surface unit of the tank structure"],
        "fuel_pressure": [0., "bar", "Maximum over pressure of the fuel in the tank"],
        "shell_parameter": [700, "bar.l/kg", "Tank structural efficiency"],
        "shell_density": [1750, "kg/m3", "Tank shell material density"]
    },
    "TankPiggyBack":{
        "x_loc_ratio": [0.4, "no_dim", "Fraction of the tank length behind the wing"],
        "length": ["function", "m", "Length of the tank"],
        "width": ["function", "m", "Diameter of the tank"],
        "surface_mass": [10., "kg/m2", "Mass per surface unit of the tank structure"],
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
        "wiring_pw_density": [20., "kW/kg", "Electric wiring power density"],
        "cooling_efficiency": [0.99, "no_dim", "Cooling efficiency, ex: 0.99 means that 1% of the power is used by cooling system"],
        "cooling_pw_density": [10., "kW/kg", "Cooling power density"],
        "battery_density": [2800., "kg/m3", "Battery density"],
        "battery_energy_density": [0.4, "kWh/kg", "Battery energy density"]
    },
    "SystemWithFuelCell":{
        "wiring_efficiency": [0.995, "no_dim", "Electric wiring efficiency"],
        "wiring_pw_density": [20., "kW/kg", "Electric wiring power density"],
        "cooling_efficiency": [0.99, "no_dim", "Cooling efficiency, ex: 0.99 means that 1% of the power is used by cooling system"],
        "cooling_pw_density": [10., "kW/kg", "Cooling power density"],
        "fuel_cell_pw_density": [1., "kW/kg", "Fuell cell power density"],
        "fuel_cell_efficiency": [0.5, "no_dim", "Fuell cell conversion efficiency"]
    },
    "SystemPartialTurboElectric":{
        "chain_power": ["function", "kW", "Electric chain power"],
        "generator_efficiency": [0.95, "no_dim", "Electric generator efficiency"],
        "generator_pw_density": [10., "kW/kg", "Electric generator power density"],
        "rectifier_efficiency": [0.98, "no_dim", "Rectifier efficiency"],
        "rectifier_pw_density": [20., "kW/kg", "Rectifier power density"],
        "wiring_efficiency": [0.995, "no_dim", "Electric wiring efficiency"],
        "wiring_pw_density": [20., "kW/kg", "Electric wiring power density"],
        "cooling_efficiency": [0.99, "no_dim", "Cooling efficiency, ex: 0.99 means that 1% of the power is used by cooling system"],
        "cooling_pw_density": [10., "kW/kg", "Cooling power density"],
        "battery_density": [2800., "kg/m3", "Battery density"],
        "battery_energy_density": [0.4, "kWh/kg", "Battery energy density"]
    },
    "SemiEmpiricTf0Nacelle":{
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "vertical_margin": [1., "no_dim", "vertical margin as a fraction of nacelle width"],
        "engine_bpr": ["function", "no_dim", "Reference By Pass Ratio of the engine (function)"],
        "core_thrust_ratio": [0.13, "no_dim", "Reference ratio of the total thrust delivered by the core"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller like fan efficiency Thrust.Speed/shaft_power"]
    },
    "SemiEmpiricTfNacelle":{
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "vertical_margin": [1., "no_dim", "vertical margin as a fraction of nacelle width"],
        "engine_bpr": ["function", "no_dim", "Reference By Pass Ratio of the engine (function)"],
        "core_thrust_ratio": [0.13, "no_dim", "Reference ratio of the total thrust delivered by the core"],
        "hub_width": [0.4, "m", "Fan hub diameter"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller like fan efficiency Thrust.Speed/shaft_power"],
        "fan_efficiency": [0.95, "no_dim", "Classical fan efficiency"]
    },
    "SemiEmpiricTpNacelle":{
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "hub_width": [0.2, "m", "Propeller hub diameter"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller efficiency Thrust.Speed/shaft_power"],
        "propeller_disk_load": [3000., "N/m2", "Propeller disk load"]
    },
    "SemiEmpiricEpNacelle":{
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller efficiency Thrust.Speed/shaft_power"],
        "propeller_disk_load": [3000., "N/m2", "Propeller disk load"],
        "hub_width": [0.2, "m", "Propeller hub diameter"],
        "motor_efficiency": [0.95, "no_dim", "Electric motor efficiency"],
        "motor_pw_density": [10., "kW/kg", "Electric motor power density"],
        "nacelle_pw_density": [10., "kW/kg", "Electric nacelle power density"],
        "controller_efficiency": [0.99, "no_dim", "Electric controller efficiency"],
        "controller_pw_density": [20., "kW/kg", "Electric controller power density"]
    },
    "SemiEmpiricEfNacelle":{
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "vertical_margin": [1., "no_dim", "vertical margin as a fraction of nacelle width"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller like fan efficiency Thrust.Speed/shaft_power"],
        "fan_efficiency": [0.95, "no_dim", "Classical fan efficiency"],
        "hub_width": [0.2, "m", "Fan hub diameter"],
        "motor_efficiency": [0.95, "no_dim", "Electric motor efficiency"],
        "controller_efficiency": [0.99, "no_dim", "Electric controller efficiency"],
        "controller_pw_density": [20., "kW/kg", "Electric controller power density"],
        "nacelle_pw_density": [10., "kW/kg", "Electric nacelle power density"],
        "motor_pw_density": [10., "kW/kg", "Electric motor power density"]
    },
    "PodTailConeMountedNacelle":{
        "bli_effect": ["no", "string", "Taking into account boundary layer ingestion, 'yes' or 'no'"],
        "hub_width": [0.6, "m", "Fan hub diameter"],
        "lateral_margin": [1.5, "no_dim", "Lateral margin as a fraction of nacelle width"],
        "x_loc_ratio": [0.5, "no_dim", "Fraction of the tank length behind the wing"],
        "specific_nacelle_cost": [0.05, "$/kg", "Specific maintenance cost per trip for tail cone mounted nacelle"]
    },
    "FuselageTailConeMountedNacelle":{
        "bli_effect": ["no", "string", "Taking into account boundary layer ingestion, 'yes' or 'no'"],
        "hub_width": [0.4, "m", "Fan hub diameter"],
        "tail_cone_height_ratio": [0.38, "no_dim", "Relative vertical position of the body tail cone"],
        "specific_nacelle_cost": [0.05, "$/kg", "Specific maintenance cost per trip for tail cone mounted nacelle"]
    },
    "AllMissionVarMass":{
        "ktow": [0.9, "no_dim", "Ratio of TOW defining the aircraft weight for mission breguet range"]
    },
    "MissionVarMassGeneric":{
        "holding_time": [30., "min", "Holding duration for reserve fuel evaluation"],
        "reserve_fuel_ratio": ["function", "no_dim", "Fraction of mission fuel for reserve fuel evaluation (function)"],
        "diversion_range": ["function", "NM", "Range of diversion mission for reserve fuel evaluation (function)"]
    },
    "AllMissionIsoMass":{
    },
    "MissionIsoMassGeneric":{
        "holding_time": [30., "min", "Holding duration for reserve fuel evaluation"],
        "reserve_enrg_ratio": ["function", "no_dim", "Fraction of mission fuel for reserve fuel evaluation (function)"],
        "diversion_range": ["function", "NM", "Range of diversion mission for reserve fuel evaluation (function)"]
    },
    "Requirement":{
        "cost_range": ["function", "NM", "Reference range for cost evaluation (function)"]
    },
    "TakeOffReq":{
        "disa": [15., "degK", "Temperature shift for take off evaluation"],
        "altp": [0., "ft", "Altitude for take off evaluation"],
        "kmtow": [1., "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "kvs1g": [1.13, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "s2_min_path": ["function", "%", "Minimum trajectory slope at 35ft for take off (function)"],
        "tofl_req": ["function", "m", "Maximum take off field length required in given conditions (function)"]
    },
    "ApproachReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": [0., "ft", "Altitude for approach speed evaluation"],
        "kmlw": [1., "no_dim", "Ratio of MLW defining the aircraft weight"],
        "kvs1g": [1.23, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "app_speed_req": ["function", "m", "Maximum approach speed required in given conditions (function)"]
    },
    "OeiCeilingReq":{
        "disa": [15., "degK", "Temperature shift for one engine performance evaluation"],
        "altp": ["function", "ft", "Altitude for one engine performance evaluation (function)"],
        "kmtow": [1., "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "rating": ["MCN", "string", "Engine rating for one engine performance evaluation"],
        "speed_mode": ["cas", "string", "Speed mode for one engine performance evaluation, 'cas':constant CAS or 'mach':constant Mach"],
        "path_req": ["function", "%", "Minimum trajectory slope required in given conditions (function)"]
    },
    "MclCeilingReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": ["function", "ft", "Altitude for approach speed evaluation (function)"],
        "mach": ["function", "ft", "Speed defined by a Mach number (function)"],
        "kmtow": [0.97, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "rating": ["MCL", "string", "Engine rating for one engine performance evaluation"],
        "speed_mode": ["cas", "string", "Speed mode for one engine performance evaluation, 'cas':constant CAS or 'mach':constant Mach"],
        "vz_req": ["function", "%", "Minimum vertical speed required in given conditions (function)"]
    },
    "McrCeilingReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": ["function", "ft", "Altitude for approach speed evaluation (function)"],
        "mach": ["function", "ft", "Speed defined by a Mach number (function)"],
        "kmtow": [0.97, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "rating": ["MCR", "string", "Engine rating for one engine performance evaluation"],
        "speed_mode": ["mach", "string", "Speed mode for one engine performance evaluation, 'cas':constant CAS or 'mach':constant Mach"],
        "vz_req": ["function", "%", "Minimum vertical speed required in given conditions (function)"]
    },
    "TtcReq":{
        "disa": [15., "degK", "Temperature shift for approach speed evaluation"],
        "altp": ["function", "ft", "Altitude for approach speed evaluation (function)"],
        "mach": ["function", "ft", "Speed defined by a Mach number (function)"],
        "kmtow": [0.97, "no_dim", "Ratio of MTOW defining the aircraft weight"],
        "altp1": [1500., "ft", "Starting pressure altitude for climb trajectory, typically 1500ft"],
        "cas1": ["function", "kt", "Calibrated Air Speed below altp1 during climb trajectory (function)"],
        "altp2": [10000., "ft", "Transtion pressure altitude from cas1 to cas2, typically 10000ft"],
        "cas2": ["function", "kt", "Calibrated Air Speed above altp2 (function)"],
        "altp": ["function", "ft", "Targetted climb altitude (function)"],
        "ttc_req": [25., "min", "Maximum time to climb required in given conditions"]
    },
    "Economics":{
        "irp": [10., "year", "Interest recovery period"],
        "period": [15., "year", "Utilization period"],
        "interest_rate": [4., "%", "Interest rate"],
        "labor_cost": [120, "$/h", "Labor cost"],
        "utilization": ["function", "int", "Number of flights per year (function)"],
        "fuel_price": [2., "$/gal", "Fuel price"],
        "energy_price": [0.10, "$/kWh", "Energy price"],
        "battery_price": [20., "$/kg", "Battery price"]
    }
}


def get_init(obj,key,val=None):
    cls = obj.__class__.__name__
    if cls=="str":
        dat = model_config[obj][key]
    else:
        dat = model_config[cls][key]
    if dat[0]=="function": return val
    else: return unit.convert_from(dat[1],dat[0])


