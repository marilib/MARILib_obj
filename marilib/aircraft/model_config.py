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
        "m_pax_max": ["function", "kg", "Maximum mass allowance per passenger for design (function)"],
        "m_pax_cabin": [80., "kg", "Mean passenger mass in the cabin including its hand luggage"]
    },
    "Fuselage":{
        "forward_limit": [4., "m", "Distance between fuselage nose and forward cabin wall"],
        "wall_thickness": [0.2, "m", "Fuselage wall total tchickness"],
        "tail_cone_ratio": [3.45, "no_dim", "Fuselage tail cone length (evolutive part) over fuselage diameter"],
        "rear_bulkhead_ratio": [1.5, "no_dim", "Distance from rear pressure bulkhead to fuselage end over fuselage diameter"]
    },
    "Wing":{
        "wing_morphing": ["aspect_ratio_driven", "no_dim", "Wing deformation mode, 'aspect_ratio_driven' or 'span_driven'"],
        "aspect_ratio": ["function", "no_dim", "Wing aspect ratio (function)"],
        "sweep25": ["function", "deg", "Wing sweep angle at 25% of the chords of outboard trapezoid (function)"],
        "dihedral": [5., "deg", "Wing dihedral"],
        "hld_type": ["function", "int", "Type of high lift device, from 0 to 10, (function)"]
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
        "fuel_pressure": ["function", "bar", "Maximum over pressure of the fuel in the tank (function)"],
        "shield_parameter": ["function", "bar.l/kg", "Tank structural efficiency (function)"],
        "shield_density": [1750, "kg/m3", "Tank shell material density"],
        "insulation_thickness": [0.06, "m", "Tank insulation thickness"],
        "insulation_density": [100., "kg/m3", "Tank insulation material density"]
    },
    "TankWingPod":{
        "span_ratio": [0.60, "no_dim", "Relative span wise position of the tank"],
        "x_loc_ratio": [0.4, "no_dim", "Fraction of the tank length behind the wing"],
        "length": ["function", "m", "Length of the tank"],
        "width": ["function", "m", "Diameter of the tank"],
        "surface_mass": [10., "kg/m2", "Mass per surface unit of the tank structure"],
        "fuel_pressure": ["function", "bar", "Maximum over pressure of the fuel in the tank (function)"],
        "shield_parameter": ["function", "bar.l/kg", "Tank structural efficiency (function)"],
        "shield_density": [1750, "kg/m3", "Tank shell material density"],
        "insulation_thickness": [0.06, "m", "Tank insulation thickness"],
        "insulation_density": [100., "kg/m3", "Tank insulation material density"]
    },
    "TankPiggyBack":{
        "x_loc_ratio": [0.4, "no_dim", "Fraction of the tank length behind the wing"],
        "length": ["function", "m", "Length of the tank"],
        "width": ["function", "m", "Diameter of the tank"],
        "fuel_pressure": ["function", "bar", "Maximum over pressure of the fuel in the tank (function)"],
        "shield_parameter": ["function", "bar.l/kg", "Tank structural efficiency  (function)"],
        "shield_density": [1750, "kg/m3", "Tank shell material density"],
        "insulation_thickness": [0.06, "m", "Tank insulation thickness"],
        "insulation_density": [100., "kg/m3", "Tank insulation material density"]
    },
    "TankRearFuselage":{
        "length": [10., "m", "Length of the tank"],
        "width_rear_factor": [0.75, "m", "Diameter of the rear pressure bulkhead over fuselage diameter"],
        "fuel_pressure": ["function", "bar", "Maximum over pressure of the fuel in the tank (function)"],
        "shield_parameter": ["function", "bar.l/kg", "Tank structural efficiency  (function)"],
        "shield_density": [1750, "kg/m3", "Tank shell material density"],
        "insulation_thickness": [0.06, "m", "Tank insulation thickness"],
        "insulation_density": [100., "kg/m3", "Tank insulation material density"]
    },
    "LandingGear":{
    },
    "InboardWingMountedNacelle":{
    },
    "OutboardWingMountedNacelle":{
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
        "battery": ["no", "string", "Use of battery, 'yes', 'no'"],
        "battery_density": [2800., "kg/m3", "Battery density"],
        "battery_energy_density": [0.4, "kWh/kg", "Battery energy density"],
        "lto_power": [1., "MW", "Battery power reserve for take off and landing"],
        "lto_time": [0., "min", "Duration of availability of the power reserve"],
        "cruise_energy": [0., "kWh", "Energy booked for cruise"],
        "generator_efficiency": [0.95, "no_dim", "Electric generator efficiency"],
        "generator_pw_density": [10., "kW/kg", "Electric generator power density"],
        "rectifier_efficiency": [0.98, "no_dim", "Rectifier efficiency"],
        "rectifier_pw_density": [20., "kW/kg", "Rectifier power density"],
        "wiring_efficiency": [0.995, "no_dim", "Electric wiring efficiency"],
        "wiring_pw_density": [20., "kW/kg", "Electric wiring power density"],
        "cooling_efficiency": [0.99, "no_dim", "Cooling efficiency, ex: 0.99 means that 1% of the power is used by cooling system"],
        "cooling_pw_density": [20., "kW/kg", "Cooling power density"]
    },
    "SemiEmpiricTf0Nacelle":{
        "eis_date": [2020., "year", "Entry into service date"],
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "vertical_margin": [1., "no_dim", "vertical margin as a fraction of nacelle width"],
        "engine_bpr": ["function", "no_dim", "Reference By Pass Ratio of the engine (function)"],
        "engine_opr": [50., "no_dim", "Reference Overall Pressure Ratio of the engine"],
        "core_thrust_ratio": [0.13, "no_dim", "Reference ratio of the total thrust delivered by the core"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller like fan efficiency Thrust.Speed/shaft_power"]
    },
    "SemiEmpiricTfNacelle":{
        "eis_date": [2020., "year", "Entry into service date"],
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "vertical_margin": [1., "no_dim", "vertical margin as a fraction of nacelle width"],
        "engine_bpr": ["function", "no_dim", "Reference By Pass Ratio of the engine (function)"],
        "engine_opr": [50., "no_dim", "Reference Overall Pressure Ratio of the engine"],
        "core_thrust_ratio": [0.13, "no_dim", "Reference ratio of the total thrust delivered by the core"],
        "hub_width": [0.4, "m", "Fan hub diameter"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller like fan efficiency Thrust.Speed/shaft_power"],
        "fan_efficiency": [0.95, "no_dim", "Classical fan efficiency"]
    },
    "ExergeticTfNacelle":{
        "eis_date": [2020., "year", "Entry into service date"],
        "engine_bpr": [14., "no_dim", "Reference By Pass Ratio of the engine"],
        "engine_fpr": [1.15, "no_dim", "Reference Fan Pressure Ratio of the engine"],
        "engine_lpc_pr": [3., "no_dim", "Reference Low Pressure Compressor Ratio of the engine"],
        "engine_hpc_pr": [14., "no_dim", "Reference High  Pressure Compressor Ratio of the engine"],
        "engine_T4_max": [1700., "K", "Entry turbine maximum temperature"],
        "cooling_flow": [0.1, "kg/s", "Reference cooling flow"],
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "vertical_margin": [1., "no_dim", "vertical margin as a fraction of nacelle width"],
        "hub_width": [0.4, "m", "Fan hub diameter"],
    },
    "SemiEmpiricTpNacelle":{
        "eis_date": [2020., "year", "Entry into service date"],
        "lateral_margin": [1., "no_dim", "Lateral margin as a fraction of nacelle width"],
        "hub_width": [0.2, "m", "Propeller hub diameter"],
        "propeller_efficiency": [0.82, "no_dim", "Propeller efficiency Thrust.Speed/shaft_power"],
        "propeller_disk_load": [3000., "N/m2", "Propeller disk load"]
    },
    "SemiEmpiricEpNacelle":{
        "eis_date": [2020., "year", "Entry into service date"],
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
        "eis_date": [2020., "year", "Entry into service date"],
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
        "bli_effect": ["yes", "string", "Taking into account boundary layer ingestion, 'yes' or 'no'"],
        "hub_width": [0.6, "m", "Fan hub diameter"],
        "lateral_margin": [1.5, "no_dim", "Lateral margin as a fraction of nacelle width"],
        "x_loc_ratio": [0.5, "no_dim", "Fraction of the tank length behind the wing"],
        "specific_nacelle_cost": [0.05, "$/kg", "Specific maintenance cost per trip for tail cone mounted nacelle"]
    },
    "PiggyBackTailConeMountedNacelle":{
        "bli_effect": ["yes", "string", "Taking into account boundary layer ingestion, 'yes' or 'no'"],
        "hub_width": [0.6, "m", "Fan hub diameter"],
        "specific_nacelle_cost": [0.05, "$/kg", "Specific maintenance cost per trip for tail cone mounted nacelle"]
    },
    "BodyTailConeMountedNacelle":{
        "bli_effect": ["yes", "string", "Taking into account boundary layer ingestion, 'yes' or 'no'"],
        "hub_width": [0.4, "m", "Fan hub diameter"],
        "tail_cone_height_ratio": [0.87, "no_dim", "Relative vertical position of the body tail cone"],
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
    "Aerodynamics":{
        "cx_correction": [0., "no_dim", "Drag correction on cx coefficient"],
        "cruise_lodmax": [16., "no_dim", "Assumption on L/D max for some initializations"],
        "hld_conf_clean": [0., "no_dim", "High lift device setting for clean wing, must be 0."],
        "hld_conf_to": [0.30, "no_dim", "High lift device setting for take off, between 0. and 0.5"],
        "hld_conf_ld": [1.00, "no_dim", "High lift device setting for landing, typically 1. for full deflection"]
    },
    "HandlingQuality":{
        "static_stab_margin": [0.05, "no_dim", "CG margin to neutral point"]
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


