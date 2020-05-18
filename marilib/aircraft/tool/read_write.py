#!/usr/bin/env python3
"""
Fonctionalities to read and write Marilib object in human readable format or binary format.
The class :class:`MarilibIO` contains several methods:

* to convert an complex object into a human readable string (JSON format)
* to write this JSON readable format in a text file
* to write an exact copy of the object in a binary file (pickle)
* to load an object from a binary file (pickle)

Use case::

    my_plane = Aircraft("This_plane")
    # ... Do some stuff on my_plane

    io = MarilibIO()
    print(io.to_string(my_plane)) # print a veeery long string
    io.to_json_file(my_plane,"my_plane")  # write into the text file "my_plane.json"
    io.to_binary_file(my_plane, "my_plane")  # save into the binary file "my_plane.pkl"

.. note::
    The JSON format is very convenient to explore the numerous variable inside a complex object such as
    :class:`marilib.aircraft.aircraft_root.Aircraft` but it is not an exact copy of the object, which is why you will not
    find a `read_json_file` method. To save your results, you shoudl rather use :meth:`MarilibIO.to_binary_file`
    and :meth:`MarilibIO.from_binary_file`.

:author: DRUOT Thierry, MONROLIN Nicolas

"""

import copy
import numpy as np
import json as json
import pickle as pickle
import re

from marilib.context.unit import convert_to

STANDARD_FORMAT = 6

DATA_DICT = {
    "body_type": {"unit":"string", "mag":8, "txt":"Type of main body, 'fuselage' or 'blended'"},
    "wing_type": {"unit":"string", "mag":6, "txt":"Type of lifting body, 'classic' or 'blended'"},
    "wing_attachment": {"unit":"string", "mag":4, "txt":"Position of wing attachment, 'low' or 'high'"},
    "stab_architecture": {"unit":"string", "mag":6, "txt":"Type of stabilizers, 'classic', 't_tail' or 'h_tail'"},
    "tank_architecture": {"unit":"string", "mag":10, "txt":"Type of tank, 'wing_box', 'piggy_back' or 'pods'"},
    "number_of_engine": {"unit":"string", "mag":6, "txt":"Number of engine, 'twin' or 'quadri'"},
    "nacelle_attachment": {"unit":"string", "mag":4, "txt":"Position of nacelle attachment, 'wing', 'pod' or 'rear'"},
    "power_architecture": {"unit":"string", "mag":4, "txt":"Propulsive architecture, 'tf', 'extf', 'ef'"},
    "energy_source": {"unit":"string", "mag":9, "txt":"Type energy storage, 'kerosene' or 'battery'"},
    "width": {"unit":"m", "mag":1e0, "txt":"Width of the component"},
    "height": {"unit":"m", "mag":1e0, "txt":"Height of the component"},
    "length": {"unit":"m", "mag":1e1, "txt":"Length of the component"},
    "tail_cone_length": {"unit":"m", "mag":1e1, "txt":"Length of the tapered rear part of the main body"},
    "projected_area": {"unit":"m2", "mag":1e2, "txt":"Cabin efficiency parameter ~ cabin length times fuselage width"},
    "frame_origin": {"unit":"m", "mag":1e1, "txt":"Position of the reference point of the component in the assembly"},
    "mass": {"unit":"kg", "mag":1e3, "txt":"Masse of the component"},
    "cg": {"unit":"m", "mag":1e1, "txt":"Position of the center of gravity of the component in the assembly"},
    "inertia_tensor": {"unit":"kg.m2", "mag":1e5, "txt":"Inertia tensor of the component in its reference point"},
    "gross_wet_area": {"unit":"m2", "mag":1e1, "txt":"Gross wetted area of the component"},
    "net_wet_area": {"unit":"m2", "mag":1e1, "txt":"Net wetted area of the component"},
    "aero_length": {"unit":"m", "mag":1e1, "txt":"Characteristic length of the component for friction drag estimation"},
    "form_factor": {"unit":"no_dim", "mag":1e0, "txt":"Form factor of the coponent for form and friction drag estimation"},
    "co2_metric_area": {"unit":"m2", "mag":1e2, "txt":"Refernce cabin area for Fuel Efficiency Metric estimation"},
    "m_furnishing": {"unit":"kg", "mag":1e3, "txt":"Furnishing mass (seats, monuments, toilets, ...)"},
    "m_op_item": {"unit":"kg", "mag":1e3, "txt":"Operator items mass (crews, water, food, ...)"},
    "nominal_payload": {"unit":"kg", "mag":1e4, "txt":"Reference payload for design"},
    "maximum_payload": {"unit":"kg", "mag":1e4, "txt":"Maximum payload for design"},
    "cg_furnishing": {"unit":"m", "mag":1e1, "txt":"Position of the CG of furnishing in the assembly"},
    "cg_op_item": {"unit":"m", "mag":1e1, "txt":"Position of the CG of operator items in the assembly"},
    "max_fwd_req_cg": {"unit":"m", "mag":1e1, "txt":"Maximum forward position of the payload"},
    "max_fwd_mass": {"unit":"kg", "mag":1e4, "txt":"Mass corresponding to maximum forward position of the payload"},
    "max_bwd_req_cg": {"unit":"m", "mag":1e1, "txt":"Maximum backward position of the payload"},
    "max_bwd_mass": {"unit":"kg", "mag":1e4, "txt":"Mass corresponding to maximum backward position of the payload"},
    "n_pax_ref": {"unit":"int", "mag":1e2, "txt":"Reference number of passenger for design"},
    "n_pax_front": {"unit":"int", "mag":1e2, "txt":"Number of front seats in economy class"},
    "n_aisle": {"unit":"int", "mag":1e0, "txt":"Number of aisle in economy class"},
    "m_pax_nominal": {"unit":"int", "mag":1e2, "txt":"Reference mass allowance per passenger for design"},
    "m_pax_max": {"unit":"int", "mag":1e2, "txt":"Maximum mass allowance per passenger for design"},
    "cruise_disa": {"unit":"degK", "mag":1e0, "txt":"Temperature shift versus ISA in cruise for design"},
    "cruise_altp": {"unit":"ft", "mag":1e4, "txt":"Mean cruise altitude for design"},
    "cruise_mach": {"unit":"mach", "mag":1e0, "txt":"Cruise Mach number for design"},
    "design_range": {"unit":"NM", "mag":1e3, "txt":"Design range"},
    "cost_range": {"unit":"NM", "mag":1e2, "txt":"Reference range for cost evaluation"},
    "area": {"unit":"m2", "mag":1e2, "txt":"Geometric reference area of the lifting surface"},
    "span": {"unit":"m", "mag":1e1, "txt":"Spanwise dimension of the lifting surface"},
    "aspect_ratio": {"unit":"no_dim", "mag":1e1, "txt":"Aspect ratio of the lifting surface"},
    "taper_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Taper ratio of the lifting surface"},
    "sweep0": {"unit":"deg", "mag":1e0, "txt":"Leading edge sweep angle of the lifting surface"},
    "sweep25": {"unit":"deg", "mag":1e0, "txt":"Sweep angle at 25% of the chords of the lifting surface"},
    "sweep100": {"unit":"deg", "mag":1e0, "txt":"Trailing edge sweep angle of the lifting surface"},
    "dihedral": {"unit":"deg", "mag":1e0, "txt":"Dihedral angle of the lifting surface"},
    "setting": {"unit":"deg", "mag":1e0, "txt":"Setting angle of the lifting surface"},
    "hld_type": {"unit":"int", "mag":1e0, "txt":"Type of high lift devices (0,1,2,....,10)"},
    "induced_drag_factor": {"unit":"no_dim", "mag":1e0, "txt":"Inverse of Oswald factor of the lifting surface"},
    "axe_loc": {"unit":"m", "mag":1e1, "txt":"Position of the central chord of the lifting surface"},
    "axe_toc": {"unit":"no_dim", "mag":1e0, "txt":"Thickness to chord ratio of the central chord of the lifting surface"},
    "axe_c": {"unit":"m", "mag":1e1, "txt":"Central chord length of the lifting surface"},
    "root_loc": {"unit":"m", "mag":1e1, "txt":"Position of the root chord of the lifting surface"},
    "root_toc": {"unit":"no_dim", "mag":1e0, "txt":"Thickness to chord ratio of the root chord of the lifting surface"},
    "root_c": {"unit":"m", "mag":1e1, "txt":"Root chord length of the lifting surface"},
    "kink_loc": {"unit":"m", "mag":1e1, "txt":"Position of the kink chord of the lifting surface"},
    "kink_toc": {"unit":"no_dim", "mag":1e0, "txt":"Thickness to chord ratio of the kink chord of the lifting surface"},
    "kink_c": {"unit":"m", "mag":1e1, "txt":"Kink chord length of the lifting surface"},
    "tip_loc": {"unit":"m", "mag":1e1, "txt":"Position of the tip chord of the lifting surface"},
    "tip_toc": {"unit":"no_dim", "mag":1e0, "txt":"Thickness to chord ratio of the tip chord of the lifting surface"},
    "tip_c": {"unit":"m", "mag":1e1, "txt":"Tip chord length of the lifting surface"},
    "mac_loc": {"unit":"m", "mag":1e1, "txt":"Position of the mean aerodynamic chord of the lifting surface"},
    "mac": {"unit":"m", "mag":1e1, "txt":"Mean aerodynamic chord length of the lifting surface"},
    "toc": {"unit":"no_dim", "mag":1e0, "txt":"Thickness to chord ratio of the lifting surface"},
    "volume_factor": {"unit":"no_dim", "mag":1e0, "txt":"Volume coefficient of the horizontal stabilizer"},
    "thrust_volume_factor": {"unit":"no_dim", "mag":1e0, "txt":"Volume coefficient of the vertical stabilizer according to engine failure"},
    "wing_volume_factor": {"unit":"no_dim", "mag":1e0, "txt":"Volume coefficient of the vertical stabilizer according to wing"},
    "anchor_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Relative position of the reference point of the stabilization surface"},
    "lever_arm": {"unit":"m", "mag":1e1, "txt":"Lever arm of the stabilization surface, 25%cma wing to 25% cma surface"},
    "fuel_density": {"unit":"kg/m3", "mag":1e3, "txt":"Density of the used storage medium"},
    "shell_parameter": {"unit":"bar.l/kg", "mag":1e2, "txt":"Tank structural efficiency"},
    "shell_density": {"unit":"kg/m3", "mag":1e3, "txt":"Tank shell material density"},
    "fuel_pressure": {"unit":"bar", "mag":1e2, "txt":"Maximum over pressure of the fuel in the tank"},
    "structure_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Ratio of the total volume of the tank used for tank structure"},
    "surface_mass": {"unit":"kg/m2", "mag":1e1, "txt":"Mass of tank structure per tank surface unit"},
    "cantilever_volume": {"unit":"m3", "mag":1e1, "txt":"Tank volume outside of the main body"},
    "central_volume": {"unit":"m3", "mag":1e1, "txt":"Tank volume inside of the main body"},
    "fuel_cantilever_cg": {"unit":"m", "mag":1e1, "txt":"Position of the CG of the fuel volume which is outside of the main body"},
    "fuel_central_cg": {"unit":"m", "mag":1e1, "txt":"Position of the CG of the fuel volume which is in the main body"},
    "fuel_total_cg": {"unit":"m", "mag":1e1, "txt":"Position of the CG of the total fuel volume"},
    "max_volume": {"unit":"m3", "mag":1e1, "txt":"Total available volume for fuel"},
    "mfw_volume_limited": {"unit":"kg", "mag":1e3, "txt":"Maximum fuel mass according to available volume"},
    "volume": {"unit":"m3", "mag":1e1, "txt":"Volume of the component"},
    "wing_axe_c": {"unit":"m", "mag":1e0, "txt":"Wing chord length at component axis span wise position"},
    "wing_axe_x": {"unit":"m", "mag":1e1, "txt":"X wise position of the wing chord at component axis span wise position"},
    "fuel_max_fwd_cg": {"unit":"m", "mag":1e1, "txt":"Maximum forward position of the fuel"},
    "fuel_max_fwd_mass": {"unit":"kg", "mag":1e3, "txt":"Fuel mass corresponding to mximum forward position"},
    "fuel_max_bwd_cg": {"unit":"m", "mag":1e1, "txt":"Maximum backward position of the fuel"},
    "fuel_max_bwd_mass": {"unit":"kg", "mag":1e3, "txt":"Fuel mass corresponding to mximum backward position"},
    "n_engine": {"unit":"int", "mag":1e0, "txt":"Algebraïc number of engine"},
    "reference_thrust": {"unit":"daN", "mag":1e5, "txt":"Engine reference thrust, thrust(sea level, ISA+15, Mach 0.25)/0.8"},
    "reference_power": {"unit":"kW", "mag":1e5, "txt":"Engine reference power, power(sea level, ISA+15, Mach 0.25)/0.8"},
    "reference_offtake": {"unit":"kW", "mag":1e4, "txt":"Refrence power offtake for design"},
    "MTO": {"unit":"no_dim", "mag":1e0, "txt":"Max Takeoff rating factor"},
    "MCN": {"unit":"no_dim", "mag":1e0, "txt":"Maxi Continuous rating factor"},
    "MCL": {"unit":"no_dim", "mag":1e0, "txt":"Max Climb rating factor"},
    "MCR": {"unit":"no_dim", "mag":1e0, "txt":"Max Cruise rating factor"},
    "FID": {"unit":"no_dim", "mag":1e0, "txt":"Flight idle rating factor"},
    "tune_factor": {"unit":"no_dim", "mag":1e0, "txt":"Factor on unitary engine thrust to match with reference thrust definition"},
    "engine_bpr": {"unit":"no_dim", "mag":1e0, "txt":"Reference By Pass Ratio of the engine"},
    "core_thrust_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Reference ratio of the total thrust delivered by the core"},
    "efficiency_fan": {"unit":"no_dim", "mag":1e0, "txt":"Classical fan efficiency"},
    "efficiency_prop": {"unit":"no_dim", "mag":1e0, "txt":"Propeller like fan efficiency Thrust.Speed/shaft_power"},
    "rating": {"unit":"string", "mag":3, "txt":"Engine rating name ('MTO','MCN','MCL','MCR','FID'"},
    "power": {"unit":"kW", "mag":1e3, "txt":"Engine input power (before controller)"},
    "motor_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Electric motor efficiency"},
    "controller_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Electric controller efficiency"},
    "generator_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Electric generator efficiency"},
    "rectifier_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Electric rectifier efficiency"},
    "wiring_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Electric wiring efficiency"},
    "motor_pw_density": {"unit":"kW/kg", "mag":1e1, "txt":"Electric motor power density"},
    "nacelle_pw_density": {"unit":"kW/kg", "mag":1e1, "txt":"Electric nacelle power density"},
    "controller_pw_density": {"unit":"kW/kg", "mag":1e1, "txt":"Electric controller power density"},
    "generator_pw_density": {"unit":"kW/kg", "mag":1e1, "txt":"Electric generator power density"},
    "rectifier_pw_density": {"unit":"kW/kg", "mag":1e1, "txt":"Electric rectifier power density"},
    "wiring_pw_density": {"unit":"kW/kg", "mag":1e1, "txt":"Electric wiring power density"},
    "cooling_pw_density": {"unit":"kW/kg", "mag":1e1, "txt":"Cooling power density"},
    "battery_density": {"unit":"kg/m3", "mag":1e3, "txt":"Battery density"},
    "battery_energy_density": {"unit":"kWh/kg", "mag":1e1, "txt":"Battery energy density"},
    "power_chain_efficiency": {"unit":"no_dim", "mag":1e0, "txt":"Global efficiency of the electric power chain"},
    "hub_width": {"unit":"m", "mag":1e0, "txt":"Fan hub diameter"},
    "fan_width": {"unit":"m", "mag":1e0, "txt":"Fan diameter"},
    "nozzle_width": {"unit":"m", "mag":1e0, "txt":"Nozzle diameter"},
    "nozzle_area": {"unit":"m2", "mag":1e0, "txt":"Nozzle area"},
    "fuel_flow": {"unit":"kg/s", "mag":1e0, "txt":"Fuel flow"},
    "fuel_heat": {"unit":"MJ/kg", "mag":1e1, "txt":"Fuel heating value"},
    "sec": {"unit":"kW/daN", "mag":1e0, "txt":"Specific Energy Consumption"},
    "sfc": {"unit":"kg/daN/h", "mag":1e0, "txt":"Specific Fuel Consumption"},
    "nei": {"unit":"int", "mag":1e0, "txt":"Number of engine inoperative, typically 0 or 1"},
    "disa": {"unit":"degK", "mag":1e1, "txt":"Temperature shift versus ISA conditions"},
    "altp": {"unit":"ft", "mag":1e4, "txt":"Target pressure altitude"},
    "mach": {"unit":"mach", "mag":1e0, "txt":"Mach number"},
    "thrust": {"unit":"daN", "mag":1e3, "txt":"Engine thrust"},
    "thrust_opt": {"unit":"daN", "mag":1e3, "txt":"Required engine thrust"},
    "kfn_opt": {"unit":"no_dim", "mag":1e0, "txt":"Ratio required thrust over effective engine thrust"},
    "speed_mode": {"unit":"string", "mag":4, "txt":"Constant CAS : 'cas' or constant Mach : 'mach'"},
    "kmtow": {"unit":"no_dim", "mag":1e0, "txt":"Ratio of MTOW defining the aircraft weight"},
    "kmlw": {"unit":"no_dim", "mag":1e0, "txt":"Ratio of MLW defining the aircraft weight"},
    "kvs1g": {"unit":"no_dim", "mag":1e0, "txt":"Applicable ratio of stalling speed at 1g"},
    "kvs1g_eff": {"unit":"no_dim", "mag":1e0, "txt":"Effective ratio of stalling speed at 1g"},
    "s2_min_path": {"unit":"%", "mag":1e0, "txt":"Minimum trajectory slope at 35ft for take off"},
    "s2_path": {"unit":"%", "mag":1e0, "txt":"Airplane trajectory slope at 35ft during take off"},
    "tofl_req": {"unit":"m", "mag":1e3, "txt":"Maximum take off field length required in given conditions"},
    "tofl_eff": {"unit":"m", "mag":1e3, "txt":"Effective take off field length in given conditions"},
    "app_speed_req": {"unit":"kt", "mag":1e2, "txt":"Maximum approach speed required in given conditions"},
    "app_speed_eff": {"unit":"kt", "mag":1e2, "txt":"Effective approach speed in given conditions"},
    "path_req": {"unit":"%", "mag":1e0, "txt":"Minimum trajectory slope required in given conditions"},
    "path_eff": {"unit":"%", "mag":1e0, "txt":"Effective trajectory slope in given conditions"},
    "vz_req": {"unit":"ft/min", "mag":1e2, "txt":"Minimum vertical speed required in given conditions"},
    "vz_eff": {"unit":"ft/min", "mag":1e2, "txt":"Effective vertical speed in given conditions"},
    "cx_correction": {"unit":"no_dim", "mag":1e-4, "txt":"Additive correction on airplane total drag coefficient"},
    "cruise_lodmax": {"unit":"no_dim", "mag":1e1, "txt":"Maximum lift over Drag ratio in cruise"},
    "cz_cruise_lodmax": {"unit":"no_dim", "mag":1e0, "txt":"Lift coefficient for maximum lift over Drag ratio in cruise"},
    "hld_conf_clean": {"unit":"no_dim", "mag":1e0, "txt":"Deflection parameter corresponding to high lift devices retracted, typically 0"},
    "czmax_conf_clean": {"unit":"no_dim", "mag":1e0, "txt":"Maximum lift coefficient with high lift devices retracted"},
    "hld_conf_to": {"unit":"no_dim", "mag":1e0, "txt":"Deflection parameter corresponding to high lift devices in take off configuration"},
    "czmax_conf_to": {"unit":"no_dim", "mag":1e0, "txt":"Maximum lift coefficient with high lift devices in take off configuration"},
    "hld_conf_ld": {"unit":"no_dim", "mag":1e0, "txt":"Deflection parameter corresponding to high lift devices in landing configuration, typically 1"},
    "czmax_conf_ld": {"unit":"no_dim", "mag":1e0, "txt":"Maximum lift coefficient with high lift devices in landing configuration"},
    "mtow": {"unit":"kg", "mag":1e5, "txt":"Maximum Take Off Weight"},
    "mzfw": {"unit":"kg", "mag":1e5, "txt":"Maximum Zero Fuel Weight"},
    "mlw": {"unit":"kg", "mag":1e5, "txt":"Maximum Landing Weight"},
    "owe": {"unit":"kg", "mag":1e5, "txt":"Operational Weight Empty"},
    "mwe": {"unit":"kg", "mag":1e5, "txt":"Manufacturer Weight Empty"},
    "mfw": {"unit":"kg", "mag":1e5, "txt":"Maximum Fuel Weight"},
    "hld_conf": {"unit":"no_dim", "mag":1e0, "txt":"Current deflection parameter corresponding to high lift devices configuration"},
    "v2": {"unit":"kt", "mag":1e2, "txt":"Airplane speed in CAS at 35ft during take off"},
    "mach2": {"unit":"mach", "mag":1e0, "txt":"Airplane speed in Mach number at 35ft during take off"},
    "limit": {"unit":"string", "mag":2, "txt":"Active limit during take off, can be 'fl':field length, 's2':second segment path"},
    "mach_opt": {"unit":"mach", "mag":1e0, "txt":"Optimal Mach number according to requirement"},
    "ttc_req": {"unit":"min", "mag":1e1, "txt":"Maximum time to climb required in given conditions"},
    "ttc_eff": {"unit":"min", "mag":1e1, "txt":"Effective time to climb in given conditions"},
    "altp1": {"unit":"ft", "mag":1e4, "txt":"Starting pressure altitude for climb trajectory, typically 1500ft"},
    "cas1": {"unit":"kt", "mag":1e2, "txt":"Calibrated Air Speed below altp1 during climb trajectory"},
    "altp2": {"unit":"ft", "mag":1e4, "txt":"Transtion pressure altitude from cas1 to cas2, typically 10000ft"},
    "cas2": {"unit":"kt", "mag":1e2, "txt":"Calibrated Air Speed above altp2"},
    "tow": {"unit":"kg", "mag":1e5, "txt":"Mission take off weight"},
    "range": {"unit":"NM", "mag":1e3, "txt":"Mission range"},
    "diversion_range": {"unit":"NM", "mag":1e1, "txt":"Range of diversion mission for reserve fuel evaluation"},
    "reserve_fuel_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Fraction of mission fuel for reserve fuel evaluation"},
    "reserve_enrg_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Fraction of mission energy for reserve evaluation"},
    "holding_time": {"unit":"min", "mag":1e1, "txt":"Holding duration for reserve fuel evaluation"},
    "payload": {"unit":"kg", "mag":1e4, "txt":"Mission payload"},
    "time_block": {"unit":"h", "mag":1e1, "txt":"Mission block time"},
    "enrg_block": {"unit":"kWh", "mag":1e4, "txt":"Mission block energy"},
    "fuel_block": {"unit":"kg", "mag":1e4, "txt":"Mission block fuel"},
    "enrg_reserve": {"unit":"kWh", "mag":1e4, "txt":"Mission reserve energy"},
    "fuel_reserve": {"unit":"kg", "mag":1e3, "txt":"Mission reserve fuel"},
    "enrg_total": {"unit":"kWh", "mag":1e4, "txt":"Mission total energy"},
    "battery_mass": {"unit":"kg", "mag":1e3, "txt":"Required battery mass"},
    "fuel_total": {"unit":"kg", "mag":1e4, "txt":"Mission total fuel"},
    "irp": {"unit":"year", "mag":1e1, "txt":"Interest recovery period"},
    "period": {"unit":"year", "mag":1e1, "txt":"Utilisation period"},
    "interest_rate": {"unit":"%", "mag":1e0, "txt":"Interest rate"},
    "labor_cost": {"unit":"$/h", "mag":1e1, "txt":"Labor cost"},
    "utilisation": {"unit":"int", "mag":1e3, "txt":"Number of flights per year"},
    "engine_price": {"unit":"M$", "mag":1e1, "txt":"Price of one engine"},
    "gear_price": {"unit":"M$", "mag":1e1, "txt":"Price of landing gears"},
    "frame_price": {"unit":"M$", "mag":1e1, "txt":"Price of the airframe"},
    "energy_price": {"unit":"$/kWh", "mag":1e0, "txt":"Energy price"},
    "fuel_price": {"unit":"$/gal", "mag":1e0, "txt":"Fuel price"},
    "battery_price": {"unit":"$/kg", "mag":1e0, "txt":"Battery price"},
    "frame_cost": {"unit":"$/trip", "mag":1e3, "txt":"Airframe maintenance cost"},
    "engine_cost": {"unit":"$/trip", "mag":1e3, "txt":"Engine maintenance cost"},
    "cockpit_crew_cost": {"unit":"$/trip", "mag":1e3, "txt":"Cockpit crew cost"},
    "cabin_crew_cost": {"unit":"$/trip", "mag":1e3, "txt":"Cabin crew cost"},
    "landing_fees": {"unit":"$/trip", "mag":1e3, "txt":"Landing fees"},
    "navigation_fees": {"unit":"$/trip", "mag":1e3, "txt":"Navigation fees"},
    "catering_cost": {"unit":"$/trip", "mag":1e3, "txt":"Catering cost"},
    "pax_handling_cost": {"unit":"$/trip", "mag":1e3, "txt":"Passenger handling cost"},
    "ramp_handling_cost": {"unit":"$/trip", "mag":1e3, "txt":"Ramp handling cost"},
    "std_op_cost": {"unit":"$/trip", "mag":1e4, "txt":"Standard operating cost"},
    "cash_op_cost": {"unit":"$/trip", "mag":1e4, "txt":"Cash operating cost"},
    "direct_op_cost": {"unit":"$/trip", "mag":1e4, "txt":"Direct operating cost"},
    "fuel_cost": {"unit":"$/trip", "mag":1e4, "txt":"Fuel cost"},
    "aircraft_price": {"unit":"M$", "mag":1e5, "txt":"Aircraft price"},
    "total_investment": {"unit":"M$", "mag":1e4, "txt":"Total investmenent"},
    "interest": {"unit":"$/trip", "mag":1e3, "txt":"Interest"},
    "insurance": {"unit":"$/trip", "mag":1e3, "txt":"Insurance"},
    "depreciation": {"unit":"$/trip", "mag":1e3, "txt":"Depreciation"},
    "CO2_metric": {"unit":"kg/km/m0.48", "mag":1e0, "txt":"Fuel efficiency metric"},
    "CO2_index": {"unit":"g/kg", "mag":1e3, "txt":"Mass of carbon dioxide emitted per kg of fuel"},
    "CO_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of carbon monoxide emitted per kg of fuel"},
    "H2O_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of water emitted per kg of fuel"},
    "SO2_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of sulfur dioxide emitted per kg of fuel"},
    "NOx_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of nitrogen oxide emitted per kg of fuel"},
    "HC_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of unburnt hydrocarbon emitted per kg of fuel"},
    "sulfuric_acid_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of sulfuric acid emitted per kg of fuel"},
    "nitrous_acid_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of nitrous acid emitted per kg of fuel"},
    "nitric_acid_index": {"unit":"g/kg", "mag":1e-5, "txt":"Mass of nitric acid emitted per kg of fuel"},
    "soot_index": {"unit":"int", "mag":1e12, "txt":"Number of soot particles emitted per kg of fuel"},
    "ktow": {"unit":"no_dim", "mag":1e0, "txt":"Ratio of TOW defining the aircraft weight"},
    "crz_altp": {"unit":"ft", "mag":1e4, "txt":"Cruise altitude for design"},
    "crz_sar": {"unit":"NM/kg", "mag":1e0, "txt":"Cruise specific air range related to fuel"},
    "crz_esar": {"unit":"NM/kW", "mag":1e0, "txt":"Cruise specific air range related to power"},
    "crz_cz": {"unit":"no_dim", "mag":1e0, "txt":"Cruise lift coefficient"},
    "crz_lod": {"unit":"no_dim", "mag":1e0, "txt":"Cruise lift to drag ratio"},
    "crz_thrust": {"unit":"kN", "mag":1e0, "txt":"Total cruise thrust"},
    "crz_throttle": {"unit":"no_dim", "mag":1e0, "txt":"Cruise throttle versus MCR"},
    "crz_sfc": {"unit":"kg/daN/h", "mag":1e0, "txt":"Cruise specific fuel consumption"},
    "crz_sec": {"unit":"kW/daN", "mag":1e0, "txt":"Cruise specific energy consumption"},
    "max_sar_altp": {"unit":"ft", "mag":1e0, "txt":"Altitude of specific air range"},
    "max_sar": {"unit":"NM/kg", "mag":1e0, "txt":"Maximum specific air range"},
    "max_sar_cz": {"unit":"no_dim", "mag":1e0, "txt":"Lift coefficient for maximum specific air range"},
    "max_sar_lod": {"unit":"no_dim", "mag":1e0, "txt":"Lift to drag ratio for maximum specific air range"},
    "max_sar_thrust": {"unit":"kN", "mag":1e0, "txt":"Total thrust for maximum specific air range"},
    "max_sar_throttle": {"unit":"no_dim", "mag":1e0, "txt":"Throttle versus MCR for maximum specific air range"},
    "max_sar_sfc": {"unit":"kg/daN/h", "mag":1e0, "txt":"Specific fuel consumption for maximum specific air range"},
    "max_esar_altp": {"unit":"ft", "mag":1e0, "txt":"Altitude of specific air range"},
    "max_esar": {"unit":"NM/kg", "mag":1e0, "txt":"Maximum specific air range"},
    "max_esar_cz": {"unit":"no_dim", "mag":1e0, "txt":"Lift coefficient for maximum specific air range"},
    "max_esar_lod": {"unit":"no_dim", "mag":1e0, "txt":"Lift to drag ratio for maximum specific air range"},
    "max_esar_thrust": {"unit":"kN", "mag":1e0, "txt":"Total thrust for maximum specific air range"},
    "max_esar_throttle": {"unit":"no_dim", "mag":1e0, "txt":"Throttle versus MCR for maximum specific air range"},
    "max_esar_sec": {"unit":"kg/daN/h", "mag":1e0, "txt":"Specific energy consumption for maximum specific air range"}
}


def isNaN(num):
    return num != num

def convert_to_original_type(lst, orig_seq):
    if isinstance(orig_seq, tuple):
        return tuple(lst)
    elif isinstance(orig_seq, np.ndarray):
        return np.array(lst)
    else:
        return lst

def convert_to_scientific_notation(value, dec_format=STANDARD_FORMAT):
    str_value = format(value, "".join((".", str(dec_format), "E")))
    return str_value

def to_user_format(value, dec_format=STANDARD_FORMAT):
    if isinstance(value, (tuple, list, np.ndarray)):
        lst = list(value)
        for i in np.arange(len(lst)):
            lst[i] = to_user_format(lst[i], dec_format)
        return str(convert_to_original_type(lst, value)).replace("'", "")
    elif isinstance(value, dict):
        for k, v in value.items():
            value[k] = to_user_format(v, dec_format)
        return str(value).replace("'", "")
    elif isinstance(value, (float, np.float64)):
        if isNaN(value):
            return value
        if value == 0. or value == -0.:
            return format(value, "".join((".", str(dec_format), "f")))
        else:
            V = abs(value)
            if V > 1:
                if V > 1e6:
                    return convert_to_scientific_notation(value,dec_format)
                correction_factor = 1e-4  # to correct 10^n values format
                nb_dec = int(max((0, (dec_format + 1) - np.ceil(np.log10(V + correction_factor)))))
            else:
                if V < 1e-3:
                    return convert_to_scientific_notation(value,dec_format)
                nb_dec = int((dec_format - 1) - np.floor(np.log10(V)))
            return format(value, "".join((".", str(nb_dec), "f")))
    elif isinstance(value, int) and abs(value) > 1e6:
        return convert_to_scientific_notation(value,dec_format)
    else:
        return value

class MarilibIO(object):
    """A collection of Input and Ouput functions for MARILib objects.

    1) Human readable format uses a *JSON-like* encoding and decoding functions adapted to MARILib objects.
    2) Binary exact copies uses pickle.

    """
    def __init__(self):
        self.datadict = DATA_DICT # default units and variable description dict

    def marilib_encoding(self, o):
        """Default encoding function for MARILIB objects of non primitive types (int,float,list,string,tuple,dict)

        * Skips `self.aircraft` entries to avoid circular reference
        * Converts numpy array to list.
        * Convert to default units described in `DATA_DICT`
        * Add a short description of each variable, found in `DATA_DICT`

        :param o: the object to encode
        :return: the attribute dict

        """
        if isinstance(o, type(np.array([]))):  # convert numpy arrays to list
            return o.tolist()

        json_dict = o.__dict__  # Store the public attributes, raises an AttributeError if no __dict__ is found.
        try:
            del json_dict['aircraft']  # Try to delete the 'aircraft' entry to avoid circular reference
        except KeyError:
            pass  # There was no aircraft entry => nothing to do

        for key,value in json_dict.items():
            if key in self.datadict.keys():  # if entry found in DATA_DICT, add units and docstring
                unit = self.datadict[key]['unit']
                text = self.datadict[key]['txt']
                try:
                    json_dict[key] = [convert_to(unit,value), f"({unit}) {text}"]
                except KeyError:
                    json_dict[key] = [value, f"WARNING: conversion to ({unit}) failed. {text}"]
                    print("WARNING : unknwon unit "+str(unit))
            else:
                pass

        return json_dict

    def to_string(self,marilib_object,datadict=None):
        """Build a human readable string output of the object in a JSON-like format.
        It uses :meth:`marilib_encoding` to serialize the object into a dictionary.

        :param marilib_object: the object to print
        :param datadict: a dictionary that give the unit and a description of each variable.
            Example of datadict::

                datadict = {
                             "MTO": {"unit":"no_dim", "txt":"Max Takeoff rating factor"},
                             "cg": {"unit":"m", "txt":"Position of the center of gravity"}
                            }

            .. note::
                by default it uses the value given during the last call. If no previous call, default is `DATA_DICT`
                defined in `marilib.aircraft.tool.module_read_write.py`

        :return: a customized JSON-like formatted string

            .. warning::
                Numpy arrays and lists are rewritten on one line only.It does not strictly follow the JSON standard

        """
        # Set the new data dict if one is provided
        if (datadict is not None):
            self.datadict = datadict

        json_string = json.dumps(marilib_object, indent=4, default=self.marilib_encoding)
        output = re.sub(r'\[\s+', '[', json_string)  # remove spaces after an opening bracket
        output = re.sub(r'(?<!\}),\s+(?!\s*".*":)', ', ', output)  # remove spaces after comma not followed by ".*":
        output = re.sub(r'\s+\]', ']', output)  # remove white spaces before a closing bracket
        # reformat floats
        float_pattern = re.compile(r'\d+\.\d+')
        floats = (float(f) for f in float_pattern.findall(output))  # detect all floats in the json string
        output_parts = float_pattern.split(output)  # split output around floats
        output = ''  # init output
        # number_format = "%0."+int(STANDARD_FORMAT)+"g" # TODO: allow format change
        for part,val in zip(output_parts[:-1],floats):  # reconstruct output with proper float format
            output += part + "%0.6g" % float(val)  # reformat with 6 significant digits max
        return output + output_parts[-1]

    def from_string(self,json_string):
        """Parse a JSON string into a dict.

        :param json_string: the string to parse
        :return: dictionary
        """
        return json.loads(json_string)

    def to_json_file(self,aircraft_object,filename,datadict=None):
        """Save a MARILib object in a human readable format:
        The object is serialized into a customized JSON-like string.

        :param marilib_object: the object to save
        :param filename: name of the file. Ex: myObjCollection/marilib_obj.json
        :param datadict: argument for to_string(). The default datadict is DATA_DICT.
        :return: None
        """
        marilib_object = copy.deepcopy(aircraft_object)
        try:  # Add .json extension if necessary
            last_point_position = filename.rindex(r'\.')
            filename = filename[:last_point_position]+".json"
        except ValueError:  # pattern not found
            filename = filename + ".json"
        with open(filename,'w') as f:
            f.write(self.to_string(marilib_object,datadict=datadict))
        return None

    def to_binary_file(self,obj,filename):
        """Save the obj as a binary file .pkl

        :param obj: the object to save
        :param filename: the path
        :return: None
        """
        try:  # Add .pkl extension if not specified
            last_point_position = filename.rindex(r'\.')
            filename = filename[:last_point_position]+".pkl"
        except ValueError:  # pattern not found
            filename = filename + ".pkl"

        with open(filename,'wb') as f:
            pickle.dump(obj,f)
        return

    def from_binary_file(self, filename):
        """Load a .pkl file as a python object

        :param filename: the binary filepath
        :return: the object
        """
        with open(filename, 'rb') as f:
            obj = pickle.load(f)

        return obj
