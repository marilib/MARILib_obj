#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry, Nicolas Monrolin
"""

import numpy as np
import json as json
import pickle as pickle
import re

from aircraft.tool import unit

STANDARD_FORMAT = 4

data_dict = {
    "body_type": {"unit":"string", "mag":8, "txt":"Type of main body, 'fuselage' or 'blended'"},
    "wing_type": {"unit":"string", "mag":6, "txt":"Type of lifting body, 'classic' or 'blended'"},
    "wing_attachment": {"unit":"string", "mag":4, "txt":"Position of wing attachment, 'low' or 'high'"},
    "stab_architecture": {"unit":"string", "mag":6, "txt":"Type of stabilizers, 'classic', 't_tail' or 'h_tail'"},
    "tank_architecture": {"unit":"string", "mag":10, "txt":"Type of tank, 'wing_box', 'piggy_back' or 'pods'"},
    "number_of_engine": {"unit":"string", "mag":6, "txt":"Number of engine, 'twin' or 'quadri'"},
    "nacelle_attachment": {"unit":"string", "mag":4, "txt":"Position of nacelle attachment, 'wing', 'pod' or 'rear'"},
    "power_architecture": {"unit":"string", "mag":4, "txt":"Propulsive architecture, 'tf', 'tp', 'pte1', 'ef1', 'ep1'"},
    "energy_source": {"unit":"string", "mag":9, "txt":"Type energy storage, 'kerosene', 'methane', 'liquid_h2', '700bar_h2' or 'battery'"},
    "width": {"unit":"m", "mag":1e0, "txt":"Width of the component"},
    "height": {"unit":"m", "mag":1e0, "txt":"Height of the component"},
    "length": {"unit":"m", "mag":1e1, "txt":"Length of the component"},
    "tail_cone_length": {"unit":"m", "mag":1e1, "txt":"Length of the tapered rear part of the main body"},
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
    "n_pax_ref": {"unit":"", "mag":1e2, "txt":"Reference number of passenger for design"},
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
    "mac_c": {"unit":"m", "mag":1e1, "txt":"Mean aerodynamic chord length of the lifting surface"},
    "toc": {"unit":"no_dim", "mag":1e0, "txt":"Thickness to chord ratio of the lifting surface"},
    "volume_factor": {"unit":"no_dim", "mag":1e0, "txt":"Volume coefficient of the stabilization surface"},
    "anchor_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Relative position of the reference point of the stabilization surface"},
    "lever_arm": {"unit":"m", "mag":1e1, "txt":"Lever arm of the stabilization surface, 25%cma wing to 25% cma surface"},
    "fuel_density": {"unit":"kg/m3", "mag":1e3, "txt":"Density of the used fuel"},
    "structure_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Ratio of the total volume of the tank used for tank structure"},
    "surface_mass": {"unit":"kg/m2", "mag":1e1, "txt":"Mass of tank structure per tank surface unit"},
    "cantilever_volume": {"unit":"m3", "mag":1e1, "txt":"Tank volume outside of the main body"},
    "central_volume": {"unit":"m3", "mag":1e1, "txt":"Tank volume inside of the main body"},
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
    "reference_thrust": {"unit":"daN", "mag":1e5, "txt":"Engine reference thrust, thrus(sea level, ISA+15, Mach 0.25)/0.8"},
    "reference_offtake": {"unit":"kW", "mag":1e4, "txt":"Refrence power offtake for design"},
    "rating_factor": {"unit":"no_dim", "mag":1e0, "txt":"Factor on max thrust corresponding to each rating"},
    "tune_factor": {"unit":"no_dim", "mag":1e0, "txt":"Factor on unitary engine thrust to match with reference thrust definition"},
    "engine_bpr": {"unit":"no_dim", "mag":1e0, "txt":"Reference By Pass Ratio of the engine"},
    "core_thrust_ratio": {"unit":"no_dim", "mag":1e0, "txt":"Reference ratio of the total thrust delivered by the core"},
    "efficiency_prop": {"unit":"no_dim", "mag":1e0, "txt":"Propeller like fan efficiency Thrust.Speed/shaft_power"},
    "rating": {"unit":"string", "mag":3, "txt":"Engine rating name ('MTO','MCN','MCL','MCR','FID'"},
    "fuel_flow": {"unit":"kg/s", "mag":1e0, "txt":"Fuel flow"},
    "sfc": {"unit":"kg/daN/h", "mag":1e0, "txt":"Specific Fuel Consumption"},
    "nei": {"unit":"int", "mag":1e0, "txt":"Number of engine inoperative, typically 0 or 1"},
    "disa": {"unit":"degK", "mag":1e1, "txt":"Temperature shift versus ISA conditions"},
    "altp": {"unit":"ft", "mag":1e4, "txt":"Target pressure altitude"},
    "mach": {"unit":"mach", "mag":1e0, "txt":"Mach number"},
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
    "mass": {"unit":"kg", "mag":1e5, "txt":"Current aircraft weight"},
    "hld_conf": {"unit":"no_dim", "mag":1e0, "txt":"Current deflection parameter corresponding to high lift devices configuration"},
    "v2": {"unit":"kt", "mag":1e2, "txt":"Airplane speed at 35ft during take off"},
    "limit": {"unit":"string", "mag":2, "txt":"Active limit during take off, can be 'fl':field length, 's2':second segment path"},
    "path_eff": {"unit":"%", "mag":1e0, "txt":"Effective trajectory slope in given conditions"},
    "mach_opt": {"unit":"mach", "mag":1e0, "txt":"Optimal Mach number according to requirement"},
    "ttc_req": {"unit":"min", "mag":1e1, "txt":"Maximum time to climb required in given conditions"},
    "ttc_eff": {"unit":"min", "mag":1e1, "txt":"Effective time to climb in given conditions"},
    "altp1": {"unit":"ft", "mag":1e4, "txt":"Starting pressure altitude for climb trajectory, typically 1500ft"},
    "cas1": {"unit":"kt", "mag":1e2, "txt":"Calibrated Air Speed below altp1 during climb trajectory"},
    "altp2": {"unit":"ft", "mag":1e4, "txt":"Transtion pressure altitude from cas1 to cas2, typically 10000ft"},
    "cas2": {"unit":"kt", "mag":1e2, "txt":"Calibrated Air Speed above altp2"},
    "tow": {"unit":"kg", "mag":1e5, "txt":"Mission take off weight"},
    "payload": {"unit":"kg", "mag":1e4, "txt":"Mission payload"},
    "time_block": {"unit":"h", "mag":1e1, "txt":"Mission block time"},
    "fuel_block": {"unit":"kg", "mag":1e4, "txt":"Mission block fuel"},
    "fuel_reserve": {"unit":"kg", "mag":1e3, "txt":"Mission reserve fuel"},
    "fuel_total": {"unit":"", "mag":1e4, "txt":"Mission total fuel"}
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

class MarilibIO():
    """A collection of Input and Ouput functions for MARILib objects.
    1) Human readable format uses a *JSON-like* encoding and decoding functions adapted to MARILib objects.
    """

    def marilib_encoding(self, o):
        """Default encoding function for MARILIB objects of non primitive types (int,float,list,string,tuple,dict)
        Raises an `AttributeError` if te object has no __dict__ attribute.
        Skips 'aircraft' entries to avoid circular reference and converts numpy array to list.
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
            if key in data_dict.keys():  # if entry found in data_dict, add units and docstring
                unit = data_dict[key]['unit']
                text = data_dict[key]['txt']
                if isinstance(value,float):
                    json_dict[key] = [PrettyFloat(value), f"({unit}) {text}"]
                else:
                    json_dict[key] = [value, f"({unit}) {text}"]
            else:
                pass

        return json_dict

    def to_string(self,marilib_object):
        """Customized Json pretty string of the object
        It uses marilib_encoding() to parse objects into dict.
        .. warning::
            Numpy arrays and list of numbers are rewritten on one line only, which is not JSON standard
        :param marilib_object: the object to print
        :return: a customized JSON-like formatted string
        """
        json_string = json.dumps(marilib_object, indent=4, default=self.marilib_encoding)
        output = re.sub(r'\[\s+', '[', json_string)  # remove spaces after an opening bracket
        output = re.sub(r'(?<!\}),\s+(?!\s*".*":)', ', ', output)  # remove spaces after comma not followed by ".*":
        output = re.sub(r'\s+\]', ']', output)  # remove white spaces before a closing bracket
        # reformat floats
        float_pattern = re.compile(r'\d+\.\d+')
        floats = (float(f) for f in float_pattern.findall(output))  # detect all floats in the json string
        output_parts = float_pattern.split(output)  # split output around floats
        output = ''  # init output
        for part,val in zip(output_parts[:-1],floats):  # reconstruct output with proper float format
            output += part + "%0.4g" % float(val)  # reformat with 4 significant digits max
        return output + output_parts[-1]

    def from_string(self,json_string):
        """Parse a JSON string into a dict.
        :param json_string: the string to parse
        :return: dictionary
        """
        return json.loads(json_string)

    def to_json_file(self,marilib_object,filename):
        """Save a MARILib object in a human readable format:
        The object is serialized into a customized JSON-like string.
        :param marilib_object: the object to save
        :param filename: name of the file. Ex: myObjCollection/marilib_obj.json
        :return: None
        """
        try:  # Add .json extension if necessary
            last_point_position = filename.rindex(r'\.')
            filename = filename[:last_point_position]+".json"
        except ValueError:  # pattern not found
            filename = filename + ".json"
        with open(filename,'w') as f:
            f.write(self.to_string(marilib_object))
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

class PrettyFloat(float):
    """Subclass of Float for pretty printing in text files
    """
    def __repr__(self):  # overwright the default representation method
        return "%d" % self



