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

from marilib.utils.unit import convert_to, convert_from

from marilib.aircraft.tool.dictionary import DATA_DICT

STANDARD_FORMAT = 6


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

    1) Human readable format : uses *JSON-like* encoding and decoding functions adapted to MARILib objects.
    2) Binary exact copy : uses pickle.

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

        json_dict = copy.deepcopy(o.__dict__) #  Store the public attributes, raises an AttributeError if no __dict__ is found.
        try:
            del json_dict['aircraft']  # Try to delete the 'aircraft' entry to avoid circular reference
        except KeyError:
            pass  # There was no aircraft entry => nothing to do

        for key,value in json_dict.items():
            if key in self.datadict.keys():  # if entry found in DATA_DICT, add units and docstring
                try:
                    unit = self.datadict[key]['unit']
                    text = self.datadict[key]['txt']
                    json_dict[key] = [convert_to(unit, value), unit, text]
                except KeyError:
                    json_dict[key] = [value, f"WARNING: conversion to ({unit}) failed. {text}"]
                    print("WARNING : unknown unit "+str(unit))
            elif key == "name":
                pass
            # TODO : check that the key is in DATA_DICT
            elif type(value) in (int,float,bool,str,list,tuple): # if not a dict, should be in the data_dict
                print("Salut Thierry, tu as oublie de mettre a jour le DATA_DICT: %s n'existe pas !" %key)

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

    def from_json_file(self,filename):
        """Reads a JSON file and parse it into a dict.

        .. warning::
                The JSON format is not an exact copy of the original object. in the following sequence, `aircraft2` is
                a dictionary which contains truncated values of the original `aircraft1`::

                    aircraft1 =  Aircraft()
                    io = MarilibIO()
                    io.to_json_file(aircraft1,"my_plane")
                    aircraft2 = io.from_json_file("my_plane")

        :param filename: the file to parse
        :return: mydict : a customized dictionary, where values can be accessed like object attributes.

                        Example::

                            aircraft2 = io.from_json_file("my_plane")
                            assert(aircraft2['name'] == aircraft2.name)

        """
        try:  # Add .json extension if necessary
            last_point_position = filename.rindex(r'.')
            filename = filename[:last_point_position]+".json"
        except ValueError as err:  # dot pattern not found
            print(err)
            filename = filename + ".json"

        with open(filename, 'r') as f:
            mydict = MyDict(json.loads(f.read()))

        return mydict

    def to_json_file(self,object,filename,datadict=None):
        """Save a MARILib object in a human readable format:
        The object is serialized into a customized JSON-like string.

        :param object: the object to save
        :param filename: name of the file, optional. Ex: myObjCollection/marilib_obj.json
        :param datadict: argument for to_string(). The default datadict is DATA_DICT.
        :return: None
        """
        try:  # Add .json extension if necessary
            last_point_position = filename.rindex(r'\.')
            filename = filename[:last_point_position]+".json"
        except ValueError:  # pattern not found
            filename = filename + ".json"
        with open(filename,'w') as f:
            f.write(self.to_string(object,datadict=datadict))
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


class MyDict(dict):
    """A customized dictionary class to convert a MARILib json dict to an "aircraft-like" object. Converts all data to
    SI units.
    Attributes can be accessed by two manners::

        obj = mydict['airframe']['wing']

    is equivalent to::

        obj = mydict.airframe.wing

    """
    def __init__(self,*args,**kwargs):
        super(MyDict,self).__init__(*args,**kwargs)
        self.__dict__ = self
        for key,val in self.__dict__.items(): # recursively converts all items of type 'dict' to 'MyDict'
            if isinstance(val, dict):
                self.__dict__[key] = MyDict(val)
            elif isinstance(val, list): # if list, extract value and convert to SI unit
                self.__dict__[key] = convert_from(val[1],val[0])
            elif key == 'name': # skip the 'name' entry which is of type string
                pass
            elif isinstance(val, type(None)):
                print("WARNING in MyDict: %s is 'None'" %key)
                self.__dict__[key] = None
            else:
                raise AttributeError("Unknown type, should be list or dict but type of %s is %s" % (key, type(val)))

