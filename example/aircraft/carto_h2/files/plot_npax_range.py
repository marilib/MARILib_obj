#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

"""Plot all aircrafts on Npax_range diagram"""

from marilib.utils.read_write import MarilibIO
import matplotlib.pyplot as plt
import glob
from data_to_capture import get_path


aircraft_types = ["A220-100", "A320-200neo", "A330-800", "A350-900"]
configurations = ["reference_opt", "rear_tank", "pod_tank", "piggyback_tank","engined_pod_tank","engined_piggyback_tank"]
points         = ["max_range","req_range","pax_range_trade","pax_range_trade_600NM","ref_600NM"]
techno_levels  = ["soa","2030"]

all_files = glob.glob("*.json")
io = MarilibIO()

# Initialize the dict
color_categories = aircraft_types
x = {key: [] for key in color_categories} # initialize x values sorted by category
y = {key: [] for key in color_categories} # initialize y values sorted by category
x_name = "Design range" # name of the variable to extract from "data"
y_name = "Nominal seat count"

for f in all_files:
    print(f)
    ac = io.from_json_file(f,skip_list=["bnd_layer"])

    # check the aircraft category from the file name
    cat = ""
    for c in color_categories:
        if c in f:
            cat = c
        else:
            pass
    if cat == "":
        raise ValueError(f"Category of {f} could not be determined")

    # extract x,y values and store them in the dict in the right category
    exec("x_value="+get_path(x_name))
    x[cat].append(x_value)
    exec("y_value=" + get_path(y_name))
    y[cat].append(y_value)

for c in color_categories:
    plt.scatter(x[c],y[c],label=c)

plt.grid()
plt.legend()
plt.show()





