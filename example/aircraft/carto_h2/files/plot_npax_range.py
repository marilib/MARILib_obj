#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

"""Plot all aircrafts on Npax_range diagram"""

from marilib.utils.read_write import MarilibIO
import marilib.utils.unit as un
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import glob
from data_to_capture import get_path,get_unit

# Aircraft segment is determined by colors.
# The first four default colors of matplotlib are used.
colors = [c for (k,c) in enumerate(plt.rcParams['axes.prop_cycle'].by_key()['color']) if k<4]
segments = {"A220-100"   : colors[0],
            "A320-200neo": colors[1],
            "A330-800"   : colors[2],
            "A350-900"   : colors[3]}

# The configuration is determined by the marker symbol.
# WARNING : the order in the "configurations" dict is important:
# It avoids confusion between normal and "engined" configurations using pattern matching in filenames.
configurations = {"reference_opt"          : 'o',
                  "rear_tank"              : 's',
                  "engined_pod_tank"       : "*",
                  "engined_piggyback_tank" : "P",
                  "pod_tank"               : 'd',
                  "piggyback_tank"         : '^'}

# Pour identifier des paires:
# les avions "pax_range_trade" vont forcement de paire avec soit un "ref_range", "req_range" ou "max_range".
points = ["max_range","req_range","pax_range_trade","pax_range_trade_600NM","ref_600NM"]

# The techno level is determined by filled and non filled markers
techno_levels  = {"soa" : "None",
                  "2030": "k"}


# ------ START
all_files = glob.glob("*.json")
io = MarilibIO()

# Name of the xy axis variable. The denomination is in data_to_capture.py
x_name = "Design range"
y_name = "Nominal seat count"

# Initialize a dict that will contain:
# file name, (pax,range), color, marker, marker face color
aircrafts = {}

for f in all_files:
    print(f)
    ac = io.from_json_file(f,skip_list=["bnd_layer"])

    kwargs = {'markersize':10} # initialize the dict containing the plot options
    for a in segments.keys():  # check the aircraft category from the file name
        if a in f:
            kwargs['c'] = segments[a] # marker color
            break # stop at first match

    for c in configurations.keys(): # check the aircraft configuration from the file name
        if c in f:
            kwargs['marker'] = configurations[c]
            break # stop at first match WARNING : the order in the "configurations" list is important !!!

    if "soa" in f:
        if techno_levels["soa"]=="None":
            kwargs['markerfacecolor'] = "None" # empty marker face color
    elif "2030" in f:
        if techno_levels["2030"]=="None":
            kwargs['markerfacecolor'] = "None" # empty marker face color

    print('-> ' + str(kwargs))

    # extract x,y values and store them in the dict in the right category
    exec("x_value=" + get_path(x_name))
    exec("y_value=" + get_path(y_name))

    aircrafts[f] = {'x':x_value, 'y':y_value, 'kwargs': kwargs}

plt.figure(figsize=(12,7))
for name,point in aircrafts.items():
    x = un.convert_to(get_unit(x_name),point['x'])
    y = un.convert_to(get_unit(y_name),point['y'])

    if True:# "A220" in name:
        plt.plot(x,y,**point['kwargs'])
    if  "reference_opt" in name:
        plt.text(x,y,name[:4])

plt.xlabel(f"{x_name} ({get_unit(x_name)})")
plt.ylabel(f"{y_name} ({get_unit(y_name)})")


conf_legend = [ Line2D([0],[0],marker=marker,label=key,color='k',linestyle="none") for key,marker in configurations.items() ]
leg0 = plt.legend(handles = conf_legend,loc = 'upper left',bbox_to_anchor=(1, 1))

seg_legend = [ Line2D([0],[0],marker='o',label=key,color=color,linestyle="none") for key,color in segments.items() ]
leg1 = plt.legend(handles = seg_legend,loc = 'upper left', bbox_to_anchor=(1, 0.5))

tech_legend = [ Line2D([0],[0],marker='o',label=key,color='k',mfc=fill,linestyle="none") for key,fill in techno_levels.items() ]
leg2 = plt.legend(handles = tech_legend,loc = 'upper left',bbox_to_anchor=(1, 0.2))

plt.gca().add_artist(leg0)
plt.gca().add_artist(leg1)
plt.gca().add_artist(leg2)

# Shrink current axis by 20%
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.grid()
plt.show()





