#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

"""Plot all aircrafts on Npax_range diagram"""

from marilib.utils.read_write import MarilibIO
import marilib.utils.unit as un
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from glob import glob
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
configurations = {"reference_opt"          : '*',
                  "rear_tank"              : 's',
                  "engined_pod_tank"       : 'P',
                  "engined_piggyback_tank" : '^',
                  "pod_tank"               : 'X',
                  "piggyback_tank"         : '<'}

# Pour identifier des paires:
# les avions "pax_range_trade" vont forcement de paire avec soit un "ref_range", "req_range" ou "max_range".
points = ["max_range","req_range","pax_range_trade","pax_range_trade_600NM","ref_600NM"]

# The techno level is determined by filled and non filled markers
techno_levels  = {"soa" : "None",
                  "2030": "k"}


# ------ START
all_files = glob("*.json")
io = MarilibIO()

# Name of the xy axis variable. The denomination is in data_to_capture.py
x_name = "Design range"
y_name = "Nominal seat count"



def set_marker_style(filename,default_markersize = 10):
    """Choose the marker style from the filename.
    Returns a dict of keyword arguments (kwargs) to set the marker style."""
    kwargs = {'markersize': default_markersize}  # initialize the dict containing the plot options
    for a in segments.keys():  # check the aircraft category from the file name
        if a in filename:
            kwargs['c'] = segments[a]  # marker color
            break  # stop at first match

    for c in configurations.keys():  # check the aircraft configuration from the file name
        if c in filename:
            kwargs['marker'] = configurations[c]
            break  # stop at first match WARNING : the order in the "configurations" list is important !!!

    if "soa" in filename:
        if techno_levels["soa"] == "None":
            kwargs['markerfacecolor'] = "None"  # empty marker face color
    elif "2030" in filename:
        if techno_levels["2030"] == "None":
            kwargs['markerfacecolor'] = "None"  # empty marker face color

    return kwargs

# Initialize a dict that will contain all xy coordinates and kwargs for marker style.
# aicrafts = { filename0 : {'x' : x_value , 'y': y_value , 'kwargs': kwargs},
#              filename1 : ... }
aircrafts = {}
# Iterate over all files to extract x_value and y_value and set the marker style
for f in all_files:
    ac = io.from_json_file(f,skip_list=["bnd_layer"])

    kwargs = set_marker_style(f)

    # extract x,y values and store them in the dict in the right category
    exec("x_value=" + get_path(x_name))
    exec("y_value=" + get_path(y_name))

    aircrafts[f] = {'x':x_value, 'y':y_value, 'kwargs': kwargs}


# -------- PLOT
plt.figure(figsize=(12,7))
for name,point in aircrafts.items():
    x = un.convert_to(get_unit(x_name),point['x'])
    y = un.convert_to(get_unit(y_name),point['y'])

    if True:# "A220" in name:
        plt.plot(x,y,**point['kwargs'])
    if  "reference_opt" in name: # add name for reference points
        plt.text(x,y,name[:4])


pairs = []
for f in all_files:
    if "pax_range_trade" in f: # this file has a pair file somewhere
        potential_pair_files = (f.replace("pax_range_trade",sub) for sub in ["max_range","req_range","ref_range","ref"])
        match = False
        while not match:
            try:
                pf = next(potential_pair_files) # iterate over potential files
                if pf in all_files:
                    pairs.append((f,pf))
                    match = True
            except StopIteration:
                print(f"!!--Pas de pair trouvé pour {f}-!!")
                break
            #end


# Join Pax-range trades points with a line
for p in pairs:
    x = un.convert_to(get_unit(x_name),[aircrafts[p[0]]['x'],aircrafts[p[1]]['x']])
    y = un.convert_to(get_unit(y_name),[aircrafts[p[0]]['y'],aircrafts[p[1]]['y']])
    plt.plot(x,y,'-',color=aircrafts[p[0]]['kwargs']['c'])



# Axis label
plt.xlabel(f"{x_name} ({get_unit(x_name)})")
plt.ylabel(f"{y_name} ({get_unit(y_name)})")

# Shrink current axis by 20% for the legend on the side
box = plt.gca().get_position()
plt.gca().set_position([box.x0, box.y0, box.width * 0.8, box.height])

# legends on the side
conf_legend = [ Line2D([0],[0],marker=marker,label=key,color='k',linestyle="none",ms=10) for key,marker in configurations.items() ]
leg0 = plt.legend(handles = conf_legend,loc = 'upper left',bbox_to_anchor=(1, 1))

seg_legend = [ Line2D([0],[0],marker='o',label=key,color=color,linestyle="none",ms=10) for key,color in segments.items() ]
leg1 = plt.legend(handles = seg_legend,loc = 'upper left', bbox_to_anchor=(1, 0.5))

tech_legend = [ Line2D([0],[0],marker='s',label=key,color='k',mfc=fill,linestyle="none",ms=10) for key,fill in techno_levels.items() ]
leg2 = plt.legend(handles = tech_legend,loc = 'upper left',bbox_to_anchor=(1, 0.2))

plt.gca().add_artist(leg0)
plt.gca().add_artist(leg1)
plt.gca().add_artist(leg2)

plt.grid()
plt.show()





