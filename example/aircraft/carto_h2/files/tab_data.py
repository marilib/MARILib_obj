#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Department, ENAC
"""

"""Extract some data from JSON files and returns a Latex table"""

from marilib.utils.read_write import MarilibIO
from marilib.utils.unit import convert_to
from data_to_capture import get_path,get_unit,get_format
import pandas as pd
from glob import glob



def tabular(files, datas, names=[], caption="", label=""):
    """Returns a formatted text tabular with custom column separator.
    filenames : the list of filenames for each line
    datanames : the data list to extract and plot in columns. (see 'data' for exact denomination)
    sep : the column separator
    endline: line ending"""

    io = MarilibIO()
    datadict = {"":[""]} # add a first column empty with first empty line
    for d in datas: # initialise a dict with unit in the first element (first line of tabular)
        datadict[d] = [f"({get_unit(d)})"]

    names = iter(names)
    for f in files:
        ac = io.from_json_file(f,skip_list=["bnd_layer"])
        try:
            datadict[""].append(next(names)) # try to fill with the name list
        except StopIteration:
            datadict[""].append(f[:-5].replace("_"," ")) # filename without extension and white spaces

        for d in datas:
            value = eval(get_path(d)) # import and convert to unit
            value = convert_to(get_unit(d),value)
            value = get_format(d) %(value)
            datadict[d].append(value)


    output = pd.DataFrame(datadict).to_latex(index=False)
    # output = output.replace("\\begin{tabular}","\\begin{tabularx}{\\textwidth}") # use tabularx package
    # output = output.replace("\\end{tabular}","\\end{tabularx}") # use tabularx package

    output = "\\begin{table}[H]\n\centering\n"+output
    output = output+caption
    output = output + "\label{tab:"+label+"}\n\end{table}\n"

    output = output.replace("\n\\midrule","")
    output = output.split("\n")
    output.insert(6,"\\midrule")

    return "\n".join(output)


def tabular_transpose(files, datas, names=[], caption="", label=""):
    """Returns a formatted text tabular with custom column separator.
    filenames : the list of filenames for each column
    datanames : the data list to extract and plot in line. (see 'data' for exact denomination)
    sep : the column separator
    endline: line ending"""

    io = MarilibIO()
    datadict = {"":[]} # add a first column empty with first empty line
    for d in datas: # fill the column with datanames and units
        exec(f"datadict[\"\"].append(d+\" ({get_unit(d)})\")") # data name and unit

    names = iter(names)
    for f in files: # initialise a dict with unit in the first element (first line of tabular)
        try:
            datadict[next(names)] = []
        except StopIteration:
            datadict[f[:-5].replace("_"," ")] = [] # filename without extension and white spaces

        ac = io.from_json_file(f,skip_list=["bnd_layer"])
        for d in datas:
            last_column_name = list(datadict.keys())[-1]
            value = eval(get_path(d))
            value = convert_to(get_unit(d),value)
            value = get_format(d) %(value)
            datadict[last_column_name].append(value)

    output = pd.DataFrame(datadict).to_latex(index=False)
    # output = output.replace("\\begin{tabular}","\\begin{tabularx}{\\textwidth}") # use tabularx package
    # output = output.replace("\\end{tabular}","\\end{tabularx}") # use tabularx package

    output = "\\begin{table}[H]\n\centering\n" + output
    output = output + "\caption{" + caption + "}\n"
    output = output + "\label{tab:" + label + "}\n\end{table}\n"


    return output

# Example
#-----------------------------------------------------------------------------------------------------------------------
# Build the list of files to read
# files = glob("A220-100*max_range_soa.json")
# files.insert(0,"A220-100_reference_opt.json")
#
# datas = ["Design range","Nominal seat count","Fuselage length"]
#
# print(tabular(files,datas))


print("\section{Short range airplanes}")
#-----------------------------------------------------------------------------------------------------------------------
files = ["A220-100_reference_opt.json",
         "A220-100_rear_tank_max_range_soa.json",
         "A220-100_rear_tank_req_range_2030.json"]

names = ["Reference",
         "Rear Tank SOA",
         "Rear Tank 2030"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length", "MTOW", "OEW", "Body aspect ratio"]

caption = "Short Range Rear tank architecture versus Reference"
label = "SR_ref"
print(tabular_transpose(files, datas, names, caption, label))


# Short range comparison SOA
#-----------------------------------------------------------------------------------------------------------------------
files = ["A220-100_rear_tank_ref_600NM_soa.json",
         "A220-100_pod_tank_ref_600NM_soa.json",
         "A220-100_engined_pod_tank_ref_600NM_soa.json",
         "A220-100_piggyback_tank_ref_600NM_soa.json",
         "A220-100_engined_piggyback_tank_ref_600NM_soa.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Short Range Hydrogen architecture comparison SOA"
label = "SR_SOA"
print(tabular_transpose(files, datas, names, caption, label))


# Short range comparison 2030
#-----------------------------------------------------------------------------------------------------------------------
files = ["A220-100_rear_tank_req_range_2030.json",
         "A220-100_pod_tank_req_range_2030.json",
         "A220-100_engined_pod_tank_req_range_2030.json",
         "A220-100_piggyback_tank_req_range_2030.json",
         "A220-100_engined_piggyback_tank_req_range_2030.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Short Range Hydrogen architecture comparison 2030"
label = "SR_2030"
print(tabular_transpose(files, datas, names, caption, label))






print("\section{Medium range airplanes}")
#-----------------------------------------------------------------------------------------------------------------------
files = ["A320-200neo_reference_opt.json",
         "A320-200neo_rear_tank_max_range_soa.json",
         "A320-200neo_rear_tank_max_range_2030.json"]

names = ["Reference",
         "Rear Tank SOA",
         "Rear Tank 2030"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length", "MTOW", "OEW", "Body aspect ratio"]

caption = "Medium Range Rear tank architecture versus Reference"
label = "MR_ref"
print(tabular_transpose(files, datas, names, caption, label))


# Medium range comparison SOA
#-----------------------------------------------------------------------------------------------------------------------
files = ["A320-200neo_rear_tank_max_range_soa.json",
         "A320-200neo_pod_tank_ref_600NM_soa.json",
         "A320-200neo_engined_pod_tank_ref_600NM_soa.json",
         "A320-200neo_piggyback_tank_max_range_soa.json",
         "A320-200neo_engined_piggyback_tank_max_range_soa.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Medium Range Hydrogen architecture comparison SOA"
label = "MR_SOA"
print(tabular_transpose(files, datas, names, caption, label))


# Medium range comparison 2030
#-----------------------------------------------------------------------------------------------------------------------
files = ["A320-200neo_rear_tank_max_range_2030.json",
         "A320-200neo_pod_tank_max_range_2030.json",
         "A320-200neo_engined_pod_tank_max_range_2030.json",
         "A320-200neo_piggyback_tank_max_range_2030.json",
         "A320-200neo_engined_piggyback_tank_max_range_2030.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Medium Range Hydrogen architecture comparison 2030"
label = "MR_2030"
print(tabular_transpose(files, datas, names, caption, label))






print("\section{Long range airplanes}")
#-----------------------------------------------------------------------------------------------------------------------
files = ["A330-800_reference_opt.json",
         "A330-800_rear_tank_max_range_soa.json",
         "A330-800_rear_tank_max_range_2030.json"]

names = ["Reference",
         "Rear Tank SOA",
         "Rear Tank 2030"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length", "MTOW", "OEW", "Body aspect ratio"]

caption = "Long Range Rear tank versus Reference"
label = "LR_ref"
print(tabular_transpose(files, datas, names, caption, label))


# Long range comparison SOA
#-----------------------------------------------------------------------------------------------------------------------
files = ["A330-800_rear_tank_max_range_soa.json",
         "A330-800_pod_tank_ref_600NM_soa.json",
         "A330-800_engined_pod_tank_ref_600NM_soa.json",
         "A330-800_piggyback_tank_max_range_soa.json",
         "A330-800_engined_piggyback_tank_max_range_soa.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Long Range Hydrogen architecture comparison SOA"
label = "LR_SOA"
print(tabular_transpose(files, datas, names, caption, label))


# Long range comparison 2030
#-----------------------------------------------------------------------------------------------------------------------
files = ["A330-800_rear_tank_max_range_2030.json",
         "A330-800_pod_tank_max_range_2030.json",
         "A330-800_engined_pod_tank_max_range_2030.json",
         "A330-800_piggyback_tank_max_range_2030.json",
         "A330-800_engined_piggyback_tank_max_range_2030.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Long Range Hydrogen architecture comparison 2030"
label = "LR_2030"
print(tabular_transpose(files, datas, names, caption, label))


#-----------------------------------------------------------------------------------------------------------------------
files = ["A330-800_engined_piggyback_tank_ultra_max_range_2030.json",
         "A330-800_engined_piggyback_tank_super_ultra_max_range_2030.json"]

names = ["MTOW ref", "MTOW ref x 1.3"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Long Range (ultimate) Piggyback tank BLI 2030"
label = "LR_2030_ultimate"
print(tabular_transpose(files, datas, names, caption, label))





print("\section{Ultra long range airplanes}")
#-----------------------------------------------------------------------------------------------------------------------
files = ["A350-900_reference_opt.json",
         "A350-900_rear_tank_max_range_soa.json",
         "A350-900_rear_tank_max_range_2030.json",
         "A350-900_rear_tank_ultra_max_range_2030.json",
         "A350-900_rear_tank_ultra_max_capacity_2030.json"]

names = ["Reference",
         "Rear Tank SOA",
         "Rear Tank 2030",
         "Max range RT 2030",
         "Max capa RT 2030"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length", "MTOW", "OEW", "Body aspect ratio"]

caption = "Ultra Long Range Rear tank versus Reference"
label = "ULR_ref"
print(tabular_transpose(files, datas, names, caption, label))


# Ultra Long range comparison SOA
#-----------------------------------------------------------------------------------------------------------------------
files = ["A350-900_rear_tank_max_range_soa.json",
         "A350-900_pod_tank_ref_800NM_soa.json",
         "A350-900_engined_pod_tank_ref_800NM_soa.json",
         "A350-900_piggyback_tank_max_range_soa.json",
         "A350-900_engined_piggyback_tank_max_range_soa.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Ultra Long Range Hydrogen architecture comparison SOA"
label = "ULR_SOA"
print(tabular_transpose(files, datas, names, caption, label))


# Ultra Long range comparison 2030
#-----------------------------------------------------------------------------------------------------------------------
files = ["A350-900_rear_tank_max_range_2030.json",
         "A350-900_pod_tank_max_range_2030.json",
         "A350-900_engined_pod_tank_max_range_2030.json",
         "A350-900_piggyback_tank_max_range_2030.json",
         "A350-900_engined_piggyback_tank_max_range_2030.json"]

names = ["Rear Tank",
         "Pod Tank",
         "Pod Tank BLI",
         "Piggyback",
         "Piggyback BLI"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Ultra Long Range Hydrogen architecture comparison 2030"
label = "ULR_2030"
print(tabular_transpose(files, datas, names, caption, label))


#-----------------------------------------------------------------------------------------------------------------------
files = ["A350-900_engined_piggyback_tank_ultra_max_capacity_2030.json",
         "A350-900_engined_piggyback_tank_ultra_max_range_2030.json"]

names = ["PB BLI req capa", "PB BLI max range"]

datas = ["Nominal seat count", "Front seats count", "Design range", "Maximum range", "Fuselage length",
         "Fuselage width", "Tank length", "Tank diameter", "MTOW", "OEW"]

caption = "Long Range (ultimate) Piggyback tank BLI 2030"
label = "ULR_2030_ultimate"
print(tabular_transpose(files, datas, names, caption, label))

