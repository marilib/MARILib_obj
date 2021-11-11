#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

"""Extract some data from JSON files and returns a Latex table"""

from marilib.utils.read_write import MarilibIO
from marilib.utils.unit import convert_to
from data_to_capture import get_path,get_unit
import pandas as pd
from glob import glob



def tabular(files, datas):
    """Returns a formatted text tabular with custom column separator.
    filenames : the list of filenames for each line
    datanames : the data list to extract and plot in columns. (see 'data' for exact denomination)
    sep : the column separator
    endline: line ending"""

    io = MarilibIO()
    datadict = {"":[""]} # add a first column empty with first empty line
    for d in datas: # initialise a dict with unit in the first element (first line of tabular)
        datadict[d] = [f"({get_unit(d)})"]

    for f in files:
        ac = io.from_json_file(f,skip_list=["bnd_layer"])
        datadict[""].append(f[:-5].replace("_"," ")) # filename without extension and white spaces
        for d in datas:
            exec(f"datadict[d].append({get_path(d)})") # import and convert to unit
            datadict[d][-1] = convert_to(get_unit(d),datadict[d][-1])


    output = pd.DataFrame(datadict).to_latex(index=False)
    output = output.replace("\\begin{tabular}","\\begin{tabularx}{\\textwidth}") # use tabularx package
    output = output.replace("\\end{tabular}","\\end{tabularx}") # use tabularx package
    output = output.replace("\n\\midrule","")
    output = output.split("\n")
    output.insert(4,"\\midrule")


    return "\n".join(output)


# Build the list of files to read
files = glob("A220-100*max_range_soa.json")
files.insert(0,"A220-100_reference_opt.json")

datas = ["Design range","Nominal seat count","Fuselage length"]

print(tabular(files,datas))


