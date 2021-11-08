#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: Conceptual Airplane Design & Operations (CADO team)
         Nicolas PETEILH, Pascal ROCHES, Nicolas MONROLIN, Thierry DRUOT
         Aircraft & Systems, Air Transport Departement, ENAC
"""

"""Extract all the data listed in data_to_capture from JSON files and returns a Latex table"""

from marilib.utils.read_write import MarilibIO
import glob


aircraft_types = ["A220-100", "A320-200neo", "A330-800", "A350-900"]
configurations = ["reference_opt", "rear_tank", "pod_tank", "piggyback_tank","engined_pod_tank","engined_piggyback_tank"]
points         = ["max_range","req_range","pax_range_trade","pax_range_trade_600NM","ref_600NM"]
techno_levels  = ["soa","2030"]

all_files = glob.glob("*.json")
io = MarilibIO()





