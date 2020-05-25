#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot, Weichang LYU
"""

import numpy as np

import pandas
import csv


file_name = "Airport_code_source.csv"
code_df = pandas.read_csv(file_name, encoding="Windows-1252")


file_name = "Country_name.csv"
name_df = pandas.read_csv(file_name, encoding="Windows-1252")


for j,key in enumerate(code_df["Country"]):
    k = next(iter(name_df[name_df["French"]==key].index), key)
    code_df["Country"][j] = name_df["English"][k]


code_df.to_csv("Airport_code.csv", sep=',', escapechar='\\', quoting=csv.QUOTE_ALL, index=None)


