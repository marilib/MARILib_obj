#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 22 22:09:20 2020
@author: Thierry Druot
"""

import numpy as np
from scipy.optimize import fsolve, least_squares

import pandas
import matplotlib.pyplot as plt

from context import unit, math

from network.pico_design.design_model import Aircraft


# ======================================================================================================
# Test
# ------------------------------------------------------------------------------------------------------
ac = Aircraft(npax=150., range=unit.m_NM(3000.), mach=0.78)

ac.payload_range()
