#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 20 20:20:20 2020
@author: Cong Tam DO, Thierry DRUOT
"""

import numpy as np

from marilib.utils import unit

import matplotlib.patches as patches



def rect(l,w,x,y,a,c):
    lk = unit.km_m(l)
    wk = unit.km_m(w)
    origin = (unit.km_m(x-(0.5*l)*np.cos(a)),
              unit.km_m(y-(0.5*l)*np.sin(a)))
    ad = unit.deg_rad(a)
    ptch = patches.Rectangle(origin, lk, wk, angle=ad, color=c)
    return ptch
