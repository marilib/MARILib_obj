#!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on August 20 20:20:20 2020
@author: Cong Tam DO, Thierry DRUOT
"""

import numpy as np
from scipy.optimize import fsolve
from marilib.utils.math import lin_interp_1d, maximize_1d

from marilib.utils import unit, earth

from marilib.airport.aircraft import AirplaneCategories, Aircraft

from marilib.airport.airport import Airport




def distance_to_center(r0, r1, d):
    """Compute the mean distance between a focal point and a cloud of points arranged into a ring

    :param r0: internal radius of the ring (can be zero)
    :param r1: external radius of the ring (must be greater than r0)
    :param n: Number of points within the radius
    :param m: Number of points in the circonference
    :param d: distance between the center of th edisk and the focal point
    :return: Mean distance per travel
    """
    n,m = 100,100
    dist = 0.
    for i in range(n):
        for j in range(m):
            dist += np.sqrt( ((r0+(r1-r0)*((1+i)/n))*np.cos(2*np.pi*(j/m)) - d)**2 + ((r0+(r1-r0)*((1+i)/n))*np.sin(2*np.pi*(j/m)))**2 )
    return dist/(m*n)

r0 = 0.5
r1 = 1.

for d in np.linspace(0., 5, 50):
    print(d, distance_to_center(r0, r1, d))
