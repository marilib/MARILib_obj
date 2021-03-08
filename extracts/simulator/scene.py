#!/usr/bin/env python3
"""
Created on Thu Jan 20 20:20:20 2020

@author: DRUOT Thierry
"""


import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import time

from marilib.utils import unit
from controler import MyJoystick


def Rx(phi):
    return np.array([[1, 0, 0],
                     [0, np.cos(phi), -np.sin(phi)],
                     [0, np.sin(phi), np.cos(phi)]])

def Ry(theta):
    return np.array([[np.cos(theta), 0, np.sin(theta)],
                     [0, 1, 0],
                     [-np.sin(theta), 0, np.cos(theta)]])

def Rz(psi):
    return np.array([[np.cos(psi), -np.sin(psi), 0],
                     [np.sin(psi), np.cos(psi), 0],
                     [0, 0, 1]])


def get_controls():
    global ctrl
    dx,dl,dm,dn = ctrl.get_axis()
    return dx,dl,dm,dn


def get_arrow(p,vec):
    global vec_length
    return p[0], p[1], p[2], vec_length*vec[0], vec_length*vec[1], vec_length*vec[2]


def update_frame(j):
    global quiver_x, quiver_y, quiver_z
    global loc

    print(j)

    dx,dl,dm,dn = get_controls()

    psi = -np.pi * dn
    theta = -np.pi * dm
    phi = -np.pi * dl

    att = np.matmul(np.matmul(Rz(psi), Ry(theta)), Rx(phi))

    quiver_x.remove()
    quiver_y.remove()
    quiver_z.remove()

    quiver_x = ax.quiver(*get_arrow(loc,att[:,0]))
    quiver_y = ax.quiver(*get_arrow(loc,att[:,1]))
    quiver_z = ax.quiver(*get_arrow(loc,att[:,2]))


# Set the joystick
#---------------------------------------------------------------------------
ctrl = MyJoystick()


# Set the scene
#---------------------------------------------------------------------------

fig, ax = plt.subplots(subplot_kw=dict(projection="3d"))

ax.set_xlim(-250, 250)
ax.set_ylim(-250, 250)
ax.set_zlim(0, 500)

ax.invert_yaxis()
ax.invert_zaxis()

vec_length = 50

loc = np.array([0,0,250])

psi = unit.rad_deg(0)
theta = unit.rad_deg(0)
phi = unit.rad_deg(0)

att = np.matmul(np.matmul(Rz(psi), Ry(theta)), Rx(phi))

quiver_x = ax.quiver(*get_arrow(loc,att[:,0]))
quiver_y = ax.quiver(*get_arrow(loc,att[:,1]))
quiver_z = ax.quiver(*get_arrow(loc,att[:,2]))


anim = FuncAnimation(fig, update_frame, interval=50, repeat=True, blit=False)



plt.show()


